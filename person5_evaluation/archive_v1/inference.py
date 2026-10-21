"""Adapters for the existing P3 raw-input pipelines and P4 classifiers."""
from __future__ import annotations

import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from common import MODEL_NAMES, P1, P2, P3, P4, require

sys.path.insert(0, str(P2))
import aki_preprocessing as ap


def dense_to_explicit_csr(x):
    """Same representation as P4, retaining every structured zero and NaN."""
    x = np.asarray(x)
    require(x.ndim == 2, "Structured input must be two-dimensional")
    n_rows, n_cols = x.shape
    return sparse.csr_matrix(
        (x.ravel(), np.tile(np.arange(n_cols, dtype=np.int32), n_rows),
         np.arange(0, (n_rows + 1) * n_cols, n_cols, dtype=np.int64)), shape=x.shape,
    )


def rebuild_tree_text(structured, original_text):
    n_structured = structured.shape[1]
    require(original_text.shape[0] == len(structured), "Text/structured row counts differ")
    explicit = dense_to_explicit_csr(structured)
    rebuilt = sparse.hstack([explicit, original_text[:, n_structured:]], format="csr")
    block = rebuilt[:, :n_structured]
    require(block.nnz == structured.size, "Explicit structured entries were removed")
    require(np.array_equal(block.toarray(), structured, equal_nan=True), "Structured values changed")
    require(np.count_nonzero(block.data == 0) == np.count_nonzero(structured == 0), "Zero values changed")
    require(np.isnan(block.data).sum() == np.isnan(structured).sum(), "NaN values changed")
    require((rebuilt[:, n_structured:] != original_text[:, n_structured:]).nnz == 0, "Text values changed")
    return rebuilt


def load_partition(partition):
    require(partition in {"validation", "test"}, "Only validation/test inference is supported")
    folder = P2 / "outputs/matrices"
    labels = pd.read_csv(folder / f"labels_{partition}.csv")
    raw = pd.read_csv(folder / f"X_raw_{partition}.csv")
    predictors = pd.read_csv(P1 / "predictor_columns.csv")["predictor"].tolist()
    require(predictors == ap.PREDICTORS, "Predictor whitelist differs from P2 module")
    require(raw.columns.tolist() == ["stay_id"] + predictors, "Raw input schema differs")
    require(np.array_equal(raw.stay_id, labels.stay_id), "Raw inputs and labels have different ID order")
    require(not labels.stay_id.duplicated().any(), "Duplicate labelled stay_id")
    require(labels.partition.eq(partition).all(), "Wrong partition labels")
    manifest = pd.read_csv(P2 / "outputs/split_manifest.csv")
    require(not manifest.stay_id.duplicated().any(), "Duplicate manifest stay_id")
    require(manifest.groupby("subject_id").partition.nunique().max() == 1, "Patient crosses partitions")
    expected = manifest.loc[manifest.partition.eq(partition), ["stay_id", "subject_id", "partition"]]
    pd.testing.assert_frame_equal(
        labels[expected.columns].sort_values("stay_id").reset_index(drop=True),
        expected.sort_values("stay_id").reset_index(drop=True),
    )
    structured = np.load(folder / f"tree_structured_{partition}.npy", allow_pickle=False)
    original_text = sparse.load_npz(folder / f"tree_text_{partition}.npz").tocsr()
    names = pd.read_csv(P2 / "outputs/preprocessors/tree_structured_feature_names.csv")["feature"].tolist()
    text_names = pd.read_csv(P2 / "outputs/preprocessors/tree_text_feature_names.csv")["feature"].tolist()
    require(names == text_names[:len(names)], "Text matrix structured prefix does not match")
    require(structured.shape == (len(labels), len(names)), "Structured matrix dimensions differ")
    require(original_text.shape == (len(labels), len(text_names)), "Text matrix dimensions differ")
    require((len(names), len(text_names)) == (19, 2791), "Unexpected feature dimensions")
    age_gender_indices = [names.index(n) for n in ["num__age", "cat__gender_M"]]
    return {
        "labels": labels, "raw": raw[predictors], "structured": structured,
        "text": rebuild_tree_text(structured, original_text),
        "age_gender": structured[:, age_gender_indices],
        "age_gender_indices": age_gender_indices,
    }


def predict_partition(partition):
    data = load_partition(partition)
    labels = data["labels"]
    probabilities = {}
    diagnostics = []
    for name in MODEL_NAMES:
        owner = P3 if name in MODEL_NAMES[:4] else P4
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(owner / f"outputs/models/{name}.joblib")
            if owner == P3:
                x = data["raw"]
                require(hasattr(model, "named_steps") and "preprocess" in model.named_steps, "P3 pipeline schema differs")
            else:
                # Limit inference threads, without changing learned model parameters.
                model.set_params(n_jobs=1)
                x = data["age_gender"] if name == "age_gender_xgboost" else data["structured"] if name == "structured_xgboost" else data["text"]
                require(model.n_features_in_ == x.shape[1], f"{name}: feature dimension differs")
            classes = list(model.classes_)
            require(set(classes) == {0, 1}, f"{name}: unexpected classes")
            probability = np.asarray(model.predict_proba(x)[:, classes.index(1)], dtype=np.float64)
        require(probability.shape == (len(labels),), f"{name}: wrong probability shape")
        require(np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all(), f"{name}: invalid probability")
        probabilities[name] = probability
        diagnostics.append({"model_name": name, "input_shape": list(x.shape),
                            "warnings": sorted(set(str(w.message) for w in caught))})
    long = pd.concat([
        pd.DataFrame({"stay_id": labels.stay_id, "subject_id": labels.subject_id,
                      "target": "aki_label", "observed_label": labels.aki_label,
                      "predicted_probability": probabilities[name], "model_name": name, "partition": partition})
        for name in MODEL_NAMES
    ], ignore_index=True)
    wide = labels[["stay_id", "subject_id"]].copy()
    wide["y_true"] = labels.aki_label
    for name in MODEL_NAMES:
        wide[f"y_prob_{name}"] = probabilities[name]
    return data, long, wide, diagnostics


def reference_validation_predictions(labels):
    aligned = {}
    for folder, names in [(P3, MODEL_NAMES[:4]), (P4, MODEL_NAMES[4:])]:
        exported = pd.read_csv(folder / "outputs/validation_predictions.csv", float_precision="round_trip")
        require(set(exported.model_name) == set(names), "Unexpected models in handoff predictions")
        require(not exported.duplicated(["stay_id", "model_name"]).any(), "Duplicate handoff prediction")
        if "target" not in exported:
            exported["target"] = "aki_label"  # P4 source code explicitly uses aki_label.
        require(exported.target.eq("aki_label").all() and exported.partition.eq("validation").all(), "Handoff target/partition differs")
        for name in names:
            group = exported.loc[exported.model_name.eq(name)].set_index("stay_id")
            require(set(group.index) == set(labels.stay_id), f"{name}: validation encounter set differs")
            group = group.loc[labels.stay_id]
            require(np.array_equal(group.subject_id, labels.subject_id), f"{name}: subject IDs differ")
            require(np.array_equal(group.observed_label, labels.aki_label), f"{name}: labels differ")
            aligned[name] = group.predicted_probability.to_numpy(dtype=float)
    return aligned
