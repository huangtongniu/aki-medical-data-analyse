"""Adapters for raw-input LR pipelines and Person 4's six-model inference API."""
from __future__ import annotations

import importlib.util
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from common import (MODEL_NAMES, OLD, P1, P2, P3_PRIMARY, P3_SECONDARY, P4, ROOT,
                    TARGETS, read_json, relative, require, sha256)

sys.path.insert(0, str(P2))
import aki_preprocessing as ap

spec = importlib.util.spec_from_file_location("p5_supplied_p4_inference", P4 / "P4_inference.py")
p4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p4)
dense_to_explicit_csr = p4.dense_to_explicit_csr


def rebuild_tree_text(structured, original):
    rebuilt = p4.prepare_p4_text_matrix(structured, original)
    width = structured.shape[1]
    block = rebuilt[:, :width]
    require(block.nnz == structured.size, "P4 explicit structured entries were removed")
    require(np.array_equal(block.toarray(), structured, equal_nan=True), "P4 structured values changed")
    require(np.count_nonzero(block.data == 0) == np.count_nonzero(structured == 0), "P4 zero count changed")
    require(np.isnan(block.data).sum() == np.isnan(structured).sum(), "P4 NaN count changed")
    require((rebuilt[:, width:] != original[:, width:]).nnz == 0, "P4 TF-IDF values changed")
    return rebuilt


def validate_secondary_provenance():
    metadata = read_json(P3_SECONDARY / "outputs/model_metadata.json")
    require(metadata["target"] == TARGETS[1], "Secondary P3 metadata target mismatch")
    for recorded, expected in metadata["source_sha256"].items():
        path = P1 / "predictor_columns.csv" if recorded == "aki_handoff/predictor_columns.csv" else ROOT / recorded
        require(sha256(path) == expected, f"Secondary development source mismatch: {recorded}")
    tree = joblib.load(P2 / "outputs/preprocessors/tree_text.joblib")
    lr = joblib.load(P2 / "outputs/preprocessors/lr_text.joblib")
    tree_v = tree.named_transformers_["text"].named_steps["tfidf"]
    lr_v = lr.named_transformers_["text"].named_steps["tfidf"]
    require(tree_v.vocabulary_ == lr_v.vocabulary_, "LR/XGBoost text vocabulary differs")
    require(np.array_equal(tree_v.idf_, lr_v.idf_), "LR/XGBoost IDF differs")
    return {"secondary_source_hashes_verified": len(metadata["source_sha256"]),
            "shared_text_vocabulary_and_idf": True, "text_terms": len(tree_v.vocabulary_)}


def load_partition(partition):
    require(partition in {"validation", "test"}, "Only evaluation partitions may be inferred")
    folder = P2 / "outputs/matrices"
    labels = pd.read_csv(folder / f"labels_{partition}.csv")
    raw = pd.read_csv(folder / f"X_raw_{partition}.csv")
    predictors = pd.read_csv(P1 / "predictor_columns.csv")["predictor"].tolist()
    require(predictors == ap.PREDICTORS, "Predictor whitelist mismatch")
    require(raw.columns.tolist() == ["stay_id"] + predictors, "Raw input schema mismatch")
    require(np.array_equal(raw.stay_id, labels.stay_id), "Raw predictor order differs from labels")
    require(not labels.stay_id.duplicated().any(), "Duplicate labelled encounter")
    require(labels.partition.eq(partition).all(), "Label partition mismatch")
    require((labels[TARGETS[1]] <= labels[TARGETS[0]]).all(), "Secondary positives must be primary positives")
    manifest = pd.read_csv(P2 / "outputs/split_manifest.csv")
    require(not manifest.stay_id.duplicated().any(), "Duplicate split manifest encounter")
    require(manifest.groupby("subject_id").partition.nunique().max() == 1, "Patient appears in multiple partitions")
    expected = manifest.loc[manifest.partition.eq(partition), ["stay_id", "subject_id", "partition"]]
    pd.testing.assert_frame_equal(labels[expected.columns].sort_values("stay_id").reset_index(drop=True),
                                  expected.sort_values("stay_id").reset_index(drop=True))
    structured = np.load(folder / f"tree_structured_{partition}.npy", allow_pickle=False)
    original = sparse.load_npz(folder / f"tree_text_{partition}.npz").tocsr()
    names = pd.read_csv(P2 / "outputs/preprocessors/tree_structured_feature_names.csv")["feature"].tolist()
    text_names = pd.read_csv(P2 / "outputs/preprocessors/tree_text_feature_names.csv")["feature"].tolist()
    require(names == text_names[:len(names)], "Structured feature prefixes differ")
    require(structured.shape == (len(labels), 19) and original.shape == (len(labels), 2791), "Feature dimensions differ")
    indices = [names.index(n) for n in ["num__age", "cat__gender_M"]]
    corrected = rebuild_tree_text(structured, original)
    return {"labels": labels, "raw": raw[predictors], "structured": structured, "original_text": original,
            "age_gender": structured[:, indices], "age_gender_indices": indices,
            "explicit_structured_entries": int(corrected[:, :19].nnz)}


def predict_partition(partition):
    data = load_partition(partition)
    labels = data["labels"]
    probabilities = {}
    diagnostics = []
    for target, folder in [(TARGETS[0], P3_PRIMARY), (TARGETS[1], P3_SECONDARY)]:
        for name in MODEL_NAMES[:4]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = joblib.load(folder / f"outputs/models/{name}.joblib")
                require("preprocess" in model.named_steps, "P3 raw-input pipeline is missing preprocessing")
                if target == TARGETS[1]:
                    require(model.target_name_ == target, f"{name}: secondary pipeline target mismatch")
                require(list(model.classes_) == [0, 1], "Unexpected LR class ordering")
                probability = model.predict_proba(data["raw"])[:, 1]
            probabilities[target, name] = np.asarray(probability, dtype=float)
            diagnostics.append({"target": target, "model_name": name, "input": "13 cleaned raw columns",
                "file": relative(folder / f"outputs/models/{name}.joblib"),
                "warnings": sorted(set(str(w.message) for w in caught))})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        models = p4.load_p4_models(P4 / "outputs/models")
        for model in models.values():
            model.set_params(n_jobs=1)
            require(list(model.classes_) == [0, 1], "P4 API assumes class 1 in column 1")
        predictions = p4.p4_predict(models, data["age_gender"], data["structured"], data["original_text"])
    for exported_name, probability in predictions.items():
        target = TARGETS[1] if exported_name.startswith("stage2_") else TARGETS[0]
        name = exported_name.removeprefix("stage2_")
        require(name in MODEL_NAMES[4:], "Unexpected P4 model name")
        width = 2 if name == "age_gender_xgboost" else 19 if name == "structured_xgboost" else 2791
        require(models[exported_name].n_features_in_ == width, "P4 feature count mismatch")
        probabilities[target, name] = np.asarray(probability, dtype=float)
        diagnostics.append({"target": target, "model_name": name, "exported_name": exported_name,
            "input_features": width, "api": "P4_inference.p4_predict",
            "warnings": sorted(set(str(w.message) for w in caught))})
    long_frames, wide_frames = [], []
    for target in TARGETS:
        wide = labels[["stay_id", "subject_id"]].copy()
        wide["target"] = target
        wide["y_true"] = labels[target]
        for name in MODEL_NAMES:
            p = probabilities[target, name]
            require(p.shape == (len(labels),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid predicted probabilities")
            wide[f"y_prob_{name}"] = p
            long_frames.append(pd.DataFrame({"stay_id": labels.stay_id, "subject_id": labels.subject_id,
                "target": target, "observed_label": labels[target], "predicted_probability": p,
                "model_name": name, "partition": partition}))
        wide_frames.append(wide)
    return data, pd.concat(long_frames, ignore_index=True), pd.concat(wide_frames, ignore_index=True), diagnostics


def reference_predictions(labels):
    references, reported = {}, {}
    for target, p3 in [(TARGETS[0], P3_PRIMARY), (TARGETS[1], P3_SECONDARY)]:
        prefix = "stage2_" if target == TARGETS[1] else ""
        for folder, prediction_file, metric_file, names in [
            (p3, "validation_predictions.csv", "validation_metrics.csv", MODEL_NAMES[:4]),
            (P4, prefix + "validation_predictions.csv", prefix + "final_model_comparison.csv", MODEL_NAMES[4:])]:
            exported = pd.read_csv(folder / "outputs" / prediction_file, float_precision="round_trip")
            exported["model_name"] = exported.model_name.str.removeprefix("stage2_")
            require(set(exported.model_name) == set(names), "Handoff prediction model set differs")
            require(exported.partition.eq("validation").all(), "Handoff partition mismatch")
            valid_targets = {target, "Stage ≥2" if target == TARGETS[1] else "Stage ≥1"}
            require(set(exported.target) <= valid_targets, "Handoff outcome mismatch")
            require(not exported.duplicated(["stay_id", "model_name"]).any(), "Duplicate handoff prediction")
            report = pd.read_csv(folder / "outputs" / metric_file)
            report["model_name"] = report.model_name.str.removeprefix("stage2_")
            report = report.set_index("model_name")
            for name in names:
                group = exported.loc[exported.model_name.eq(name)].set_index("stay_id")
                require(set(group.index) == set(labels.stay_id), "Handoff encounter set differs")
                group = group.loc[labels.stay_id]
                require(np.array_equal(group.subject_id, labels.subject_id), "Handoff patient mismatch")
                require(np.array_equal(group.observed_label, labels[target]), "Handoff labels mismatch")
                references[target, name] = group.predicted_probability.to_numpy()
                reported[target, name] = report.loc[name]
    return references, reported
