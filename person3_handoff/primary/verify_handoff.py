"""Independent numerical and input-isolation checks; does not fit models."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from person3_baselines import C_GRID, MODEL_ORDER, find_project_dir, sha256


def manual_metrics(y, p):
    """Compute AP with tied scores, rank-based AUROC and squared-error Brier."""
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    group_ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(p) - 1]
    cumulative_tp = np.cumsum(sorted_y)[group_ends]
    precision = cumulative_tp / (group_ends + 1)
    recall_increments = np.diff(np.r_[0, cumulative_tp]) / y.sum()
    ap = np.sum(precision * recall_increments)
    ranks = pd.Series(p).rank(method="average").to_numpy()
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"average_precision": float(ap), "auroc": float(auc),
            "brier_score": float(np.mean((p - y) ** 2))}


def verify(project_dir=None):
    root = find_project_dir(project_dir)
    p3 = root / "person3_handoff"
    out = p3 / "outputs"
    base = root / "person2_handoff/person2_split_preprocessing/outputs/matrices"
    raw = pd.read_csv(base / "X_raw_validation.csv")
    labels = pd.read_csv(base / "labels_validation.csv", usecols=["stay_id", "subject_id", "aki_label"])
    X = raw.drop(columns="stay_id")
    y = labels.aki_label.to_numpy()
    predictions = pd.read_csv(out / "validation_predictions.csv", float_precision="round_trip")
    metrics = pd.read_csv(out / "validation_metrics.csv", float_precision="round_trip").set_index("model_name")
    check_results = {}

    def require(name, condition):
        check_results[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    # Pin the upstream provenance without opening test data files.
    metadata = json.loads((out / "model_metadata.json").read_text())
    require("Every upstream source hash still matches", all(sha256(root / file) == value
            for file, value in metadata["source_sha256"].items()))
    pipelines = {}
    for name in MODEL_ORDER:
        pipeline = joblib.load(out / "models" / f"{name}.joblib")
        pipelines[name] = pipeline
        exported = predictions[predictions.model_name.eq(name)].reset_index(drop=True)
        require(f"{name}: source validation IDs and labels agree", exported[["stay_id", "subject_id"]].equals(labels[["stay_id", "subject_id"]])
                and np.array_equal(exported.observed_label.to_numpy(), y))
        probability = pipeline.predict_proba(X)[:, 1]
        require(f"{name}: exported probabilities exactly equal reloaded predictions",
                np.array_equal(probability, exported.predicted_probability.to_numpy()))
        own = manual_metrics(y, probability)
        require(f"{name}: AP, AUROC and Brier independently reproduced", all(
            abs(value - metrics.loc[name, metric]) <= 1e-12 for metric, value in own.items()))
        transformed = pipeline.named_steps["preprocess"].transform(X)
        if "select" in pipeline.named_steps:
            transformed = pipeline.named_steps["select"].transform(transformed)
        estimator = pipeline.named_steps["model"]
        if hasattr(estimator, "coef_"):
            linear_score = transformed @ estimator.coef_[0] + estimator.intercept_[0]
            manual_probability = 1 / (1 + np.exp(-linear_score))
            require(f"{name}: probabilities reproduced from coefficients", np.allclose(probability, manual_probability, atol=1e-14, rtol=0))

    sample = X.iloc[:64].copy()
    changed = sample.copy()
    for field in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "acuity", "pain_numeric"]:
        changed[field] = 0.0
    changed["arrival_transport"] = "OTHER"
    changed["pain_status"] = "missing"
    changed["chiefcomplaint"] = "synthetic changed text"
    require("Age/gender probabilities unaffected by all other predictors", np.array_equal(
        pipelines["age_gender_lr"].predict_proba(sample), pipelines["age_gender_lr"].predict_proba(changed)))
    require("Age/gender classifier has exactly two predictive columns",
            pipelines["age_gender_lr"].named_steps["model"].coef_.shape == (1, 2))
    all_changed = changed.copy()
    all_changed["age"] = 18
    all_changed["gender"] = "F"
    require("Constant probabilities unaffected by any predictor", np.array_equal(
        pipelines["prevalence"].predict_proba(sample), pipelines["prevalence"].predict_proba(all_changed)))
    text_changed = sample.copy()
    text_changed["chiefcomplaint"] = "synthetic changed text"
    require("Structured model ignores chief-complaint text", np.array_equal(
        pipelines["structured_lr"].predict_proba(sample), pipelines["structured_lr"].predict_proba(text_changed)))
    require("Text model uses chief-complaint information", bool(np.any(np.abs(
        pipelines["structured_text_lr"].predict_proba(sample)[:, 1]
        - pipelines["structured_text_lr"].predict_proba(text_changed)[:, 1]) > 1e-8)))
    text_pipeline = pipelines["structured_text_lr"]
    text_matrix = text_pipeline.named_steps["preprocess"].transform(sample)
    text_classifier = text_pipeline.named_steps["model"]
    require("Text LR sparse and dense zero representations give equivalent probabilities",
            np.allclose(text_classifier.predict_proba(text_matrix), text_classifier.predict_proba(text_matrix.toarray()), atol=1e-13, rtol=0))
    require("Text classifier has exactly 2,799 predictive columns", text_classifier.coef_.shape == (1, 2799))
    tuning = pd.read_csv(out / "tuning_record.csv", float_precision="round_trip")
    text_tuning = tuning[tuning.model_name.eq("structured_text_lr")]
    chosen = text_tuning[text_tuning.selected]
    best_ap = text_tuning.average_precision.max()
    expected_C = text_tuning.loc[best_ap - text_tuning.average_precision <= 1e-12, "C"].min()
    require("Text tuning follows the six-value bounded AP selection rule",
            len(text_tuning) == 6 and sorted(text_tuning.C.tolist()) == C_GRID
            and text_tuning.converged.all() and len(chosen) == 1
            and chosen.C.iloc[0] == expected_C == metadata["selected_text_C"])
    require("Only four primary validation model predictions exist",
            set(predictions.model_name) == set(MODEL_ORDER)
            and set(predictions.target) == {"aki_label"}
            and set(predictions.partition) == {"validation"}
            and len(predictions) == 67728)
    reference = json.loads((p3 / "reference_baselines_v1.json").read_text())
    for name, original in reference["models"].items():
        exported = predictions[predictions.model_name.eq(name)]
        hashes = {column: hashlib.sha256(np.asarray(exported[column], dtype="<f8" if column == "predicted_probability" else "<i8").tobytes()).hexdigest()
                  for column in ["stay_id", "subject_id", "observed_label", "predicted_probability"]}
        require(f"{name}: original v1 validation prediction vectors are unchanged", hashes == original["vectors_sha256"])
        require(f"{name}: original v1 AP, AUROC and Brier are unchanged", all(
            metrics.loc[name, metric] == value for metric, value in original["metrics"].items()))
    source = (p3 / "person3_baselines.py").read_text()
    import nbformat
    notebook = nbformat.read(p3 / "P3_baseline_models.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    require("Notebook fully executed without cell errors", all(cell.execution_count is not None for cell in code_cells)
            and not any(output.output_type == "error" for cell in code_cells for output in cell.outputs))
    notebook_source = "\n".join(cell.source for cell in code_cells)
    chunks = source.split('\nif __name__ == "__main__":')[0].split("# %% ")[1:]
    require("Notebook workflow definitions match executable source", all(
        chunk.partition("\n")[2].strip() in notebook_source for chunk in chunks))
    payload = {"purpose": "Independent checks on saved models and validation outputs; no model fitting",
               "checks": check_results, "checks_passed": len(check_results), "test_data_opened": False}
    (out / "independent_validation_checks.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Independent checks passed: {len(check_results)} / {len(check_results)}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=None)
    verify(parser.parse_args().project_dir)
