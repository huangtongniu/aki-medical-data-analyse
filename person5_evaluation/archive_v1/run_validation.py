"""Reproduce validation predictions, select cutoffs, then freeze before testing."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from common import (CHECKS, HERE, MODEL_NAMES, P3, P4, PREDICTIONS, TABLES, environment,
                    freeze_sources, make_output_dirs, read_json, require, save_csv,
                    settings, sha256, utc_now, write_json)
from inference import predict_partition, reference_validation_predictions
from metrics import PreparedMetrics, choose_threshold


def main():
    make_output_dirs()
    require(not (CHECKS / "test_run_started.json").exists(), "Test evaluation has started; validation thresholds must not be changed")
    cfg = settings()
    data, long, wide, diagnostics = predict_partition("validation")
    labels = data["labels"]
    y = labels.aki_label.to_numpy()
    references = reference_validation_predictions(labels)
    metric_rows, threshold_rows, checks, handoff_rows = [], [], [], []
    p3_reported = pd.read_csv(P3 / "outputs/validation_metrics.csv").set_index("model_name")
    p4_reported = pd.read_csv(P4 / "outputs/final_model_comparison.csv").set_index("model_name")
    p4_groups = pd.read_csv(P4 / "outputs/predictor_group_comparison.csv").set_index("predictor_group")
    for name in MODEL_NAMES:
        probability = wide[f"y_prob_{name}"].to_numpy()
        error = float(np.max(np.abs(probability - references[name])))
        require(error <= cfg["probability_atol"], f"{name}: saved model does not reproduce validation probabilities ({error})")
        threshold = choose_threshold(y, probability, cfg["sensitivity_target"])
        result = PreparedMetrics(y, probability, threshold).evaluate()
        require(result["sensitivity"] >= cfg["sensitivity_target"] - 1e-12, "Sensitivity target not attained")
        independent = {
            "average_precision": average_precision_score(y, probability),
            "auroc": roc_auc_score(y, probability),
            "brier_score": brier_score_loss(y, probability),
        }
        reference_metrics = {
            "average_precision": average_precision_score(y, references[name]),
            "auroc": roc_auc_score(y, references[name]),
            "brier_score": brier_score_loss(y, references[name]),
        }
        reported = p3_reported.loc[name] if name in MODEL_NAMES[:4] else p4_groups.loc["Age/Gender"] if name == "age_gender_xgboost" else p4_reported.loc[name]
        for metric in independent:
            require(abs(result[metric] - independent[metric]) < 1e-12, f"{name}: independent {metric} implementation differs")
            require(abs(reference_metrics[metric] - float(reported[metric])) < 1e-7, f"{name}: exported validation {metric} differs from report")
            tolerance = cfg["ranking_metric_reproduction_atol"] if metric in {"average_precision", "auroc"} else 1e-7
            require(abs(result[metric] - reference_metrics[metric]) < tolerance, f"{name}: local reproduction {metric} materially differs")
        common = {"model_name": name, "target": "aki_label", "partition": "validation",
                  "encounters": len(labels), "patients": labels.subject_id.nunique(),
                  "positives": int(y.sum()), "prevalence": float(y.mean()), "threshold": threshold}
        metric_rows.append({**common, **result})
        handoff_rows.append({**common, **reference_metrics,
                             **{f"local_minus_handoff_{k}": result[k] - reference_metrics[k] for k in reference_metrics},
                             "max_abs_probability_difference": error})
        threshold_rows.append({"model_name": name, "threshold": threshold,
                               "sensitivity_target": cfg["sensitivity_target"],
                               "validation_sensitivity": result["sensitivity"],
                               "validation_specificity": result["specificity"],
                               "validation_ppv": result["ppv"], "validation_flagged_fraction": result["flagged_fraction"],
                               "constant_score": bool(np.ptp(probability) < 1e-12),
                               "decision_rule": "predicted_probability >= threshold"})
        checks.append({"model_name": name, "encounters": len(labels), "max_abs_probability_error": error,
                       "probability_atol": cfg["probability_atol"], "sample_ids_and_labels_match": True,
                       "unified_metrics_match_sklearn": True, "handoff_csv_metrics_match_report": True,
                       "local_metrics_match_handoff_within_recorded_tolerances": True,
                       "local_minus_handoff_ap": result["average_precision"] - reference_metrics["average_precision"],
                       "local_minus_handoff_auroc": result["auroc"] - reference_metrics["auroc"],
                       "threshold_sensitivity_attained": True})
        print(f"{name}: AP={result['average_precision']:.6f}, cutoff={threshold:.9g}, sensitivity={result['sensitivity']:.4f}, reproduction error={error:.3g}", flush=True)
    save_csv(long, PREDICTIONS / "validation_predictions_all_models.csv")
    save_csv(wide, PREDICTIONS / "validation_predictions_wide.csv")
    save_csv(pd.DataFrame(metric_rows), TABLES / "validation_metrics.csv")
    save_csv(pd.DataFrame(handoff_rows), TABLES / "handoff_validation_metrics.csv")
    save_csv(pd.DataFrame(threshold_rows), TABLES / "validation_thresholds.csv")
    save_csv(pd.DataFrame(checks), CHECKS / "validation_checks.csv")
    write_json(CHECKS / "validation_inference.json", {"environment": environment(), "models": diagnostics,
               "age_gender_zero_based_indices": data["age_gender_indices"], "test_data_evaluated": False})
    freeze = {
        "created_at_utc": utc_now(), "protocol": cfg, "environment": environment(),
        "validation_encounters": len(labels), "validation_patients": int(labels.subject_id.nunique()),
        "model_names": MODEL_NAMES,
        "thresholds": {r["model_name"]: r["threshold"] for r in threshold_rows},
        "threshold_file_sha256": sha256(TABLES / "validation_thresholds.csv"),
        "source_sha256": freeze_sources(),
        "validation_passed": True, "test_evaluation_started": False,
        "scope": "Frozen existing primary-outcome models; conditional evaluation, no model/preprocessor fitting",
        "sensitivity_target_authorization": "User explicitly confirmed Sensitivity >= 0.80 on 2026-10-19",
    }
    path = CHECKS / "analysis_freeze.json"
    if path.exists():
        old = read_json(path)
        require(old["source_sha256"] == freeze["source_sha256"] and old["thresholds"] == freeze["thresholds"], "Existing analysis freeze differs; do not overwrite silently")
    else:
        write_json(path, freeze)
    print("Validation passed. Analysis protocol and thresholds frozen before test evaluation.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
