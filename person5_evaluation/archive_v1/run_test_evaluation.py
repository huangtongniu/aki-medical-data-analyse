"""Evaluate frozen models once; resume bootstrap/reporting from cached predictions."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from common import (CHECKS, MODEL_NAMES, P2, P2_QA, PREDICTIONS, ROOT, TABLES,
                    check_frozen_sources, environment, make_output_dirs, read_json,
                    relative, require, save_csv, settings, sha256, upstream_files,
                    utc_now, write_json)
from inference import predict_partition
from metrics import METRIC_NAMES, PreparedMetrics


def summarize_bootstrap(point, samples, cfg, scope):
    alpha = (1 - cfg["confidence_level"]) / 2
    rows = []
    for j, name in enumerate(MODEL_NAMES):
        for k, metric in enumerate(METRIC_NAMES):
            values = samples[:, j, k]
            values = values[np.isfinite(values)]
            lo, hi = np.quantile(values, [alpha, 1 - alpha]) if len(values) else (np.nan, np.nan)
            rows.append({"scope": scope, "model_name": name, "metric": metric,
                         "estimate": point[j][metric], "ci_lower": lo, "ci_upper": hi,
                         "valid_replicates": len(values), "requested_replicates": cfg["bootstrap_replicates"],
                         "method": "patient-cluster percentile bootstrap; fixed models and validation cutoffs"})
    return pd.DataFrame(rows)


def paired_intervals(point, samples, cfg, scope):
    alpha = (1 - cfg["confidence_level"]) / 2
    rows = []
    for a, b in cfg["paired_comparisons"]:
        ai, bi = MODEL_NAMES.index(a), MODEL_NAMES.index(b)
        for metric in ["average_precision", "auroc", "brier_score"]:
            k = METRIC_NAMES.index(metric)
            differences = samples[:, ai, k] - samples[:, bi, k]
            valid = differences[np.isfinite(differences)]
            lo, hi = np.quantile(valid, [alpha, 1 - alpha]) if len(valid) else (np.nan, np.nan)
            rows.append({"scope": scope, "model_a": a, "model_b": b, "metric": metric,
                         "difference_a_minus_b": point[ai][metric] - point[bi][metric],
                         "ci_lower": lo, "ci_upper": hi, "valid_replicates": len(valid),
                         "interpretation": "positive favors A for AP/AUROC; negative favors A for Brier",
                         "method": "paired patient-cluster percentile bootstrap; no formal p-value"})
    return pd.DataFrame(rows)


def bootstrap_scope(labels, probabilities, thresholds, cfg, scope, scope_index, include_calibration):
    y = labels.aki_label.to_numpy()
    prepared = [PreparedMetrics(y, probabilities[name], thresholds[name]) for name in MODEL_NAMES]
    point = [m.evaluate() for m in prepared]
    patients, patient_index = np.unique(labels.subject_id.to_numpy(), return_inverse=True)
    n_patients = len(patients)
    b_count = cfg["bootstrap_replicates"]
    fingerprint = sha256(CHECKS / "analysis_freeze.json")
    array_path = CHECKS / f"bootstrap_{scope}.npy"
    progress_path = CHECKS / f"bootstrap_{scope}_progress.json"
    samples = np.full((b_count, len(MODEL_NAMES), len(METRIC_NAMES)), np.nan)
    start = 0
    if progress_path.exists():
        progress = read_json(progress_path)
        require(progress["freeze_sha256"] == fingerprint, "Bootstrap freeze differs")
        require(progress["replicates"] == b_count and progress["patients"] == n_patients, "Bootstrap configuration differs")
        samples = np.load(array_path, allow_pickle=False)
        start = progress["completed"]
    for b in range(start, b_count):
        # Iteration-specific deterministic RNG makes interrupted runs exactly resumable.
        rng = np.random.default_rng(np.random.SeedSequence([cfg["bootstrap_seed"], scope_index, b]))
        multiplicity = np.bincount(rng.integers(n_patients, size=n_patients), minlength=n_patients)
        row_weights = multiplicity[patient_index]
        for j, metric in enumerate(prepared):
            values = metric.evaluate(row_weights, include_calibration=include_calibration)
            samples[b, j] = [values[k] for k in METRIC_NAMES]
        if (b + 1) % 100 == 0 or b + 1 == b_count:
            np.save(array_path, samples, allow_pickle=False)
            write_json(progress_path, {"scope": scope, "completed": b + 1, "replicates": b_count,
                       "patients": n_patients, "encounters": len(labels), "freeze_sha256": fingerprint,
                       "seed": cfg["bootstrap_seed"], "scope_index": scope_index,
                       "all_models_share_patient_resamples": True,
                       "calibration_bootstrapped": include_calibration,
                       "patient_order": "np.unique(sorted subject_id)", "updated_at_utc": utc_now()})
            print(f"{scope}: paired patient bootstrap {b + 1}/{b_count}", flush=True)
    # AP, AUROC and Brier must be defined in every replicate in this population.
    require(np.isfinite(samples[:, :, :3]).all(), f"{scope}: a discrimination/Brier bootstrap replicate is undefined")
    if include_calibration:
        # Joint calibration is not identifiable for the constant prevalence baseline.
        require(np.isfinite(samples[:, 1:, -3:]).all(), "A nonconstant calibration bootstrap fit failed")
    replicate_rows = []
    for b in range(b_count):
        for j, name in enumerate(MODEL_NAMES):
            replicate_rows.append({"scope": scope, "replicate": b, "model_name": name,
                                   **dict(zip(METRIC_NAMES, samples[b, j]))})
    save_csv(pd.DataFrame(replicate_rows), TABLES / f"bootstrap_{scope}_replicates.csv")
    intervals = summarize_bootstrap(point, samples, cfg, scope)
    if not include_calibration:
        intervals = intervals[~intervals.metric.isin(METRIC_NAMES[-3:])]
    pairs = paired_intervals(point, samples, cfg, scope)
    return point, intervals, pairs


def main():
    make_output_dirs()
    freeze = read_json(CHECKS / "analysis_freeze.json")
    check_frozen_sources(freeze)
    require(freeze["validation_passed"], "Validation reproduction must pass before test inference")
    require(sha256(TABLES / "validation_thresholds.csv") == freeze["threshold_file_sha256"], "Validation thresholds changed")
    cfg = settings()
    require(cfg == freeze["protocol"], "Analysis settings differ from freeze")
    thresholds = freeze["thresholds"]
    start_path = CHECKS / "test_run_started.json"
    snapshot_path = CHECKS / "upstream_original_sha256.json"
    if not start_path.exists():
        write_json(snapshot_path, {relative(p): sha256(p) for p in upstream_files()})
        write_json(start_path, {"started_at_utc": utc_now(), "freeze_sha256": sha256(CHECKS / "analysis_freeze.json"),
                               "environment": environment(), "models_and_thresholds_already_frozen": True})
    else:
        require(read_json(start_path)["freeze_sha256"] == sha256(CHECKS / "analysis_freeze.json"), "Test run freeze changed")
    cache_path = PREDICTIONS / "test_predictions_all_models.csv"
    inference_path = CHECKS / "test_inference.json"
    if not cache_path.exists():
        data, long, wide, diagnostics = predict_partition("test")
        save_csv(long, PREDICTIONS / "test_predictions_long.csv")
        save_csv(wide, cache_path)
        write_json(inference_path, {"environment": environment(), "models": diagnostics,
                   "age_gender_zero_based_indices": data["age_gender_indices"],
                   "prediction_sha256": sha256(cache_path),
                   "test_input_sha256": {relative(p): sha256(p) for p in (P2 / "outputs/matrices").glob("*test*")},
                   "inference_completed_at_utc": utc_now(), "model_fitting_performed": False})
    else:
        require(inference_path.exists(), "Incomplete prediction cache: missing provenance")
        provenance = read_json(inference_path)
        require(sha256(cache_path) == provenance["prediction_sha256"], "Cached test predictions changed")
        for path, expected in provenance["test_input_sha256"].items():
            require(sha256(ROOT / path) == expected, f"Test input changed: {path}")
        wide = pd.read_csv(cache_path, float_precision="round_trip")
        print("Using frozen cached test probabilities; no repeated inference or model selection.", flush=True)
    labels = pd.read_csv(P2 / "outputs/matrices/labels_test.csv")
    require(np.array_equal(wide.stay_id, labels.stay_id) and np.array_equal(wide.subject_id, labels.subject_id), "Cached prediction IDs differ")
    require(np.array_equal(wide.y_true, labels.aki_label), "Cached test labels differ")
    probabilities = {name: wide[f"y_prob_{name}"].to_numpy() for name in MODEL_NAMES}
    all_rows, all_ci, all_pairs = [], [], []
    point, ci, pairs = bootstrap_scope(labels, probabilities, thresholds, cfg, "test", 0, True)
    for name, result in zip(MODEL_NAMES, point):
        row = {"scope": "test", "model_name": name, "target": "aki_label", "partition": "test",
               "encounters": len(labels), "patients": labels.subject_id.nunique(),
               "positives": int(labels.aki_label.sum()), "prevalence": float(labels.aki_label.mean()),
               "threshold": thresholds[name], **result}
        all_rows.append(row)
        print(f"Test {name}: AP={result['average_precision']:.6f}, AUROC={result['auroc']:.6f}, sensitivity={result['sensitivity']:.4f}", flush=True)
    save_csv(pd.DataFrame(all_rows), TABLES / "test_metrics.csv")
    save_csv(ci, TABLES / "test_confidence_intervals.csv")
    save_csv(pairs, TABLES / "test_paired_differences.csv")
    all_ci.append(ci)
    all_pairs.append(pairs)
    flags = pd.read_csv(P2_QA / "outputs/sensitivity_subset_flags.csv")
    require(not flags.stay_id.duplicated().any(), "Duplicate sensitivity flags")
    aligned = flags.set_index("stay_id").loc[labels.stay_id].reset_index()
    require(np.array_equal(aligned.subject_id, labels.subject_id), "Sensitivity subject IDs differ")
    sensitivity_rows = []
    for scope_index, subset in enumerate(cfg["subsets"], start=1):
        mask_values = aligned[subset]
        require(mask_values.isin([True, False]).all(), "Sensitivity flags must be boolean")
        mask = mask_values.to_numpy(dtype=bool)
        subset_labels = labels.loc[mask].reset_index(drop=True)
        sub_probabilities = {name: p[mask] for name, p in probabilities.items()}
        sub_point, sub_ci, sub_pairs = bootstrap_scope(subset_labels, sub_probabilities, thresholds, cfg, subset, scope_index, False)
        for name, result in zip(MODEL_NAMES, sub_point):
            sensitivity_rows.append({"scope": subset, "model_name": name, "target": "aki_label", "partition": "test",
                    "encounters": len(subset_labels), "patients": subset_labels.subject_id.nunique(),
                    "positives": int(subset_labels.aki_label.sum()), "prevalence": float(subset_labels.aki_label.mean()),
                    "threshold": thresholds[name], "analysis_type": "frozen-model subgroup evaluation; no refitting", **result})
        all_ci.append(sub_ci)
        all_pairs.append(sub_pairs)
    save_csv(pd.DataFrame(sensitivity_rows), TABLES / "sensitivity_metrics.csv")
    save_csv(pd.concat(all_ci, ignore_index=True), TABLES / "all_confidence_intervals.csv")
    save_csv(pd.concat(all_pairs, ignore_index=True), TABLES / "all_paired_differences.csv")
    before = read_json(snapshot_path)
    changes = [path for path, expected in before.items() if not (ROOT / path).is_file() or sha256(ROOT / path) != expected]
    require(not changes, f"Original handoff files changed: {changes}")
    write_json(CHECKS / "test_evaluation_complete.json", {
        "completed_at_utc": utc_now(), "freeze_sha256": sha256(CHECKS / "analysis_freeze.json"),
        "models": MODEL_NAMES, "test_encounters": len(labels), "test_patients": int(labels.subject_id.nunique()),
        "bootstrap_replicates_per_scope": cfg["bootstrap_replicates"], "scopes": ["test"] + cfg["subsets"],
        "original_files_verified_unchanged": len(before), "changed_original_files": changes,
        "test_used_for_tuning": False, "model_refitting_performed": False,
        "calibration_note": "Diagnostic fits only, not recalibration; joint slope/intercept undefined for constant baseline",
        "sensitivity_calibration_ci": "not computed; subgroup calibration point estimates only",
    })
    print("Test and sensitivity evaluation complete. Original handoff hashes unchanged.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
