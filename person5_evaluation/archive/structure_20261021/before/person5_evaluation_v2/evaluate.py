"""Validation freeze and paired evaluation of fourteen fixed models."""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from common import (CHECKS, MODEL_NAMES, OLD, P1, P2, P2_QA, PREDICTIONS, ROOT, TABLES,
    TARGETS, check_sources, environment, make_output_dirs, original_hashes, read_json,
    recover_primary_package, relative, require, save_csv, settings, sha256, source_hashes,
    utc_now, write_json)
from inference import predict_partition, reference_predictions, validate_secondary_provenance
from metrics import METRIC_NAMES, PreparedMetrics, choose_threshold

KEYS = [(target, name) for target in TARGETS for name in MODEL_NAMES]


def core_metrics(y, p, weights=None):
    return {"average_precision": average_precision_score(y, p, sample_weight=weights),
            "auroc": roc_auc_score(y, p, sample_weight=weights),
            "brier_score": brier_score_loss(y, p, sample_weight=weights)}


def validate():
    make_output_dirs()
    freeze_path = CHECKS / "analysis_freeze.json"
    if freeze_path.exists():
        freeze = read_json(freeze_path)
        check_sources(freeze)
        require(freeze["protocol"] == settings(), "Settings differ from freeze")
        require(sha256(TABLES / "validation_thresholds.csv") == freeze["threshold_file_sha256"], "Threshold file changed")
        print("Existing two-outcome validation freeze verified.", flush=True)
        return
    require(not (CHECKS / "test_run_started.json").exists(), "Cannot choose thresholds after test evaluation")
    recover_primary_package()
    write_json(CHECKS / "original_sha256.json", original_hashes())
    provenance = validate_secondary_provenance()
    cfg = settings()
    data, long, wide, diagnostics = predict_partition("validation")
    labels = data["labels"]
    require((len(labels), labels.subject_id.nunique(), int(labels.aki_label.sum()), int(labels.aki_stage2_label.sum()))
            == (16932, 9913, 1165, 333), "Validation cohort counts differ")
    references, reported = reference_predictions(labels)
    old = read_json(OLD / "outputs/checks/analysis_freeze.json")
    old_validation = pd.read_csv(OLD / "outputs/predictions/validation_predictions_wide.csv", float_precision="round_trip")
    require(np.array_equal(old_validation.stay_id, labels.stay_id), "Archived validation order differs")
    rows, thresholds, checks, handoff = [], [], [], []
    for target, name in KEYS:
        y = labels[target].to_numpy()
        p = wide.loc[wide.target.eq(target), f"y_prob_{name}"].to_numpy()
        error = float(np.max(np.abs(p - references[target, name])))
        require(error <= cfg["probability_atol"], f"{target}/{name}: probability reproduction failed ({error})")
        cutoff = choose_threshold(y, p, cfg["sensitivity_target"])
        if target == TARGETS[0]:
            require(abs(cutoff - old["thresholds"][name]) < 1e-12, "Primary cutoff differs from original freeze")
            require(np.max(np.abs(p - old_validation[f"y_prob_{name}"])) <= cfg["probability_atol"], "Primary validation predictions changed")
            cutoff = old["thresholds"][name]
        got = PreparedMetrics(y, p, cutoff).evaluate()
        independent = core_metrics(y, p)
        ref_metrics = core_metrics(y, references[target, name])
        for metric in independent:
            require(abs(got[metric] - independent[metric]) < 1e-12, "Metric implementation differs from sklearn")
            require(abs(ref_metrics[metric] - float(reported[target, name][metric])) < 1e-7, f"{target}/{name}: handoff report mismatch for {metric}")
            tolerance = cfg["ranking_metric_reproduction_atol"] if metric != "brier_score" else cfg["probability_atol"]
            require(abs(got[metric] - ref_metrics[metric]) < tolerance, f"{target}/{name}: local {metric} reproduction differs")
        require(got["sensitivity"] >= cfg["sensitivity_target"] - 1e-12, "Sensitivity target not attained")
        require((p[y == 1] >= np.nextafter(cutoff, np.inf)).mean() < cfg["sensitivity_target"], "Cutoff is not the largest qualifying cutoff")
        base = {"target": target, "model_name": name, "partition": "validation", "encounters": len(labels),
                "patients": labels.subject_id.nunique(), "positives": int(y.sum()), "prevalence": float(y.mean()), "threshold": cutoff}
        rows.append({**base, **got})
        thresholds.append({"target": target, "model_name": name, "threshold": cutoff,
            "sensitivity_target": cfg["sensitivity_target"], "validation_sensitivity": got["sensitivity"],
            "validation_specificity": got["specificity"], "validation_ppv": got["ppv"],
            "decision_rule": "probability >= threshold", "preserved_primary_cutoff": target == TARGETS[0]})
        handoff.append({**base, **ref_metrics, "max_abs_probability_difference": error,
                       **{f"local_minus_handoff_{k}": got[k] - ref_metrics[k] for k in independent}})
        checks.append({"target": target, "model_name": name, "max_abs_probability_difference": error,
            "identifiers_labels_and_target_match": True, "reported_metrics_reproduced": True,
            "independent_metrics_match": True, "largest_sensitivity_cutoff_verified": True})
        print(f"Validation {target}/{name}: AP={got['average_precision']:.6f}, cutoff={cutoff:.9g}, sensitivity={got['sensitivity']:.4f}, probability error={error:.3g}", flush=True)
    save_csv(long, PREDICTIONS / "validation_predictions_long.csv")
    save_csv(wide, PREDICTIONS / "validation_predictions_all_models.csv")
    save_csv(pd.DataFrame(rows), TABLES / "validation_metrics.csv")
    save_csv(pd.DataFrame(thresholds), TABLES / "validation_thresholds.csv")
    save_csv(pd.DataFrame(handoff), TABLES / "handoff_validation_metrics.csv")
    save_csv(pd.DataFrame(checks), CHECKS / "validation_checks.csv")
    for target in TARGETS:
        save_csv(wide[wide.target.eq(target)], PREDICTIONS / target / "validation_predictions.csv")
    write_json(CHECKS / "validation_inference.json", {"environment": environment(), "models": diagnostics,
        "age_gender_indices": data["age_gender_indices"], "explicit_structured_entries": data["explicit_structured_entries"],
        "secondary_development_provenance": provenance, "test_partition_inferred": False})
    freeze = {"created_at_utc": utc_now(), "protocol": cfg, "environment": environment(),
        "thresholds": {target: {r["model_name"]: r["threshold"] for r in thresholds if r["target"] == target} for target in TARGETS},
        "threshold_file_sha256": sha256(TABLES / "validation_thresholds.csv"), "source_sha256": source_hashes(),
        "validation_passed": True, "primary_original_freeze_sha256": sha256(OLD / "outputs/checks/analysis_freeze.json"),
        "chronology": "Primary evaluation already completed; this validation-only freeze precedes new secondary model test inference.",
        "authorization": "User requested both outcomes on 2026-10-21; retain previously confirmed validation sensitivity >=0.80 rule."}
    write_json(freeze_path, freeze)
    print("Both outcomes passed validation. Separate target-specific cutoffs frozen before new test inference.", flush=True)


def patient_layout(labels):
    """Stratify patients by their highest outcome category among eligible encounters."""
    patients, inverse = np.unique(labels.subject_id.to_numpy(), return_inverse=True)
    encounter_stage = labels.aki_label.to_numpy(dtype=int) + labels.aki_stage2_label.to_numpy(dtype=int)
    patient_stage = np.zeros(len(patients), dtype=int)
    np.maximum.at(patient_stage, inverse, encounter_stage)
    groups = [np.flatnonzero(patient_stage == stage) for stage in [0, 1, 2]]
    return patients, inverse, patient_stage, groups


def draw_multiplicities(groups, n_patients, seed, scope_index, replicate):
    rng = np.random.default_rng(np.random.SeedSequence([seed, scope_index, replicate]))
    counts = np.zeros(n_patients, dtype=np.int64)
    for group in groups:
        if len(group):
            selected = group[rng.integers(len(group), size=len(group))]
            counts += np.bincount(selected, minlength=n_patients)
    return counts


def bootstrap_scope(labels, probabilities, freeze, scope, scope_index, include_calibration):
    cfg = freeze["protocol"]
    prepared = [PreparedMetrics(labels[target].to_numpy(), probabilities[target, name], freeze["thresholds"][target][name]) for target, name in KEYS]
    points = [metric.evaluate() for metric in prepared]
    patients, inverse, stages, groups = patient_layout(labels)
    count = cfg["bootstrap_replicates"]
    path = CHECKS / f"bootstrap_{scope}.npy"
    progress_path = CHECKS / f"bootstrap_{scope}_progress.json"
    fingerprint = sha256(CHECKS / "analysis_freeze.json")
    samples = np.full((count, len(KEYS), len(METRIC_NAMES)), np.nan)
    start = 0
    if progress_path.exists():
        progress = read_json(progress_path)
        require(progress["freeze_sha256"] == fingerprint and progress["requested_replicates"] == count, "Bootstrap cache fingerprint differs")
        samples = np.load(path)
        require(samples.shape == (count, len(KEYS), len(METRIC_NAMES)), "Bootstrap cache dimensions differ")
        start = progress["completed_replicates"]
    for b in range(start, count):
        multiplicities = draw_multiplicities(groups, len(patients), cfg["bootstrap_seed"], scope_index, b)
        weights = multiplicities[inverse]
        for j, metric in enumerate(prepared):
            result = metric.evaluate(weights, include_calibration=include_calibration)
            samples[b, j] = [result[k] for k in METRIC_NAMES]
        if (b + 1) % 100 == 0 or b + 1 == count:
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, samples)
            temporary.replace(path)
            write_json(progress_path, {"scope": scope, "scope_index": scope_index, "freeze_sha256": fingerprint,
                "completed_replicates": b + 1, "requested_replicates": count, "patients": len(patients),
                "encounters": len(labels), "patient_stratum_counts": {str(s): len(g) for s, g in enumerate(groups)},
                "all_models_and_targets_share_draws": True, "calibration_intervals": include_calibration,
                "updated_at_utc": utc_now()})
            print(f"{scope}: shared stratified patient bootstrap {b + 1}/{count}", flush=True)
    alpha = (1 - cfg["confidence_level"]) / 2
    rows, intervals, pairs, replicate_rows = [], [], [], []
    for j, (target, name) in enumerate(KEYS):
        y = labels[target].to_numpy()
        base = {"target": target, "model_name": name, "scope": scope}
        rows.append({**base, "partition": "test", "encounters": len(labels), "patients": len(patients),
            "positives": int(y.sum()), "prevalence": float(y.mean()), "threshold": freeze["thresholds"][target][name], **points[j]})
        for k, metric in enumerate(METRIC_NAMES):
            if not include_calibration and metric in METRIC_NAMES[-3:]:
                continue
            values = samples[:, j, k]
            values = values[np.isfinite(values)]
            lo, hi = np.quantile(values, [alpha, 1 - alpha]) if len(values) else (np.nan, np.nan)
            intervals.append({**base, "metric": metric, "estimate": points[j][metric], "ci_lower": lo, "ci_upper": hi,
                "valid_replicates": len(values), "requested_replicates": count, "method": cfg["bootstrap_method"]})
        for b in range(count):
            replicate_rows.append({**base, "replicate": b, **dict(zip(METRIC_NAMES, samples[b, j]))})
    for target in TARGETS:
        for a, b in cfg["paired_comparisons"]:
            ai, bi = KEYS.index((target, a)), KEYS.index((target, b))
            for metric in METRIC_NAMES[:3]:
                k = METRIC_NAMES.index(metric)
                values = samples[:, ai, k] - samples[:, bi, k]
                values = values[np.isfinite(values)]
                lo, hi = np.quantile(values, [alpha, 1 - alpha]) if len(values) else (np.nan, np.nan)
                pairs.append({"target": target, "scope": scope, "model_a": a, "model_b": b, "metric": metric,
                    "difference_a_minus_b": points[ai][metric] - points[bi][metric], "ci_lower": lo, "ci_upper": hi,
                    "valid_replicates": len(values), "method": cfg["bootstrap_method"], "multiplicity_adjusted": False})
    save_csv(pd.DataFrame(replicate_rows), TABLES / f"bootstrap_{scope}_replicates.csv")
    return pd.DataFrame(rows), pd.DataFrame(intervals), pd.DataFrame(pairs)


def test():
    freeze = read_json(CHECKS / "analysis_freeze.json")
    check_sources(freeze)
    require(freeze["protocol"] == settings(), "Protocol changed after freeze")
    require(sha256(TABLES / "validation_thresholds.csv") == freeze["threshold_file_sha256"], "Thresholds changed")
    started_path = CHECKS / "test_run_started.json"
    if not started_path.exists():
        write_json(started_path, {"started_at_utc": utc_now(), "freeze_sha256": sha256(CHECKS / "analysis_freeze.json"),
            "new_secondary_inference": True, "primary_test_results_previously_known": True})
    else:
        require(read_json(started_path)["freeze_sha256"] == sha256(CHECKS / "analysis_freeze.json"), "Test-start freeze differs")
    cache = PREDICTIONS / "test_predictions_all_models.csv"
    provenance_path = CHECKS / "test_inference.json"
    if not provenance_path.exists():
        data, long, wide, diagnostics = predict_partition("test")
        save_csv(long, PREDICTIONS / "test_predictions_long.csv")
        save_csv(wide, cache)
        for target in TARGETS:
            save_csv(wide[wide.target.eq(target)], PREDICTIONS / target / "test_predictions.csv")
        write_json(provenance_path, {"completed_at_utc": utc_now(), "environment": environment(), "models": diagnostics,
            "prediction_sha256": sha256(cache), "long_prediction_sha256": sha256(PREDICTIONS / "test_predictions_long.csv"),
            "input_sha256": {relative(p): sha256(p) for p in (P2 / "outputs/matrices").glob("*test*")},
            "model_refitting": False, "p4_supplied_api_used": True})
    else:
        provenance = read_json(provenance_path)
        require(sha256(cache) == provenance["prediction_sha256"], "Prediction cache changed")
        for path, expected in provenance["input_sha256"].items():
            require(sha256(ROOT / path) == expected, "Test inputs changed")
        wide = pd.read_csv(cache, float_precision="round_trip")
        print("Using verified frozen test prediction cache.", flush=True)
    labels = pd.read_csv(P2 / "outputs/matrices/labels_test.csv")
    require((len(labels), labels.subject_id.nunique(), int(labels.aki_label.sum()), int(labels.aki_stage2_label.sum()))
            == (16801, 9914, 1155, 332), "Test cohort counts differ from agreed split")
    probabilities = {}
    for target in TARGETS:
        group = wide[wide.target.eq(target)]
        require(np.array_equal(group.stay_id, labels.stay_id) and np.array_equal(group.subject_id, labels.subject_id), "Test IDs/order differ")
        require(np.array_equal(group.y_true, labels[target]), "Test target labels differ")
        for name in MODEL_NAMES:
            probabilities[target, name] = group[f"y_prob_{name}"].to_numpy()
    old = pd.read_csv(OLD / "outputs/predictions/test_predictions_all_models.csv", float_precision="round_trip")
    require(np.array_equal(old.stay_id, labels.stay_id), "Original test IDs differ")
    reproduction = []
    for name in MODEL_NAMES:
        error = float(np.max(np.abs(probabilities[TARGETS[0], name] - old[f"y_prob_{name}"])))
        require(error <= settings()["probability_atol"], "Primary test predictions differ from archived evaluation")
        reproduction.append({"model_name": name, "max_abs_probability_error": error, "same_primary_model_hash": True,
                             "same_primary_cutoff": True, "bootstrap_method_changed_to_stratified": True})
    save_csv(pd.DataFrame(reproduction), CHECKS / "primary_v1_reproduction.csv")
    all_points, all_intervals, all_pairs = [], [], []
    flags = pd.read_csv(P2_QA / "outputs/sensitivity_subset_flags.csv")
    require(not flags.stay_id.duplicated().any(), "Duplicate sensitivity flag encounter")
    flags = flags.set_index("stay_id").loc[labels.stay_id]
    require(np.array_equal(flags.subject_id, labels.subject_id), "Sensitivity flag patients differ")
    scopes = ["test"] + settings()["subsets"]
    for index, scope in enumerate(scopes):
        if scope == "test":
            mask = np.ones(len(labels), dtype=bool)
        else:
            require(flags[scope].isin([True, False]).all(), "Nonboolean sensitivity flag")
            mask = flags[scope].to_numpy(dtype=bool)
        scoped_labels = labels.loc[mask].reset_index(drop=True)
        scoped_probabilities = {key: p[mask] for key, p in probabilities.items()}
        point, intervals, pairs = bootstrap_scope(scoped_labels, scoped_probabilities, freeze, scope, index, scope == "test")
        all_points.append(point)
        all_intervals.append(intervals)
        all_pairs.append(pairs)
        if scope == "test":
            save_csv(point, TABLES / "test_metrics.csv")
            save_csv(intervals, TABLES / "test_confidence_intervals.csv")
            save_csv(pairs, TABLES / "test_paired_differences.csv")
            print(point[["target", "model_name", "average_precision", "auroc", "sensitivity", "ppv"]].to_string(index=False), flush=True)
    save_csv(pd.concat(all_points[1:], ignore_index=True), TABLES / "sensitivity_metrics.csv")
    save_csv(pd.concat(all_points, ignore_index=True), TABLES / "all_metrics.csv")
    save_csv(pd.concat(all_intervals, ignore_index=True), TABLES / "all_confidence_intervals.csv")
    save_csv(pd.concat(all_pairs, ignore_index=True), TABLES / "all_paired_differences.csv")
    cohort = pd.read_csv(P1 / "modelling_data.csv", usecols=["subject_id", "aki_label", "aki_stage2_label"])
    require((len(cohort), cohort.subject_id.nunique(), int(cohort.aki_label.sum()), int(cohort.aki_stage2_label.sum()))
            == (113097, 66090, 7866, 2254), "Full cohort counts differ")
    before = read_json(CHECKS / "original_sha256.json")
    changes = [path for path, expected in before.items() if not (ROOT / path).is_file() or sha256(ROOT / path) != expected]
    require(not changes, f"Original deliverables changed: {changes}")
    write_json(CHECKS / "evaluation_complete.json", {"completed_at_utc": utc_now(), "targets": TARGETS,
        "models_per_target": 7, "bootstrap_replicates_per_scope": settings()["bootstrap_replicates"], "scopes": scopes,
        "full_cohort": {"encounters": len(cohort), "patients": int(cohort.subject_id.nunique()), "primary_positives": 7866, "secondary_positives": 2254},
        "test_encounters": len(labels), "test_patients": int(labels.subject_id.nunique()),
        "original_files_verified_unchanged": len(before), "changed_original_files": changes,
        "model_refitting": False, "test_used_for_threshold_selection": False,
        "interval_change": "V2 uses the requested stratified patient bootstrap; archived V1 unstratified intervals are preserved."})
    print("Two-outcome test and sensitivity evaluation complete; original deliverables unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["validate", "test"])
    with threadpool_limits(limits=1):
        args = parser.parse_args()
        validate() if args.step == "validate" else test()
