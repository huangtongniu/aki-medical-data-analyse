"""Independent numerical checks of exported two-outcome results."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import re
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits

from common import (CHECKS, FIGURES, HERE, MODEL_NAMES, OLD, P2, P2_QA, PREDICTIONS, ROOT, TABLES,
    TARGETS, check_sources, read_json, require, settings, sha256, utc_now, write_json, resolve_recorded_path)
from metrics import METRIC_NAMES


def write_artifact_manifest():
    artifacts = [p for p in HERE.rglob("*") if p.is_file() and "inputs" not in p.parts and "__pycache__" not in p.parts
                 and "notebook_runtime" not in p.parts and p.name != "artifact_sha256.json"]
    write_json(CHECKS / "artifact_sha256.json", {p.relative_to(HERE).as_posix(): sha256(p) for p in artifacts})


def main():
    checks = []

    def check(condition, label):
        require(condition, label)
        checks.append(label)

    freeze = read_json(CHECKS / "analysis_freeze.json")
    check_sources(freeze)
    checks.append("Frozen models, inference, metrics, protocol and inputs verified; path-only code amendments and original backups verified")
    cfg = settings()
    check(cfg == freeze["protocol"], "Protocol matches freeze")
    check(read_json(CHECKS / "statistical_unit_checks.json")["successful"], "Nine numerical and cluster-sampling unit checks passed")
    check(datetime.fromisoformat(freeze["created_at_utc"]) < datetime.fromisoformat(read_json(CHECKS / "test_run_started.json")["started_at_utc"]), "New validation freeze predates new test inference")
    check(sha256(TABLES / "validation_thresholds.csv") == freeze["threshold_file_sha256"], "Validation cutoffs unchanged")
    labels = pd.read_csv(P2 / "outputs/matrices/labels_test.csv")
    wide = pd.read_csv(PREDICTIONS / "test_predictions_all_models.csv", float_precision="round_trip")
    long = pd.read_csv(PREDICTIONS / "test_predictions_long.csv", float_precision="round_trip")
    validation = pd.read_csv(PREDICTIONS / "validation_predictions_all_models.csv", float_precision="round_trip")
    points = pd.read_csv(TABLES / "all_metrics.csv", float_precision="round_trip")
    intervals = pd.read_csv(TABLES / "all_confidence_intervals.csv", float_precision="round_trip")
    pairs = pd.read_csv(TABLES / "all_paired_differences.csv", float_precision="round_trip")
    check(len(wide) == 2 * len(labels) and not wide.duplicated(["target", "stay_id"]).any(), "One wide prediction per outcome and encounter")
    check(len(long) == 14 * len(labels) and not long.duplicated(["target", "model_name", "stay_id"]).any(), "One long prediction per outcome/model/encounter")
    check((labels.aki_stage2_label <= labels.aki_label).all(), "Stage >=2 positives nested in primary positives")
    check((labels.aki_label.eq(1) & labels.aki_stage2_label.eq(0)).sum() == 823, "823 Stage 1 test encounters retained as secondary negatives")
    keys = [(target, name) for target in TARGETS for name in MODEL_NAMES]
    old = pd.read_csv(OLD / "outputs/tables/test_metrics.csv", float_precision="round_trip").set_index("model_name")
    for target, name in keys:
        group = wide[wide.target.eq(target)]
        row = points[points.target.eq(target) & points.model_name.eq(name) & points.scope.eq("test")].iloc[0]
        y, p = group.y_true.to_numpy(), group[f"y_prob_{name}"].to_numpy()
        check(np.array_equal(group.stay_id, labels.stay_id) and np.array_equal(group.subject_id, labels.subject_id), f"{target}/{name}: patient/encounter alignment")
        check(np.array_equal(y, labels[target]), f"{target}/{name}: correct outcome")
        detail = long[long.target.eq(target) & long.model_name.eq(name)].set_index("stay_id").loc[group.stay_id]
        check(np.array_equal(detail.predicted_probability, p), f"{target}/{name}: long/wide predictions agree")
        expected = [average_precision_score(y, p), roc_auc_score(y, p), brier_score_loss(y, p)]
        check(np.allclose(row[METRIC_NAMES[:3]].to_numpy(dtype=float), expected, atol=1e-12, rtol=0), f"{target}/{name}: AP/AUROC/Brier reproduced by sklearn")
        threshold = freeze["thresholds"][target][name]
        check(row.threshold == threshold, f"{target}/{name}: test cutoff equals frozen validation cutoff")
        tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
        check(np.array_equal(row[["true_positives", "false_positives", "true_negatives", "false_negatives"]].to_numpy(), [tp, fp, tn, fn]), f"{target}/{name}: confusion counts independently reproduced")
        val = validation[validation.target.eq(target)]
        positive_scores = val.loc[val.y_true.eq(1), f"y_prob_{name}"].to_numpy()
        check((positive_scores >= threshold).mean() >= .8 - 1e-12 and (positive_scores >= np.nextafter(threshold, np.inf)).mean() < .8,
              f"{target}/{name}: maximum cutoff attaining validation sensitivity >=0.8")
        x = logit(np.clip(p, 1e-12, 1 - 1e-12))
        offset = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=x).fit()
        check(abs(offset.params[0] - row.calibration_in_the_large) < 1e-7, f"{target}/{name}: CITL matches independent GLM")
        if name != "prevalence":
            fitted = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
            check(np.allclose(fitted.params, [row.calibration_intercept, row.calibration_slope], atol=1e-7, rtol=0), f"{target}/{name}: slope/intercept match independent GLM")
        else:
            check(np.isnan(row.calibration_slope) and np.isnan(row.calibration_intercept) and np.isnan(row.npv), f"{target}: undefined constant-model statistics are missing")
        if target == TARGETS[0]:
            check(np.allclose(row[METRIC_NAMES[:10]].to_numpy(dtype=float), old.loc[name, METRIC_NAMES[:10]].to_numpy(dtype=float), atol=1e-12, rtol=0, equal_nan=True), f"{name}: primary discrimination and cutoff metrics unchanged")

    flags = pd.read_csv(P2_QA / "outputs/sensitivity_subset_flags.csv").set_index("stay_id").loc[labels.stay_id]
    scopes = ["test"] + cfg["subsets"]
    for scope_index, scope in enumerate(scopes):
        mask = np.ones(len(labels), dtype=bool) if scope == "test" else flags[scope].to_numpy(dtype=bool)
        scoped = labels[mask].copy()
        patient_max = scoped.assign(stage=scoped.aki_label + scoped.aki_stage2_label).groupby("subject_id", sort=True).stage.max()
        patient_ids = patient_max.index.to_numpy()
        positions = pd.Index(patient_ids).get_indexer(scoped.subject_id)
        array = np.load(CHECKS / f"bootstrap_{scope}.npy")
        progress = read_json(CHECKS / f"bootstrap_{scope}_progress.json")
        check(progress["completed_replicates"] == cfg["bootstrap_replicates"], f"{scope}: 1000 replicates complete")
        check(array.shape == (1000, 14, 13), f"{scope}: fourteen-model bootstrap dimensions")
        check(np.isfinite(array[:, :, :3]).all(), f"{scope}: all ranking/Brier replicates defined")
        for target, name in keys:
            group = wide[wide.target.eq(target)]
            y, p = group.y_true.to_numpy()[mask], group[f"y_prob_{name}"].to_numpy()[mask]
            row = points[points.target.eq(target) & points.model_name.eq(name) & points.scope.eq(scope)].iloc[0]
            check(row.encounters == len(scoped) and row.patients == len(patient_ids) and row.positives == y.sum(), f"{scope}/{target}/{name}: subgroup denominators")
            check(np.allclose(row[METRIC_NAMES[:3]].to_numpy(dtype=float), [average_precision_score(y, p), roc_auc_score(y, p), brier_score_loss(y, p)], atol=1e-12, rtol=0), f"{scope}/{target}/{name}: subgroup discrimination reproduced")
        # Reconstruct selected draws independently, including their patient strata.
        for b in [0, 499, 999]:
            rng = np.random.default_rng(np.random.SeedSequence([cfg["bootstrap_seed"], scope_index, b]))
            sampled_ids = []
            for stage in [0, 1, 2]:
                members = patient_ids[patient_max.to_numpy() == stage]
                if len(members):
                    sampled_ids.extend(members[rng.integers(len(members), size=len(members))].tolist())
            frequencies = pd.Series(sampled_ids).value_counts().reindex(patient_ids, fill_value=0).to_numpy()
            weights = frequencies[positions]
            check(int(frequencies.sum()) == len(patient_ids), f"{scope}/{b}: correct patient draw count")
            for stage in [0, 1, 2]:
                in_stratum = patient_max.to_numpy() == stage
                check(frequencies[in_stratum].sum() == in_stratum.sum(), f"{scope}/{b}: stratum {stage} count preserved")
            for j, (target, name) in enumerate(keys):
                group = wide[wide.target.eq(target)]
                y, p = group.y_true.to_numpy()[mask], group[f"y_prob_{name}"].to_numpy()[mask]
                expected = [average_precision_score(y, p, sample_weight=weights), roc_auc_score(y, p, sample_weight=weights), brier_score_loss(y, p, sample_weight=weights)]
                check(np.allclose(array[b, j, :3], expected, atol=1e-12, rtol=0), f"{scope}/{b}/{target}/{name}: independently reconstructed shared patient draw")
        for row in intervals[intervals.scope.eq(scope)].itertuples():
            values = array[:, keys.index((row.target, row.model_name)), METRIC_NAMES.index(row.metric)]
            valid = values[np.isfinite(values)]
            check(len(valid) == row.valid_replicates, f"{scope}/{row.target}/{row.model_name}/{row.metric}: valid replicate count")
            expected = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
            check(np.allclose(expected, [row.ci_lower, row.ci_upper], atol=1e-12, rtol=0, equal_nan=True), f"{scope}/{row.target}/{row.model_name}/{row.metric}: percentile CI")
        for row in pairs[pairs.scope.eq(scope)].itertuples():
            k = METRIC_NAMES.index(row.metric)
            values = array[:, keys.index((row.target, row.model_a)), k] - array[:, keys.index((row.target, row.model_b)), k]
            valid = values[np.isfinite(values)]
            check(len(valid) == row.valid_replicates and np.allclose(np.quantile(valid, [.025, .975]), [row.ci_lower, row.ci_upper], atol=1e-12, rtol=0),
                  f"{scope}/{row.target}/{row.model_a}-{row.model_b}/{row.metric}: direct paired difference CI")

    combined = pd.read_csv(TABLES / "combined_model_comparison.csv")
    check(len(combined) == 14 and not combined.duplicated(["target", "model_name"]).any(), "Complete fourteen-model comparison table")
    bins = pd.read_csv(TABLES / "test_calibration_bins.csv")
    for target, name in keys:
        frame = bins[bins.target.eq(target) & bins.model_name.eq(name)]
        check(frame.encounters.sum() == len(labels) and frame.positives.sum() == labels[target].sum(), f"{target}/{name}: calibration bins cover all test encounters")
    original = read_json(CHECKS / "original_sha256.json")
    for path, expected in original.items():
        check(resolve_recorded_path(path).is_file() and sha256(resolve_recorded_path(path)) == expected, f"Original file unchanged after path resolution: {path}")
    for path in FIGURES.glob("*.png"):
        check(path.stat().st_size > 10000 and path.with_suffix(".pdf").is_file(), f"PNG and PDF exist: {path.stem}")
    for name in ["README.md", "results_summary_zh.md"]:
        text = (HERE / name).read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if not link.startswith(("http:", "https:")):
                destination = (HERE / link).resolve()
                if destination == (CHECKS / "final_verification.json").resolve():
                    # This check record is committed below, after verification passes.
                    checks.append(f"Relative Markdown link names this verification output: {name} -> {link}")
                else:
                    check(destination.exists(), f"Relative Markdown link resolves: {name} -> {link}")
    write_json(CHECKS / "final_verification.json", {"verified_at_utc": utc_now(), "all_passed": True,
        "checks_passed": len(checks), "checks": checks, "models_refitted": False,
        "primary_artifacts_and_teammate_files_unchanged": True,
        "independent_methods": ["sklearn metrics", "statsmodels GLM", "patient-stratified sample reconstruction", "direct replicate quantiles", "SHA-256"]})
    write_artifact_manifest()
    print(f"Independent verification passed: {len(checks)} checks; {len(original)} original files unchanged.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
