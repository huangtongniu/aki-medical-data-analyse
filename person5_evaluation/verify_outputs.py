"""Independent checks on frozen outputs; no model fitting or test-driven changes."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import re
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits

from common import (CHECKS, FIGURES, HERE, MODEL_NAMES, P2, P2_QA, PREDICTIONS, ROOT,
                    TABLES, check_frozen_sources, read_json, require, settings, sha256,
                    utc_now, write_json)
from metrics import METRIC_NAMES


def main():
    checks = []

    def check(condition, message):
        require(condition, message)
        checks.append(message)

    cfg = settings()
    freeze = read_json(CHECKS / "analysis_freeze.json")
    check_frozen_sources(freeze)
    checks.append("All frozen source hashes unchanged")
    started = read_json(CHECKS / "test_run_started.json")
    check(datetime.fromisoformat(freeze["created_at_utc"]) < datetime.fromisoformat(started["started_at_utc"]), "Freeze predates test evaluation")
    check(freeze["protocol"] == cfg and cfg["sensitivity_target"] == .8, "User-confirmed 0.80 sensitivity protocol unchanged")
    check(read_json(CHECKS / "statistical_unit_checks.json")["successful"], "Six independent statistical/input checks passed")
    check(sha256(TABLES / "validation_thresholds.csv") == freeze["threshold_file_sha256"], "Frozen threshold CSV unchanged")
    wide = pd.read_csv(PREDICTIONS / "test_predictions_all_models.csv", float_precision="round_trip")
    long = pd.read_csv(PREDICTIONS / "test_predictions_long.csv", float_precision="round_trip")
    label = pd.read_csv(P2 / "outputs/matrices/labels_test.csv")
    test = pd.read_csv(TABLES / "test_metrics.csv", float_precision="round_trip").set_index("model_name")
    check((len(wide), wide.subject_id.nunique(), int(wide.y_true.sum())) == (16801, 9914, 1155), "Test denominators agree with frozen split")
    check(np.array_equal(wide.stay_id, label.stay_id) and np.array_equal(wide.subject_id, label.subject_id), "Test IDs and row order match P2")
    check(np.array_equal(wide.y_true, label.aki_label), "Test outcomes match P2")
    check(len(long) == 7 * len(wide) and not long.duplicated(["stay_id", "model_name", "target"]).any(), "Exactly one long-table test prediction per model/encounter/target")
    check(long.target.eq("aki_label").all() and long.partition.eq("test").all(), "All test predictions have primary target and test partition")
    y = wide.y_true.to_numpy()
    for name in MODEL_NAMES:
        p = wide[f"y_prob_{name}"].to_numpy()
        group = long[long.model_name.eq(name)].set_index("stay_id").loc[wide.stay_id]
        check(np.array_equal(p, group.predicted_probability) and np.array_equal(group.subject_id, wide.subject_id), f"{name}: long/wide probabilities and patients agree")
        expected = [average_precision_score(y, p), roc_auc_score(y, p), brier_score_loss(y, p)]
        actual = test.loc[name, ["average_precision", "auroc", "brier_score"]].to_numpy(dtype=float)
        check(np.allclose(actual, expected, atol=1e-12, rtol=0), f"{name}: test AP/AUROC/Brier independently reproduced")
        cutoff = freeze["thresholds"][name]
        check(test.loc[name, "threshold"] == cutoff, f"{name}: test cutoff equals validation freeze exactly")
        tn, fp, fn, tp = confusion_matrix(y, p >= cutoff, labels=[0, 1]).ravel()
        check(np.array_equal([tp, fp, tn, fn], test.loc[name, ["true_positives", "false_positives", "true_negatives", "false_negatives"]]), f"{name}: threshold confusion counts reproduced")
        x = logit(np.clip(p, 1e-12, 1 - 1e-12))
        offset = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=x).fit()
        check(abs(offset.params[0] - test.loc[name, "calibration_in_the_large"]) < 1e-7, f"{name}: CITL agrees with independent statsmodels GLM")
        if name != "prevalence":
            fitted = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
            check(np.allclose(fitted.params, test.loc[name, ["calibration_intercept", "calibration_slope"]].to_numpy(dtype=float), atol=1e-7, rtol=0), f"{name}: joint calibration agrees with independent statsmodels GLM")
    validation = pd.read_csv(PREDICTIONS / "validation_predictions_wide.csv", float_precision="round_trip")
    for name in MODEL_NAMES:
        cutoff = freeze["thresholds"][name]
        positive_scores = validation.loc[validation.y_true.eq(1), f"y_prob_{name}"].to_numpy()
        check((positive_scores >= cutoff).mean() >= .8 and (positive_scores >= np.nextafter(cutoff, np.inf)).mean() < .8,
              f"{name}: frozen cutoff is largest score satisfying validation sensitivity")
    sensitivity = pd.read_csv(TABLES / "sensitivity_metrics.csv", float_precision="round_trip")
    flags = pd.read_csv(P2_QA / "outputs/sensitivity_subset_flags.csv").set_index("stay_id").loc[wide.stay_id]
    all_ci = pd.read_csv(TABLES / "all_confidence_intervals.csv", float_precision="round_trip")
    all_pairs = pd.read_csv(TABLES / "all_paired_differences.csv", float_precision="round_trip")
    alpha = (1 - cfg["confidence_level"]) / 2
    for scope_index, scope in enumerate(["test"] + cfg["subsets"]):
        subset_mask = np.ones(len(wide), dtype=bool) if scope == "test" else flags[scope].to_numpy(dtype=bool)
        scope_wide = wide.loc[subset_mask].reset_index(drop=True)
        scope_point = test if scope == "test" else sensitivity[sensitivity.scope.eq(scope)].set_index("model_name")
        check(scope_point.encounters.eq(len(scope_wide)).all() and scope_point.patients.eq(scope_wide.subject_id.nunique()).all(), f"{scope}: subgroup denominators reproduced")
        check(all(scope_point.loc[name, "threshold"] == freeze["thresholds"][name] for name in MODEL_NAMES), f"{scope}: original validation cutoffs retained")
        samples = np.load(CHECKS / f"bootstrap_{scope}.npy", allow_pickle=False)
        progress = read_json(CHECKS / f"bootstrap_{scope}_progress.json")
        check(samples.shape == (1000, 7, 13) and progress["completed"] == 1000, f"{scope}: all 1000 bootstrap replicates completed")
        patients, inv = np.unique(scope_wide.subject_id, return_inverse=True)
        # Independent sklearn evaluation on selected actual patient resamples.
        for b in [0, 499, 999]:
            rng = np.random.default_rng(np.random.SeedSequence([cfg["bootstrap_seed"], scope_index, b]))
            w = np.bincount(rng.integers(len(patients), size=len(patients)), minlength=len(patients))[inv]
            sy = scope_wide.y_true.to_numpy()
            for j, name in enumerate(MODEL_NAMES):
                p = scope_wide[f"y_prob_{name}"].to_numpy()
                expected = [average_precision_score(sy, p, sample_weight=w), roc_auc_score(sy, p, sample_weight=w), brier_score_loss(sy, p, sample_weight=w)]
                check(np.allclose(samples[b, j, :3], expected, atol=1e-12, rtol=0), f"{scope}/{b}/{name}: bootstrap metrics reproduce identical patient weights")
        for row in all_ci[all_ci.scope.eq(scope)].itertuples():
            values = samples[:, MODEL_NAMES.index(row.model_name), METRIC_NAMES.index(row.metric)]
            values = values[np.isfinite(values)]
            expected = np.quantile(values, [alpha, 1 - alpha]) if len(values) else [np.nan, np.nan]
            check(np.allclose(expected, [row.ci_lower, row.ci_upper], atol=1e-12, rtol=0, equal_nan=True), f"{scope}/{row.model_name}/{row.metric}: CI is saved-replicate percentile")
        for row in all_pairs[all_pairs.scope.eq(scope)].itertuples():
            k = METRIC_NAMES.index(row.metric)
            values = samples[:, MODEL_NAMES.index(row.model_a), k] - samples[:, MODEL_NAMES.index(row.model_b), k]
            expected = np.quantile(values, [alpha, 1 - alpha])
            check(np.allclose(expected, [row.ci_lower, row.ci_upper], atol=1e-12, rtol=0), f"{scope}/{row.model_a}-{row.model_b}/{row.metric}: paired CI uses direct shared-resample differences")
    before = read_json(CHECKS / "upstream_original_sha256.json")
    for path, expected in before.items():
        check(sha256(ROOT / path) == expected, f"Original file unchanged: {path}")
    for path in [HERE / "report_draft.md", HERE / "person5_work_summary_zh.md", HERE / "README.md"]:
        text = path.read_text(encoding="utf-8")
        check("\ufffd" not in text and sum(line.startswith("```") for line in text.splitlines()) % 2 == 0, f"{path.name}: UTF-8 text and balanced fences")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#")):
                continue
            check((path.parent / unquote(target)).is_file(), f"{path.name}: relative link resolves: {target}")
    check(len(list(FIGURES.glob("*.png"))) == 7 and len(list(FIGURES.glob("*.pdf"))) == 7, "Seven PNG and vector-PDF figure pairs exist")
    write_json(CHECKS / "final_verification.json", {"verified_at_utc": utc_now(), "checks_passed": len(checks),
               "checks": checks, "all_passed": True, "model_fitting_performed": False,
               "independent_methods": ["sklearn metrics and confusion counts", "statsmodels calibration GLM", "saved-replicate quantiles", "patient-sampling reconstruction", "SHA-256 integrity"]})
    # Relative paths only; this also makes future artifact changes visible.
    inventory = {p.relative_to(HERE).as_posix(): sha256(p) for p in sorted(HERE.rglob("*")) if p.is_file()
                 and p.name != "artifact_sha256.json" and "__pycache__" not in p.parts}
    write_json(CHECKS / "artifact_sha256.json", inventory)
    print(f"Final verification passed: {len(checks)} checks; no model fitting or original-file edits.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
