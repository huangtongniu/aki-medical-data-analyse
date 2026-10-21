"""Meaningful statistical and input-representation checks; no held-out test data."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from inference import dense_to_explicit_csr, rebuild_tree_text
from metrics import PreparedMetrics, calibration, choose_threshold
from evaluate import patient_layout, draw_multiplicities


class EvaluationChecks(unittest.TestCase):
    def test_stratified_patient_draws_keep_whole_mixed_outcome_clusters(self):
        labels = pd.DataFrame({"subject_id": [10, 10, 20, 30, 30, 40, 50, 60],
            "aki_label": [0, 1, 0, 1, 1, 0, 1, 1],
            "aki_stage2_label": [0, 0, 0, 0, 1, 0, 0, 1]})
        patients, inverse, stages, groups = patient_layout(labels)
        np.testing.assert_array_equal(stages, [1, 0, 2, 0, 1, 2])
        for b in range(15):
            counts = draw_multiplicities(groups, len(patients), 9008905, 0, b)
            self.assertEqual(counts.sum(), len(patients))
            for group in groups:
                self.assertEqual(counts[group].sum(), len(group))
            weights = counts[inverse]
            self.assertEqual(weights[0], weights[1])
            self.assertEqual(weights[3], weights[4])
            repeated = np.repeat(np.arange(len(labels)), weights)
            p = np.array([.1, .4, .2, .3, .6, .2, .5, .8])
            for target in ["aki_label", "aki_stage2_label"]:
                y = labels[target].to_numpy()
                got = PreparedMetrics(y, p, .3).evaluate(weights, include_calibration=False)
                self.assertAlmostEqual(got["average_precision"], average_precision_score(y[repeated], p[repeated]), places=13)
                self.assertAlmostEqual(got["auroc"], roc_auc_score(y[repeated], p[repeated]), places=13)

    def test_secondary_cutoff_is_selected_from_its_own_labels(self):
        p = np.array([.9, .8, .7, .6, .5, .4, .3, .2])
        primary = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        secondary = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(choose_threshold(primary, p, .8), .6)
        self.assertEqual(choose_threshold(secondary, p, .8), .8)

    def test_undefined_ranking_statistic_remains_missing(self):
        got = PreparedMetrics(np.array([0, 0]), np.array([.1, .2]), .5).evaluate()
        self.assertTrue(np.isnan(got["auroc"]))
        self.assertTrue(np.isnan(got["average_precision"]))
        self.assertTrue(np.isnan(got["ppv"]))

    def test_weighted_tied_scores_equal_explicit_patient_replication(self):
        y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        p = np.array([.1, .3, .3, .8, .8, .1, .3, .5])
        patient = np.array([0, 0, 1, 1, 1, 2, 2, 3])
        multiplicities = np.array([2, 0, 3, 1])
        weights = multiplicities[patient]
        repeated = np.repeat(np.arange(len(y)), weights)
        result = PreparedMetrics(y, p, .3).evaluate(weights)
        self.assertAlmostEqual(result["average_precision"], average_precision_score(y[repeated], p[repeated]), places=14)
        self.assertAlmostEqual(result["auroc"], roc_auc_score(y[repeated], p[repeated]), places=14)
        self.assertAlmostEqual(result["brier_score"], brier_score_loss(y[repeated], p[repeated]), places=14)
        expanded = PreparedMetrics(y[repeated], p[repeated], .3).evaluate()
        for key in ["sensitivity", "specificity", "ppv", "npv", "flagged_fraction"]:
            self.assertAlmostEqual(result[key], expanded[key], places=8)
        # The two likelihoods are mathematically identical; their numerical
        # optimizers can stop at slightly different coefficients near separation.
        for key in ["calibration_intercept", "calibration_slope", "calibration_in_the_large"]:
            np.testing.assert_allclose(result[key], expanded[key], atol=1e-7, rtol=1e-8)

    def test_largest_sensitivity_cutoff_preserves_ties(self):
        y = np.array([1, 1, 1, 1, 1, 0, 0])
        p = np.array([.9, .8, .7, .7, .1, .7, .5])
        cutoff = choose_threshold(y, p, .8)
        self.assertEqual(cutoff, .7)
        self.assertGreaterEqual((p[y == 1] >= cutoff).mean(), .8)
        self.assertLess((p[y == 1] >= np.nextafter(cutoff, np.inf)).mean(), .8)

    def test_constant_baseline_diagnostics_are_explicit(self):
        y = np.array([0, 0, 0, 1])
        p = np.full(4, .2)
        result = PreparedMetrics(y, p, choose_threshold(y, p, .8)).evaluate()
        self.assertEqual(result["average_precision"], .25)
        self.assertEqual(result["auroc"], .5)
        self.assertEqual(result["sensitivity"], 1)
        self.assertEqual(result["specificity"], 0)
        self.assertTrue(np.isnan(result["npv"]))
        self.assertTrue(np.isnan(result["calibration_slope"]))
        self.assertAlmostEqual(result["calibration_in_the_large"], logit(.25) - logit(.2), places=8)

    def test_explicit_structured_zeros_and_nans_survive_text_merge(self):
        x = np.array([[0., 2., np.nan], [1., 0., -1.]])
        ordinary = sparse.csr_matrix(x)
        text = sparse.csr_matrix([[0., .5], [.1, 0.]])
        original = sparse.hstack([ordinary, text], format="csr")
        explicit = dense_to_explicit_csr(x)
        self.assertEqual(explicit.nnz, x.size)
        rebuilt = rebuild_tree_text(x, original)
        self.assertEqual(rebuilt[:, :3].nnz, x.size)
        np.testing.assert_array_equal(rebuilt[:, :3].toarray(), x)
        np.testing.assert_array_equal(rebuilt[:, 3:].toarray(), text.toarray())

    def test_calibration_matches_independent_unpenalized_glm(self):
        rng = np.random.default_rng(90089)
        x = rng.normal(-2, .8, 500)
        p = expit(x)
        y = rng.binomial(1, expit(.3 + .8 * x))
        weights = rng.integers(0, 4, len(y))
        a, b, citl = calibration(y, p, weights)
        glm = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial(), freq_weights=weights).fit()
        offset = sm.GLM(y, np.ones((len(y), 1)), offset=x, family=sm.families.Binomial(), freq_weights=weights).fit()
        np.testing.assert_allclose([a, b], glm.params, atol=1e-7)
        self.assertAlmostEqual(citl, offset.params[0], places=7)

    def test_random_weighted_metrics_against_sklearn(self):
        rng = np.random.default_rng(42)
        for _ in range(12):
            y = rng.binomial(1, .2, 300)
            p = np.round(rng.uniform(0, 1, 300), 1)
            w = rng.integers(0, 5, 300)
            got = PreparedMetrics(y, p, .5).evaluate(w, include_calibration=False)
            self.assertAlmostEqual(got["average_precision"], average_precision_score(y, p, sample_weight=w), places=13)
            self.assertAlmostEqual(got["auroc"], roc_auc_score(y, p, sample_weight=w), places=13)
            self.assertAlmostEqual(got["brier_score"], brier_score_loss(y, p, sample_weight=w), places=13)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        from common import CHECKS, make_output_dirs, write_json
        make_output_dirs()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        write_json(CHECKS / "statistical_unit_checks.json", {
            "tests_run": result.testsRun, "successful": result.wasSuccessful(),
            "failures": [name.id() for name, _ in result.failures],
            "errors": [name.id() for name, _ in result.errors], "test_data_read": False,
        })
        raise SystemExit(0 if result.wasSuccessful() else 1)
