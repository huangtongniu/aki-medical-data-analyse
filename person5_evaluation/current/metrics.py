"""Frozen-threshold metrics and patient-cluster weighted bootstrap primitives."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

METRIC_NAMES = [
    "average_precision", "auroc", "brier_score", "sensitivity", "specificity", "ppv", "npv",
    "false_positive_rate", "flagged_fraction", "false_positives_per_1000",
    "calibration_intercept", "calibration_slope", "calibration_in_the_large",
]


def divide(a, b):
    return float(a / b) if b else float("nan")


def choose_threshold(y, p, sensitivity_target):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not 0 < sensitivity_target <= 1 or y.sum() == 0:
        raise ValueError("A valid sensitivity target and positive observations are required")
    # kth largest positive probability is the largest attainable qualifying cutoff;
    # every tied score is included by the >= decision rule.
    positive_probabilities = np.sort(p[y == 1])[::-1]
    count_needed = int(np.ceil(sensitivity_target * len(positive_probabilities)))
    return float(positive_probabilities[count_needed - 1])


def calibration(y, p, weights, initial=None):
    """Joint logit calibration intercept/slope plus offset-only calibration-in-large.

    These are diagnostic fits only; they never alter prediction probabilities.
    A constant score has no identifiable joint intercept/slope, but has a CITL.
    """
    x = logit(np.clip(p, 1e-12, 1 - 1e-12))
    n = weights.sum()
    positives = np.dot(weights, y)
    if positives <= 0 or positives >= n:
        return np.nan, np.nan, np.nan
    citl = brentq(lambda a: np.dot(weights, expit(x + a)) - positives, -60.0, 60.0, xtol=1e-10)
    if np.ptp(x[weights > 0]) < 1e-12:
        return np.nan, np.nan, float(citl)
    a, b = initial if initial is not None and np.isfinite(initial).all() else (citl, 1.0)
    converged = False
    for _ in range(60):
        eta = a + b * x
        fitted = expit(eta)
        residual = weights * (fitted - y)
        curvature = weights * fitted * (1 - fitted)
        gradient = np.array([residual.sum(), np.dot(residual, x)])
        h00, h01, h11 = curvature.sum(), np.dot(curvature, x), np.dot(curvature, x * x)
        hessian = np.array([[h00, h01], [h01, h11]])
        if np.max(np.abs(gradient)) < 1e-8:
            converged = True
            break
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        loss = np.dot(weights, np.logaddexp(0, eta) - y * eta)
        scale = 1.0
        for _ in range(25):
            new_a, new_b = a - scale * step[0], b - scale * step[1]
            new_eta = new_a + new_b * x
            new_loss = np.dot(weights, np.logaddexp(0, new_eta) - y * new_eta)
            if new_loss <= loss + 1e-10:
                break
            scale *= 0.5
        a, b = float(new_a), float(new_b)
        if np.max(np.abs(scale * step)) < 1e-8:
            converged = True
            break
    return (a, b, float(citl)) if converged else (np.nan, np.nan, float(citl))


class PreparedMetrics:
    """Pre-sort scores once; integer cluster weights exactly represent row replication."""

    def __init__(self, y, p, threshold):
        self.y = np.asarray(y, dtype=float)
        self.p = np.asarray(p, dtype=float)
        if self.y.ndim != 1 or self.y.shape != self.p.shape or not np.isin(self.y, [0, 1]).all():
            raise ValueError("Invalid outcomes/prediction dimensions")
        if not np.isfinite(self.p).all() or not ((self.p >= 0) & (self.p <= 1)).all():
            raise ValueError("Invalid probabilities")
        self.threshold = float(threshold)
        self.flag = self.p >= self.threshold
        self.order = np.argsort(-self.p, kind="stable")
        sorted_p = self.p[self.order]
        self.ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(sorted_p) - 1]
        self.squared_error = (self.y - self.p) ** 2
        self.calibration_initial = None

    def evaluate(self, weights=None, include_calibration=True):
        w = np.ones(len(self.y)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != self.y.shape or (w < 0).any() or w.sum() <= 0:
            raise ValueError("Invalid evaluation weights")
        y, p = self.y, self.p
        positive = np.dot(w, y)
        total = w.sum()
        negative = total - positive
        sorted_positive = (w * y)[self.order]
        sorted_negative = (w * (1 - y))[self.order]
        tps = np.cumsum(sorted_positive)[self.ends]
        fps = np.cumsum(sorted_negative)[self.ends]
        precision = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=tps + fps > 0)
        ap = float(np.dot(np.diff(np.r_[0.0, tps]) / positive, precision)) if positive else np.nan
        auc = float(np.dot(np.diff(np.r_[0.0, fps]), (tps + np.r_[0.0, tps[:-1]]) / 2) / (positive * negative)) if positive and negative else np.nan
        tp = np.dot(w, y * self.flag)
        fp = np.dot(w, (1 - y) * self.flag)
        fn = positive - tp
        tn = negative - fp
        result = {
            "average_precision": ap, "auroc": auc, "brier_score": float(np.dot(w, self.squared_error) / total),
            "sensitivity": divide(tp, positive), "specificity": divide(tn, negative),
            "ppv": divide(tp, tp + fp), "npv": divide(tn, tn + fn),
            "false_positive_rate": divide(fp, negative), "flagged_fraction": divide(tp + fp, total),
            "false_positives_per_1000": float(1000 * fp / total),
            "true_positives": float(tp), "false_positives": float(fp),
            "true_negatives": float(tn), "false_negatives": float(fn),
        }
        if include_calibration:
            a, b, citl = calibration(y, p, w, self.calibration_initial)
            result.update(calibration_intercept=a, calibration_slope=b, calibration_in_the_large=citl)
            if weights is None:
                self.calibration_initial = (a, b)
        else:
            result.update({name: np.nan for name in METRIC_NAMES[-3:]})
        return result
