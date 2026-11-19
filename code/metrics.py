"""Evaluation metrics for Tier 1 onward.

Per-class, joint and calibration metrics in one place so every phase reports
the same quantities. Bootstrap CIs use the percentile method over resampled
evaluation rows.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y, p, n_bins=10):
    """Equal-width binning, as used in the Work-1 calibration analysis."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def per_class_metrics(y, p, threshold=0.5):
    if len(np.unique(y)) < 2:
        return None
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "recall": float(recall_score(y, (p >= threshold).astype(int), zero_division=0)),
        "ece": expected_calibration_error(y, p),
        "brier": float(brier_score_loss(y, p)),
        "n_pos": int(y.sum()),
        "n": int(len(y)),
    }


def joint_metrics(y_mat, p_mat, threshold=0.5):
    """Binary Relevance can score well per-class yet poorly jointly; the size of
    that gap is itself a reportable measure of how much label correlation matters."""
    pred = (p_mat >= threshold).astype(int)
    return {
        "hamming_loss": float((pred != y_mat).mean()),
        "exact_match": float((pred == y_mat).all(axis=1).mean()),
    }


def bootstrap_ci(y, p, metric="auc", n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    fn = roc_auc_score if metric == "auc" else average_precision_score
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarise(per_seed):
    """Mean and SD across seeds for a list of per-class metric dicts."""
    keys = per_seed[0].keys()
    return {
        k: {
            "mean": float(np.mean([s[k] for s in per_seed])),
            "sd": float(np.std([s[k] for s in per_seed], ddof=1))
            if len(per_seed) > 1
            else 0.0,
        }
        for k in keys
    }
