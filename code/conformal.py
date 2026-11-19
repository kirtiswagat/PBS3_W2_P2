"""Conformal prediction layers (H5 core): split and Mondrian binary sets.

Both are calibrated on Train-B and evaluated on Train-C, per invariant 2.
Nonconformity for a binary sigmoid output is the natural "1 - p(that class)":
low when the model was confident and correct, high otherwise.

Mondrian = label-conditional coverage: split conformal repeated separately
within each true class of the calibration set, which matters here because
none of these 14 pathologies sit near 50% prevalence, so a single pooled
threshold would under-cover the minority class.

FNR risk control (the clinically load-bearing guarantee for a screening
task) is deliberately a separate module, risk_control.py - it is the paper's
novel contribution and should not be buried inside the standard-conformal file.
"""

import numpy as np


def binary_nonconformity(p):
    """(n, 2) nonconformity of each row under candidate label 0 and label 1."""
    p = np.asarray(p, float)
    return np.stack([p, 1 - p], axis=1)  # label0 nonconf = p, label1 nonconf = 1-p


def _quantile_level(n, alpha):
    """Finite-sample-corrected conformal quantile level, clipped to [0, 1]."""
    return float(np.clip(np.ceil((n + 1) * (1 - alpha)) / n, 0.0, 1.0))


def split_conformal(cal_scores, test_scores, alpha=0.1):
    """Standard (marginal) split conformal binary prediction sets.

    cal_scores  : (n_cal,) nonconformity of the calibration rows under their
                  TRUE label, e.g. 1 - p_true = binary_nonconformity(p)[cal_y].
    test_scores : (n_test, 2) nonconformity of each test row under each
                  candidate label, e.g. binary_nonconformity(test_p).

    Returns {"threshold", "pred_sets" (n_test, 2) bool, "avg_set_size"}.
    """
    n = len(cal_scores)
    q = _quantile_level(n, alpha)
    threshold = float(np.quantile(cal_scores, q, method="higher"))
    pred_sets = test_scores <= threshold
    return {
        "threshold": threshold,
        "pred_sets": pred_sets,
        "avg_set_size": float(pred_sets.sum(axis=1).mean()),
    }


def mondrian_conformal(cal_scores, cal_y, test_scores, alpha=0.1):
    """Label-conditional (Mondrian) split conformal - one threshold per true class."""
    thresholds = {}
    for c in (0, 1):
        s = cal_scores[cal_y == c]
        if len(s) == 0:
            thresholds[c] = float("inf")
            continue
        q = _quantile_level(len(s), alpha)
        thresholds[c] = float(np.quantile(s, q, method="higher"))

    pred_sets = np.stack([test_scores[:, c] <= thresholds[c] for c in (0, 1)], axis=1)
    return {
        "thresholds": {str(c): t for c, t in thresholds.items()},
        "pred_sets": pred_sets,
        "avg_set_size": float(pred_sets.sum(axis=1).mean()),
    }
