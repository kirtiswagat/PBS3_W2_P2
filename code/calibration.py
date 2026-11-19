"""Per-disease calibration and threshold selection.

Both are fitted on Train-B only and never on the expert set. Work-1 was
criticised for tuning and testing on the same 234 images; keeping every tunable
quantity on Train-B is the structural answer to that.
"""

import numpy as np
from sklearn.metrics import f1_score


def fit_temperature(y, p, grid=None):
    """Single-parameter temperature scaling on the logit scale.

    A coarse-to-fine grid search is used rather than gradient descent: the
    objective is 1-D and smooth, and a grid cannot silently fail to converge.
    """
    eps = 1e-7
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    grid = (
        grid
        if grid is not None
        else np.concatenate([np.arange(0.05, 2.0, 0.05), np.arange(2.0, 10.1, 0.25)])
    )

    best_t, best_nll = 1.0, np.inf
    for t in grid:
        q = 1.0 / (1.0 + np.exp(-logit / t))
        q = np.clip(q, eps, 1 - eps)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def apply_temperature(p, t):
    eps = 1e-7
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    return 1.0 / (1.0 + np.exp(-logit / t))


def fit_threshold(y, p):
    """Operating threshold maximising F1 on Train-B.

    A fixed 0.5 is meaningless for low-prevalence classes: Consolidation at
    15.8% prevalence scored AUC 0.854 with recall 0.0 at 0.5 in Tier 1.
    """
    if len(np.unique(y)) < 2:
        return 0.5
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        f1 = f1_score(y, (p >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr
