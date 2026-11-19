"""Convex sparse feature selector (H2 core).

Replaces NSGA-II, which is out of scope for the deadline: L1-penalised
logistic regression is convex, deterministic, and gives a directly usable
support set (its non-zero coefficients) instead of a Pareto front that still
needs a downstream selection rule.

WHY BINARY SEARCH ON C
    liblinear's L1 path does not expose "give me exactly k non-zero
    coefficients" - only a penalty strength that indirectly controls sparsity.
    Binary search on C (log-scale) converges on a support size within the
    requested tolerance in a handful of fits rather than a dense grid.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

DEFAULT_C_GRID = (1e-4, 10.0)


def _n_nonzero(x, y, w, C, max_iter):
    # sklearn 1.9 deprecates penalty=; l1_ratio=1 is the supported spelling of pure L1.
    clf = LogisticRegression(
        l1_ratio=1, solver="liblinear", C=C, max_iter=max_iter
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(x, y, sample_weight=w)
    nonzero = np.flatnonzero(clf.coef_[0] != 0)
    return nonzero, clf


def select_l1(
    x,
    y,
    w,
    target_fraction=0.25,
    C_grid=None,
    tol=0.10,
    max_iter=2000,
    max_steps=20,
):
    """Binary-search C until the non-zero count is within +/-tol of target.

    Returns {"indices", "n_selected", "C_used"}. Warns (via the return dict's
    absence of a clean convergence) rather than raising if the target band is
    never reached in max_steps - the caller can still use the closest result.
    """
    n_features = x.shape[1]
    target = max(1, int(round(target_fraction * n_features)))
    lo, hi = C_grid if C_grid else DEFAULT_C_GRID

    best = None  # (abs distance to target, indices, C)
    reached = False
    for _ in range(max_steps):
        c = np.sqrt(lo * hi)  # geometric midpoint, C spans orders of magnitude
        nonzero, _ = _n_nonzero(x, y, w, c, max_iter)
        n = len(nonzero)
        dist = abs(n - target)
        if best is None or dist < best[0]:
            best = (dist, nonzero, c)
        if dist <= tol * target:
            reached = True
            break
        # More non-zeros needs weaker penalty (larger C); fewer needs smaller C.
        if n < target:
            lo = c
        else:
            hi = c

    if not reached:
        print(
            f"select_l1: WARNING - target not reached within tol={tol}; "
            f"closest n_selected={len(best[1])} vs target={target} at C={best[2]:.4g}"
        )

    return {
        "indices": best[1],
        "n_selected": int(len(best[1])),
        "C_used": float(best[2]),
    }
