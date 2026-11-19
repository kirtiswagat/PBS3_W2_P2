"""Univariate and redundancy-aware feature ranking (H2 support).

No CLI - imported by phase4_selection.py, which needs a ranked feature order
per disease before select_l1 (group_lasso.py) sparsifies it further.

MRMR PRESELECTS BY MUTUAL INFORMATION
    Full mRMR scores every candidate pair's redundancy, which is O(p^2) and not
    affordable at p=2048 (Xception). Preselecting the top n_candidates by plain
    MI first, then running the greedy relevance-minus-redundancy loop only
    inside that pool, keeps the cost bounded while still returning a
    redundancy-aware order.

SUBSAMPLING FOR MI
    sklearn's mutual_info_classif uses a k-NN estimator that is O(n log n) per
    feature. The rank stabilises well before the full Train-A size, so MI-based
    ranking subsamples to ~20,000 rows; ANOVA F is closed-form and does not
    need it.
"""

import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

MI_SUBSAMPLE = 20_000


def _subsample(x, y, n, seed=0):
    if len(y) <= n:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return x[idx], y[idx]


def rank_mutual_info(x, y, top_k=None, seed=0):
    """Indices into x's columns, best (highest MI with y) first."""
    xs, ys = _subsample(x, y, MI_SUBSAMPLE, seed)
    scores = mutual_info_classif(xs, ys, random_state=seed)
    order = np.argsort(scores)[::-1]
    return order[:top_k] if top_k else order


def rank_anova_f(x, y, top_k=None):
    """Indices into x's columns, best (highest ANOVA F vs y) first."""
    f_scores, _ = f_classif(x, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    order = np.argsort(f_scores)[::-1]
    return order[:top_k] if top_k else order


def rank_mrmr(x, y, top_k, n_candidates=500, seed=0):
    """Greedy max-relevance min-redundancy order, best first.

    Redundancy is the mean absolute Pearson correlation between a candidate
    and the features already selected - cheap relative to mutual information
    between features, and adequate since x is standardised.
    """
    n_candidates = min(n_candidates, x.shape[1])
    top_k = min(top_k, n_candidates)
    candidates = rank_mutual_info(x, y, top_k=n_candidates, seed=seed)

    xs, ys = _subsample(x, y, MI_SUBSAMPLE, seed)
    relevance = mutual_info_classif(xs[:, candidates], ys, random_state=seed)
    relevance = dict(zip(candidates.tolist(), relevance.tolist()))

    xc = xs[:, candidates]
    # Column-standardised so dot products give Pearson correlation directly.
    xc = (xc - xc.mean(0)) / np.where(xc.std(0) < 1e-8, 1.0, xc.std(0))
    corr = np.abs(xc.T @ xc) / len(xc)

    remaining = list(candidates)
    selected = []
    pos = {c: i for i, c in enumerate(candidates)}

    first = max(remaining, key=lambda c: relevance[c])
    selected.append(first)
    remaining.remove(first)

    while len(selected) < top_k and remaining:
        best_c, best_score = None, -np.inf
        for c in remaining:
            redundancy = np.mean([corr[pos[c], pos[s]] for s in selected])
            score = relevance[c] - redundancy
            if score > best_score:
                best_c, best_score = c, score
        selected.append(best_c)
        remaining.remove(best_c)

    return np.array(selected)
