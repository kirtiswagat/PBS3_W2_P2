"""Baselines answering: is the selector doing real work, or would any k features do?

Three controls at matched subset size k, used alongside select_l1's output in
phase4_selection.py:
    top_k_mi        plain univariate ranking, no redundancy awareness
    pca_projection  k learned linear combinations, not a feature subset
    random_subsets  the null - if random draws match the selector, it isn't
                    selecting anything meaningful
"""

import numpy as np
from sklearn.decomposition import PCA

from filters import rank_mutual_info


def top_k_mi(x, y, k):
    return rank_mutual_info(x, y, top_k=k)


def pca_projection(x_train, x_eval, k):
    """Fit PCA on x_train only; project both. Returns (train_proj, eval_proj)."""
    pca = PCA(n_components=k, random_state=0)
    train_proj = pca.fit_transform(x_train)
    eval_proj = pca.transform(x_eval)
    return train_proj, eval_proj


def random_subsets(n_features, k, n_draws=20, seed=0):
    """n_draws independent random index draws of size k.

    Uses its own RNG stream, independent of the patient-subsample seeds in
    heads.subsample_patients, so shared seed values never couple the two.
    """
    rng = np.random.default_rng(seed)
    return [rng.choice(n_features, size=k, replace=False) for _ in range(n_draws)]
