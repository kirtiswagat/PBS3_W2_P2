"""Selection-rule comparison for external retention (H3 core).

Rule A picks the candidate feature subset with the best Train-B AUC - the
conventional approach. Rule B picks the subset with the best *worst-case*
AUC across internal environments. H3 asks whether B's distributional
robustness carries over to the terminal external set better than A's.

ENVIRONMENTS ARE INTERNAL ONLY
    AP/PA view and age band are read from the CheXpert metadata CSV (the same
    split CSVs already on disk carry these columns, keyed by ``Path``). NIH
    and the expert set are terminal and must never inform a selection
    decision, or Phase 6's retention numbers would be circular.

MATCHED SUBSET SIZE
    Rule A and Rule B must choose from the same candidate_sets pool at the
    same size for each candidate, or a difference in outcome could just be a
    difference in capacity rather than in the selection criterion.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from heads import fit_head
from metrics import per_class_metrics

AGE_BANDS = [(-np.inf, 40), (40, 60), (60, 75), (75, np.inf)]
AGE_BAND_NAMES = ["<40", "40-60", "60-75", ">75"]
CV_FOLDS = 5


def _age_band(age):
    for name, (lo, hi) in zip(AGE_BAND_NAMES, AGE_BANDS):
        if lo <= age < hi:
            return name
    return AGE_BAND_NAMES[-1]


def build_environments(paths, chexpert_csv):
    """Map each row index to an environment key: view x age band.

    Returns {env_name: row_index_array}. Rows whose metadata is missing (age
    NaN, view not AP/PA) are dropped from the environment split entirely
    rather than guessed.
    """
    meta = pd.read_csv(chexpert_csv)
    meta = meta.set_index("Path")

    envs = {}
    for i, p in enumerate(paths):
        row = meta.loc[p] if p in meta.index else None
        if row is None:
            continue
        view = row.get("AP/PA")
        age = row.get("Age")
        if pd.isna(view) or view not in ("AP", "PA") or pd.isna(age):
            continue
        key = f"{view}_{_age_band(float(age))}"
        envs.setdefault(key, []).append(i)

    return {k: np.array(v) for k, v in envs.items()}


def _fit_and_auc(x, y, w, idx, x_eval, y_eval, C):
    clf = fit_head(x[idx], y[idx], w[idx], C=C)
    if clf is None:
        return None
    p = clf.predict_proba(x_eval)[:, 1]
    m = per_class_metrics(y_eval, p)
    return m["auc"] if m else None


def rule_a_auc(x, y, w, candidate_sets, C):
    """Best cross-validated AUC among the candidate feature subsets.

    x, y, w are Train-B (the split calibration/thresholds already use, per
    invariant 2) - conventional selection never looks past it. Stratified
    k-fold on Train-B alone gives each candidate a fair AUC estimate without
    needing a further held-out split.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
    scores = []
    for cand in candidate_sets:
        xs = x[:, cand]
        fold_aucs = []
        for train_idx, test_idx in skf.split(xs, y):
            auc = _fit_and_auc(xs, y, w, train_idx, xs[test_idx], y[test_idx], C)
            if auc is not None:
                fold_aucs.append(auc)
        scores.append(np.mean(fold_aucs) if fold_aucs else -np.inf)
    return candidate_sets[int(np.argmax(scores))]


def rule_b_worst_env(x, y, w, candidate_sets, envs, C):
    """Best worst-environment AUC among the candidate feature subsets.

    For each candidate subset, fit on all rows outside an environment and
    score on that environment (leave-one-environment-out), then take the
    minimum AUC across environments. The winning subset maximises that
    minimum - the distributional-robustness analogue of Rule A's plain CV AUC.
    """
    worst_scores = []
    for cand in candidate_sets:
        xs = x[:, cand]
        env_aucs = []
        for env_idx in envs.values():
            if len(env_idx) < 10:
                continue
            train_idx = np.setdiff1d(np.arange(len(y)), env_idx)
            auc = _fit_and_auc(xs, y, w, train_idx, xs[env_idx], y[env_idx], C)
            if auc is not None:
                env_aucs.append(auc)
        worst_scores.append(min(env_aucs) if env_aucs else -np.inf)
    return candidate_sets[int(np.argmax(worst_scores))]
