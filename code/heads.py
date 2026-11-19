"""Binary-relevance heads over the cached embeddings.

Design decisions and their reasons:

HEAD = LOGISTIC REGRESSION, FIXED REGULARISATION
    The classifier head of the frozen trunk is provably an affine map of the
    512-d embedding (verified: max |reconstruction - cached preds| = 2.5e-07),
    so a linear head is the natural probe rather than an arbitrary choice.
    More importantly, C is tuned once and then held fixed across every
    uncertainty policy. Work-1 was criticised for tuning asymmetry - a
    fixed-hyperparameter CNN compared against a Bayesian-tuned HybridViT - and
    holding C fixed means any AUC difference between policies is attributable
    to the policy alone.

SEEDS RESAMPLE PATIENTS, THEY DO NOT RE-INITIALISE THE MODEL
    Logistic regression is convex and deterministic, so re-running it with a
    different RNG seed returns an identical solution and would report zero
    variance. Seeds therefore select a patient-level subsample of Train-A.
    The resulting mean +/- SD reflects sensitivity to the training sample,
    which is what the single-seed criticism of Work-1 was actually about.
    Subsampling is by patient, not by row, because CheXpert has ~3 images per
    patient and row-level resampling would understate the variance.

UNCERTAINTY POLICIES UNDER A LINEAR HEAD
    Focal-style losses have no direct analogue in logistic regression, so the
    two soft policies are expressed through sample weights instead:
        u_ignore   drop uncertain rows entirely (mask, not relabel)
        u_ones     -1 -> 1, full weight
        u_zeros    -1 -> 0, full weight
        u_soft     -1 -> 1, half weight        (downweighted positive)
        u_both     -1 -> duplicated as 0 and 1, half weight each
                   (equivalent to a soft label of 0.5, i.e. label smoothing)
"""

import os
import re
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import trunks

# Derived from this file's location so the code is portable; override with PS3_BASE.
BASE = os.environ.get("PS3_BASE") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
EMB_DIR = trunks.emb_dir(BASE, "densenet121")


def set_trunk(name):
    """Point every later load_split() at this trunk's cache."""
    global EMB_DIR
    EMB_DIR = trunks.emb_dir(BASE, name)
    return EMB_DIR


POLICIES = ["u_ignore", "u_ones", "u_zeros", "u_soft", "u_both"]

# Work-1 used a per-disease policy, not a global rule. Tier 1 must replicate it
# exactly or the Gate-2 comparison against 0.7892 is meaningless.
WORK1_POLICY = {
    "Atelectasis": "u_ones",
    "Edema": "u_ones",
    "Cardiomegaly": "u_zeros",
    "Consolidation": "u_zeros",
    "Pleural Effusion": "u_zeros",
}

HEAD_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def load_split(name):
    d = np.load(f"{EMB_DIR}/{name}.npz", allow_pickle=True)
    # The DenseNet cache predates the trunk-agnostic key names.
    backbone = "emb_backbone" if "emb_backbone" in d.files else "emb_1024"
    dense = "emb_dense" if "emb_dense" in d.files else "emb_512"
    return {
        "emb_backbone": d[backbone],
        "emb_dense": d[dense],
        "labels": d["labels"],
        "label_names": [str(x) for x in d["label_names"]],
        "paths": [str(p) for p in d["paths"]],
        "preds": d["preds"] if "preds" in d.files else None,
    }


def patient_ids(paths):
    return np.array(
        [m.group(1) if (m := re.search(r"(patient\d+)", p)) else p for p in paths]
    )


class Standardizer:
    """Fitted on Train-A only; ReLU activations span 0-20 and lbfgs needs scaling."""

    def fit(self, x):
        self.mean_ = x.mean(0)
        self.scale_ = x.std(0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_


def apply_policy(y_raw, policy):
    """Map raw labels (1, 0, -1, NaN) to (indices, targets, weights).

    NaN means the pathology was never mentioned in the report, which CheXpert
    convention treats as negative.
    """
    y = np.nan_to_num(y_raw, nan=0.0)
    idx = np.arange(len(y))
    unc = y == -1

    if policy == "u_ignore":
        keep = ~unc
        return idx[keep], y[keep], np.ones(keep.sum(), np.float32)

    if policy in ("u_ones", "u_zeros", "u_soft"):
        target = y.copy()
        target[unc] = 1.0 if policy in ("u_ones", "u_soft") else 0.0
        w = np.ones(len(y), np.float32)
        if policy == "u_soft":
            w[unc] = 0.5
        return idx, target, w

    if policy == "u_both":
        # Duplicate each uncertain row as both classes at half weight.
        certain = ~unc
        idx_out = np.concatenate([idx[certain], idx[unc], idx[unc]])
        y_out = np.concatenate([y[certain], np.ones(unc.sum()), np.zeros(unc.sum())])
        w_out = np.concatenate(
            [
                np.ones(certain.sum(), np.float32),
                np.full(unc.sum(), 0.5, np.float32),
                np.full(unc.sum(), 0.5, np.float32),
            ]
        )
        return idx_out, y_out, w_out

    raise ValueError(f"unknown policy: {policy}")


def subsample_patients(paths, seed, fraction=0.8):
    """Patient-level subsample. Returns row indices."""
    pids = patient_ids(paths)
    unique = np.unique(pids)
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(unique, size=int(len(unique) * fraction), replace=False))
    return np.flatnonzero(np.isin(pids, list(chosen)))


# lbfgs needs more iterations on 2048-d Xception features than on 1024-d
# DenseNet ones. Rather than silencing the ConvergenceWarning, count how many
# fits hit the cap so a non-converged run can never pass unnoticed into a result.
CONVERGENCE = {"fits": 0, "nonconverged": 0, "max_iter_seen": 0}


def reset_convergence():
    CONVERGENCE.update(fits=0, nonconverged=0, max_iter_seen=0)


def convergence_report():
    c = CONVERGENCE
    pct = 100 * c["nonconverged"] / c["fits"] if c["fits"] else 0.0
    status = "OK" if c["nonconverged"] == 0 else "WARNING - raise --max-iter"
    return (
        f"lbfgs: {c['fits']} fits, {c['nonconverged']} hit the iteration cap "
        f"({pct:.1f}%), max iterations used {c['max_iter_seen']}  [{status}]"
    )


def fit_head(x, y, w, C=1.0, max_iter=2000):
    if len(np.unique(y)) < 2:
        return None  # degenerate class under this policy
    # n_jobs is not passed: it is ignored by lbfgs for binary problems and is
    # deprecated in recent sklearn, which otherwise floods the log.
    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(x, y, sample_weight=w)

    n_iter = int(np.max(clf.n_iter_))
    CONVERGENCE["fits"] += 1
    CONVERGENCE["max_iter_seen"] = max(CONVERGENCE["max_iter_seen"], n_iter)
    if n_iter >= max_iter:
        CONVERGENCE["nonconverged"] += 1
    return clf
