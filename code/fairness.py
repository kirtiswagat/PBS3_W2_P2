"""Subgroup fairness metrics (Phase 6): AUC gap and equalised-odds gap.

Subgroups reuse the age bands from selection_rules.py (<40, 40-60, 60-75, >75)
so H3's environments and Phase 6's fairness report use one consistent
definition throughout. CheXpert supplies Sex directly; NIH's
Data_Entry_2017.csv supplies "Patient Age" and "Patient Gender" under
different column names, joined by "Image Index" against the NIH cache paths.

NARROWING, DECLARED EXPLICITLY
    Race and insurance status would need MIMIC-IV, which is unavailable to
    this study. That is a real limitation of the fairness analysis and must be
    stated in the paper, not silently dropped.
"""

import numpy as np
import pandas as pd

DECLARED_NARROWING = (
    "Race and insurance status are not analysed - they require MIMIC-IV, "
    "which is unavailable to this study. Fairness results below cover only "
    "sex and age."
)

AGE_BANDS = [(-np.inf, 40), (40, 60), (60, 75), (75, np.inf)]
AGE_BAND_NAMES = ["<40", "40-60", "60-75", ">75"]


def _age_band(age):
    for name, (lo, hi) in zip(AGE_BAND_NAMES, AGE_BANDS):
        if lo <= age < hi:
            return name
    return AGE_BAND_NAMES[-1]


def chexpert_groups(paths, chexpert_csv):
    """Row -> {"sex": ..., "age_band": ...} for CheXpert paths, via the split CSV."""
    meta = pd.read_csv(chexpert_csv).set_index("Path")
    groups = []
    for p in paths:
        if p not in meta.index:
            groups.append(None)
            continue
        row = meta.loc[p]
        sex, age = row.get("Sex"), row.get("Age")
        if pd.isna(sex) or pd.isna(age):
            groups.append(None)
            continue
        groups.append({"sex": str(sex), "age_band": _age_band(float(age))})
    return groups


def nih_groups(image_ids, nih_entry_csv):
    """Row -> {"sex": ..., "age_band": ...} for NIH images, via Data_Entry_2017.csv."""
    meta = pd.read_csv(nih_entry_csv).set_index("Image Index")
    groups = []
    for iid in image_ids:
        if iid not in meta.index:
            groups.append(None)
            continue
        row = meta.loc[iid]
        sex, age = row.get("Patient Gender"), row.get("Patient Age")
        if pd.isna(sex) or pd.isna(age):
            groups.append(None)
            continue
        sex = "Female" if str(sex).upper().startswith("F") else "Male"
        groups.append({"sex": sex, "age_band": _age_band(float(age))})
    return groups


MIN_POS = 30


def subgroup_auc(y, p, groups, min_pos=MIN_POS):
    """AUC per distinct group value (e.g. per sex, or per age band).

    groups is a list/array of hashable group keys, same length as y. Returns
    {group: {"auc", "n", "n_pos", "underpowered"}}, skipping groups with a
    single class. Groups below min_pos are flagged rather than dropped so the
    narrowing is visible in the output (invariant 7) instead of silent.
    """
    from sklearn.metrics import roc_auc_score

    y, p = np.asarray(y), np.asarray(p)
    groups = np.asarray(groups, dtype=object)
    out = {}
    for g in sorted({v for v in groups if v is not None}, key=str):
        mask = groups == g
        yg = y[mask]
        if len(np.unique(yg)) < 2:
            continue
        n_pos = int(yg.sum())
        out[g] = {
            "auc": float(roc_auc_score(yg, p[mask])),
            "n": int(mask.sum()),
            "n_pos": n_pos,
            "underpowered": n_pos < min_pos,
        }
    return out


def equalised_odds_gap(y, p, groups, threshold):
    """Max gap in TPR and in FPR across groups, at a fixed operating threshold.

    Equalised odds requires both TPR and FPR to match across groups; the gap
    (max - min) in each is the standard scalar summary of how far that is from
    holding. Returns {"tpr_gap", "fpr_gap", "per_group": {...}}.
    """
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    groups = np.asarray(groups, dtype=object)

    per_group = {}
    for g in sorted({v for v in groups if v is not None}, key=str):
        mask = groups == g
        yg, predg = y[mask], pred[mask]
        pos, neg = yg == 1, yg == 0
        tpr = float(predg[pos].mean()) if pos.sum() else float("nan")
        fpr = float(predg[neg].mean()) if neg.sum() else float("nan")
        per_group[g] = {"tpr": tpr, "fpr": fpr, "n": int(mask.sum())}

    tprs = [v["tpr"] for v in per_group.values() if not np.isnan(v["tpr"])]
    fprs = [v["fpr"] for v in per_group.values() if not np.isnan(v["fpr"])]
    return {
        "tpr_gap": float(max(tprs) - min(tprs)) if len(tprs) > 1 else float("nan"),
        "fpr_gap": float(max(fprs) - min(fprs)) if len(fprs) > 1 else float("nan"),
        "per_group": per_group,
    }
