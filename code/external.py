"""External (NIH) evaluation and retention (Phase 6 support).

NIH's cache is already restricted to the five classes the trunk was trained
on (extract.py's NIH_TO_CHEXPERT), with 0/1 labels only - there is no
uncertainty label on this set, so apply_policy is never needed here.

MAPPING CAVEATS (print these with every result; they belong in the paper)
    - NIH "Effusion" is not identical to CheXpert "Pleural Effusion": the two
      datasets' radiologists used different reporting conventions.
    - NIH "Infiltration" is not used here at all; nothing in NIH_TO_CHEXPERT
      claims it equals "Consolidation" one-for-one, but the two findings
      overlap in what they visually represent, which is a real confound.
    - NIH labels are NLP-mined from reports by a different, published pipeline
      (Wang et al. 2017), not read by the same radiologists as CheXpert.
"""

import numpy as np

MAPPING_CAVEATS = [
    "NIH 'Effusion' != CheXpert 'Pleural Effusion' (different reporting convention)",
    "NIH 'Infiltration' partially overlaps 'Consolidation' but is not mapped 1:1",
    "NIH labels are NLP-mined by a different pipeline (Wang et al. 2017), not radiologist-read like CheXpert",
]


def load_nih_cached(emb_dir):
    """Load the NIH cache for one trunk. Returns the same dict shape as heads.load_split."""
    d = np.load(f"{emb_dir}/nih.npz", allow_pickle=True)
    backbone = "emb_backbone" if "emb_backbone" in d.files else "emb_1024"
    dense = "emb_dense" if "emb_dense" in d.files else "emb_512"
    return {
        "emb_backbone": d[backbone],
        "emb_dense": d[dense],
        "labels": d["labels"],
        "label_names": [str(x) for x in d["label_names"]],
        "paths": [str(p) for p in d["paths"]],
    }


def evaluate_external(models, split):
    """Score each fitted classifier against its matching NIH label column.

    models : {label_name: fitted sklearn classifier expecting standardised
              emb_backbone as input}
    split  : dict from load_nih_cached (raw, unstandardised emb_backbone)

    Returns {label_name: {"auc", "ap", "n", "n_pos"}}; from metrics.py so the
    numbers are directly comparable to the internal per_class_metrics tables.
    """
    from metrics import per_class_metrics

    names = split["label_names"]
    x = split["emb_backbone"]
    out = {}
    for lab, clf in models.items():
        if lab not in names or clf is None:
            continue
        y = split["labels"][:, names.index(lab)]
        if len(np.unique(y)) < 2:
            continue
        p = clf.predict_proba(x)[:, 1]
        m = per_class_metrics(y, p)
        if m:
            out[lab] = m
    return out


def retention(internal_auc, external_auc):
    """Fraction of internal AUC preserved on the external set (Work-1's headline metric)."""
    return float(external_auc / internal_auc) if internal_auc else float("nan")
