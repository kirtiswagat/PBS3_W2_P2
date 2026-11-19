"""Benjamini-Hochberg FDR control, shared across every reported comparison family.

WHY A SHARED MODULE
    mechanism.py and phase6_h3_external.py each grew their own BH routine.
    Two implementations of the same correction is how a study ends up quoting
    two different adjusted p-values for the same comparison, so both should
    import this one.

WHAT COUNTS AS A FAMILY
    BH controls the false discovery rate *within a family of tests*, so the
    family has to be declared rather than assumed. This project's families are
    one per (run, comparison type): the 57 selection TOST tests of one run are
    a family, the 140 policy contrasts of one run are another. Pooling runs
    that a reader would never compare side by side would only inflate the
    correction and understate real effects.

TOST P-VALUES
    Equivalence tests are corrected the same way. The quantity controlled is
    then the false discovery rate among *equivalence* claims, which is the
    relevant error for H2: the risk of declaring subsets interchangeable when
    they are not.
"""

import numpy as np

DEFAULT_ALPHA = 0.05


def bh_fdr(pvals, alpha=DEFAULT_ALPHA):
    """Benjamini-Hochberg step-up. Returns (rejected, adjusted), input order preserved.

    NaN p-values pass through as NaN and are excluded from the ranking rather
    than silently treated as 1.0, which would weaken the correction for every
    other test in the family.
    """
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    adjusted = np.full(p.shape, np.nan)
    rejected = np.zeros(p.shape, dtype=bool)
    if not ok.any():
        return rejected, adjusted

    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind="stable")]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    rejected[order] = adjusted[order] < alpha
    return rejected, adjusted


def correct_within_families(
    rows,
    p_key,
    family_keys,
    alpha=DEFAULT_ALPHA,
    out_p="p_bh",
    out_flag="significant_bh",
):
    """Apply BH independently within each family, writing results back onto rows.

    rows        : list of dicts (CSV records)
    p_key       : which field holds the raw p-value
    family_keys : fields whose combination defines one family
    Returns {family: {"n_tests", "n_rejected"}} for reporting.
    """
    families = {}
    for i, r in enumerate(rows):
        key = tuple(r.get(k) for k in family_keys)
        families.setdefault(key, []).append(i)

    summary = {}
    for key, members in families.items():
        # one test per distinct comparison; identical p-values repeated across
        # method rows must not each consume a rank
        seen, unique_idx = {}, []
        for i in members:
            p = rows[i].get(p_key)
            sig = (
                p,
                tuple(
                    rows[i].get(k)
                    for k in ("pathology", "fraction", "policy_a", "policy_b")
                    if k in rows[i]
                ),
            )
            if sig not in seen:
                seen[sig] = len(unique_idx)
                unique_idx.append(i)

        raw = []
        for i in unique_idx:
            try:
                raw.append(float(rows[i][p_key]))
            except (TypeError, ValueError):
                raw.append(np.nan)

        rejected, adjusted = bh_fdr(raw, alpha=alpha)
        lookup = {}
        for slot, i in enumerate(unique_idx):
            lookup[i] = (adjusted[slot], rejected[slot])

        for i in members:
            p = rows[i].get(p_key)
            sig = (
                p,
                tuple(
                    rows[i].get(k)
                    for k in ("pathology", "fraction", "policy_a", "policy_b")
                    if k in rows[i]
                ),
            )
            adj, rej = lookup[unique_idx[seen[sig]]]
            rows[i][out_p] = None if np.isnan(adj) else round(float(adj), 6)
            rows[i][out_flag] = bool(rej)

        summary["|".join(str(k) for k in key)] = {
            "n_tests": len(unique_idx),
            "n_rejected": int(rejected.sum()),
        }
    return summary
