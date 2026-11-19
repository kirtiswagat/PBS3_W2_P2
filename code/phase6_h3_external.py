"""H3 terminal readout - does shift-aware selection buy external retention?

THE HYPOTHESIS, STATED PRECISELY
    H3 is not a claim about internal AUC. Phase 4 already reports Rule A vs
    Rule B on Train-C, and that comparison cannot test H3: both rules were
    *chosen* on internal data, so an internal readout only measures how well
    each rule optimised the thing it optimised. H3 claims that Rule B (best
    worst-internal-environment AUC) retains more of its CheXpert performance
    on NIH than Rule A (best Train-B AUC) does, at matched subset size.
    Retention is the endpoint; AUC alone is not.

WHY THIS IS STILL LEGITIMATE AFTER THE PHASE 6 DOOR OPENED
    Neither rule looks at NIH. Rule A scores candidates on Train-B; Rule B
    scores them on AP/PA x age-band partitions of Train-B. NIH enters exactly
    once, here, as a pre-registered readout of a decision that was already
    frozen upstream. That is a terminal evaluation, not selection on the test
    set - but it is a *second* opening of NIH and is recorded as such, with
    its own timestamp, rather than folded silently into phase6_terminal.

NO RE-SELECTION HAPPENS HERE
    The subsets are read back from the Phase 4 JSON's "selected_indices"
    block. If that key is missing the run aborts rather than re-deriving them,
    because re-deriving would let this script quietly become a second
    selection pass rather than a readout of the first.

MULTIPLICITY
    One retention comparison per (label, fraction). Benjamini-Hochberg across
    all of them, reported with the retention delta as the effect size.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

import heads
import trunks
from calibration import apply_temperature, fit_temperature, fit_threshold
from external import MAPPING_CAVEATS, load_nih_cached, retention
from heads import (
    Standardizer,
    apply_policy,
    convergence_report,
    fit_head,
    load_split,
    reset_convergence,
    set_trunk,
    subsample_patients,
)
from metrics import per_class_metrics

BASE = str(Path(__file__).resolve().parent.parent)
heads.BASE = BASE  # heads.py hardcodes a stale machine path; repoint it here
OUT = f"{BASE}/results"

RULES = ("rule_a", "rule_b")

# Smallest retention difference worth claiming. Work-1's trunks spanned 76%-99%
# retention, so a selection rule that moves it by under one point is not a
# deployment-relevant intervention however small its p-value.
MIN_RETENTION_EFFECT = 0.01


def banner(trunk, source):
    ts = datetime.now().isoformat(timespec="seconds")
    print("\n" + "#" * 78)
    print(f"# H3 TERMINAL READOUT - trunk={trunk} - opened at {ts}")
    print(f"# subsets read back from {os.path.basename(source)} (no re-selection)")
    print("# NIH is being opened a SECOND time, as a pre-registered readout only.")
    print("#" * 78 + "\n")
    return ts


def benjamini_hochberg(pvals, alpha=0.05):
    """Thin wrapper so this runner and mechanism.py share one BH implementation."""
    from multiplicity import bh_fdr

    return bh_fdr(pvals, alpha=alpha)


def _calibrated_scores(x_tr, y, w, indices, x_b, y_b, x_int, x_ext, C, max_iter):
    """Fit on the subset, calibrate and threshold on Train-B, score internally and on NIH."""
    clf = fit_head(x_tr[:, indices], y, w, C=C, max_iter=max_iter)
    if clf is None:
        return None
    p_b = clf.predict_proba(x_b[:, indices])[:, 1]
    t = fit_temperature(y_b, p_b)
    thr = fit_threshold(y_b, apply_temperature(p_b, t))
    return {
        "p_internal": clf.predict_proba(x_int[:, indices])[:, 1],
        "p_external": clf.predict_proba(x_ext[:, indices])[:, 1],
        "threshold": thr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default="densenet121", choices=list(trunks.TRUNKS))
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument(
        "--selection-tag", default=None, help="Phase 4 JSON to read subsets from"
    )
    ap.add_argument("--min-pos", type=int, default=30)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05, help="BH-FDR level")
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--confirm-terminal",
        action="store_true",
        help="required; NIH is a terminal set and this is its second opening",
    )
    args = ap.parse_args()

    if not args.confirm_terminal:
        raise SystemExit(
            "REFUSING TO RUN: this opens NIH, a terminal set. Pass --confirm-terminal "
            "only once Phase 4 has been re-run and frozen."
        )

    set_trunk(args.trunk)
    reset_convergence()
    sel_tag = args.selection_tag or f"phase4_selection_{args.trunk}_C{args.C}"
    tag = args.tag or f"phase6_h3_external_{args.trunk}"
    source = f"{OUT}/{sel_tag}.json"
    if not os.path.exists(source):
        raise SystemExit(f"Phase 4 result not found: {source}")

    with open(source) as f:
        sel = json.load(f)
    C = sel.get("C", args.C)
    seeds = sel.get("seeds", [])
    subsample_fraction = sel.get("subsample_fraction", sel.get("fraction", 0.8))

    opened_at = banner(args.trunk, source)
    t_start = time.time()

    train, trainb, evalset = (
        load_split("train_a"),
        load_split("train_b"),
        load_split("train_c"),
    )
    nih = load_nih_cached(trunks.emb_dir(BASE, args.trunk))
    names = train["label_names"]

    scaler = Standardizer().fit(train["emb_backbone"])
    x_tr_all = scaler.transform(train["emb_backbone"])
    x_b_all = scaler.transform(trainb["emb_backbone"])
    x_e_all = scaler.transform(evalset["emb_backbone"])
    x_nih = scaler.transform(nih["emb_backbone"])
    row_sets = {
        s: subsample_patients(train["paths"], s, subsample_fraction) for s in seeds
    }

    print(f"Source    : {source}")
    print(f"C         : {C}   seeds: {seeds}")
    print(f"NIH labels: {nih['label_names']}")

    cells, pvals = [], []
    for lab, lab_r in sel.get("results", {}).items():
        if lab not in names or lab not in nih["label_names"]:
            continue
        li = names.index(lab)
        nih_li = nih["label_names"].index(lab)
        policy = lab_r.get("policy", "u_ignore")

        y_tr_raw = train["labels"][:, li]

        y_b_raw = trainb["labels"][:, li]
        b_mask = y_b_raw != -1
        y_b = np.nan_to_num(y_b_raw[b_mask], nan=0.0)
        x_b = x_b_all[b_mask]

        y_e_raw = evalset["labels"][:, li]
        e_mask = y_e_raw != -1
        y_e = np.nan_to_num(y_e_raw[e_mask], nan=0.0)
        x_e = x_e_all[e_mask]

        y_ext = nih["labels"][:, nih_li]
        if (
            len(np.unique(y_b)) < 2
            or int(y_e.sum()) < args.min_pos
            or len(np.unique(y_ext)) < 2
        ):
            print(f"{lab:<28} skipped - insufficient positives")
            continue

        for frac, frac_r in lab_r.get("fractions", {}).items():
            stored = frac_r.get("selected_indices")
            if not stored:
                raise SystemExit(
                    f"{source} has no 'selected_indices' for {lab} @ {frac}. "
                    "Re-run phase4_selection.py first - this script will not re-select."
                )

            per_rule = {}
            for rule in RULES:
                ret_seeds, int_seeds, ext_seeds = [], [], []
                for s in seeds:
                    indices = stored.get(rule, {}).get(str(s))
                    if not indices:
                        continue
                    indices = np.asarray(indices, dtype=int)
                    rows = row_sets[s]
                    idx, y, w = apply_policy(y_tr_raw[rows], policy)
                    x_tr = x_tr_all[rows][idx]

                    scored = _calibrated_scores(
                        x_tr, y, w, indices, x_b, y_b, x_e, x_nih, C, args.max_iter
                    )
                    if scored is None:
                        continue
                    m_int = per_class_metrics(
                        y_e, scored["p_internal"], threshold=scored["threshold"]
                    )
                    m_ext = per_class_metrics(
                        y_ext, scored["p_external"], threshold=scored["threshold"]
                    )
                    if not m_int or not m_ext:
                        continue
                    int_seeds.append(m_int["auc"])
                    ext_seeds.append(m_ext["auc"])
                    ret_seeds.append(retention(m_int["auc"], m_ext["auc"]))

                if ret_seeds:
                    per_rule[rule] = {
                        "n_features": frac_r.get("n_features"),
                        "internal_auc_mean": float(np.mean(int_seeds)),
                        "external_auc_mean": float(np.mean(ext_seeds)),
                        "retention_mean": float(np.mean(ret_seeds)),
                        "retention_sd": float(np.std(ret_seeds, ddof=1))
                        if len(ret_seeds) > 1
                        else 0.0,
                        "retention_per_seed": [float(v) for v in ret_seeds],
                    }

            if not all(r in per_rule for r in RULES):
                continue

            a, b = (
                per_rule["rule_a"]["retention_per_seed"],
                per_rule["rule_b"]["retention_per_seed"],
            )
            delta = (
                per_rule["rule_b"]["retention_mean"]
                - per_rule["rule_a"]["retention_mean"]
            )
            if len(a) > 1 and len(b) > 1 and (np.std(a) > 0 or np.std(b) > 0):
                p = (
                    float(stats.ttest_rel(b, a).pvalue)
                    if len(a) == len(b)
                    else float(stats.ttest_ind(b, a, equal_var=False).pvalue)
                )
            else:
                p = float("nan")

            cells.append(
                {
                    "pathology": lab,
                    "fraction": frac,
                    "n_features": per_rule["rule_a"]["n_features"],
                    "rule_a": per_rule["rule_a"],
                    "rule_b": per_rule["rule_b"],
                    "retention_delta_b_minus_a": delta,
                    "p_raw": p,
                }
            )
            pvals.append(p)
            print(
                f"  {lab:<26} frac={frac:<5} k={per_rule['rule_a']['n_features']:<5} "
                f"retA={per_rule['rule_a']['retention_mean']:.4f} "
                f"retB={per_rule['rule_b']['retention_mean']:.4f} "
                f"delta={delta:+.4f}"
            )

    if not cells:
        raise SystemExit(
            "No comparable (label, fraction) cells produced - nothing to test."
        )

    rejected, adjusted = benjamini_hochberg(pvals, alpha=args.alpha)
    for cell, rej, adj in zip(cells, rejected, adjusted):
        cell["p_bh"] = None if np.isnan(adj) else float(adj)
        cell["significant_bh"] = bool(rej)

    deltas = [c["retention_delta_b_minus_a"] for c in cells]
    n_fav = int(sum(d > 0 for d in deltas))
    mean_delta = float(np.mean(deltas))
    # a single lucky cell out of fifteen is not support; require a real effect,
    # a consistent direction, and more than one surviving comparison
    supports = bool(
        mean_delta >= MIN_RETENTION_EFFECT
        and n_fav > len(cells) * 0.6
        and int(rejected.sum()) > 1
    )
    verdict = {
        "n_cells": len(cells),
        "n_favouring_rule_b": n_fav,
        "mean_retention_delta": mean_delta,
        "median_retention_delta": float(np.median(deltas)),
        "max_retention_delta": float(np.max(deltas)),
        "n_significant_bh": int(rejected.sum()),
        "min_effect_threshold": MIN_RETENTION_EFFECT,
        "supports_h3": supports,
    }
    print(
        f"\nH3: {verdict['n_favouring_rule_b']}/{verdict['n_cells']} cells favour Rule B, "
        f"mean retention delta {verdict['mean_retention_delta']:+.4f} "
        f"(median {verdict['median_retention_delta']:+.4f}), "
        f"{verdict['n_significant_bh']} significant after BH -> supports_h3={verdict['supports_h3']}"
    )
    if not supports:
        print(
            f"  Interpreted as NULL: effect is below the {MIN_RETENTION_EFFECT:.3f} "
            "retention threshold pre-set as the smallest difference worth claiming."
        )
    for c in MAPPING_CAVEATS:
        print(f"  caveat: {c}")

    os.makedirs(OUT, exist_ok=True)
    payload = {
        "phase": "h3_external",
        "trunk": args.trunk,
        "opened_at": opened_at,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "C": C,
        "seeds": seeds,
        "selection_source": os.path.basename(source),
        "bh_alpha": args.alpha,
        "mapping_caveats": MAPPING_CAVEATS,
        "verdict": verdict,
        "results": cells,
    }
    path = f"{OUT}/{tag}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWritten {path}")
    print(convergence_report())
    print(f"Total  : {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
