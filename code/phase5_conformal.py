"""Phase 5 runner - conformal layer (H5): split, Mondrian, FNR risk control.

DEVIATION FROM "LOAD CACHED PROBABILITIES" (TODO_CODE Task 8 / plan item 13)
    sweep.py never persists per-row probabilities to disk - it only writes the
    aggregated metrics (auc_mean, ap_mean, ...) to results/*.json. There is no
    probability cache to load. This runner instead re-derives probabilities by
    refitting with the EXACT frozen recipe read back from a sweep result JSON
    (same trunk, same C, same seeds, same per-label policy) on the same cached
    embeddings. No embeddings are re-extracted and the trunk itself is never
    touched, so the spirit of "no refitting of the expensive part" holds; only
    the cheap logistic head is refit, exactly as sweep.py itself does per seed.

WHY MONDRIAN AND FNR CONTROL BOTH NEED TRAIN-B CALIBRATION
    Invariant 2 (calibration and thresholds fit on Train-B only) applies here
    identically: conformal calibration is just another tunable quantity.

SET SIZE VS UNCERTAINTY FRACTION
    Same Spearman test as mechanism.py, applied to split-conformal average set
    size instead of policy spread - this links the conformal layer back to the
    study's central mechanism (H1).
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
from conformal import binary_nonconformity, mondrian_conformal, split_conformal
from heads import (
    Standardizer,
    WORK1_POLICY,
    apply_policy,
    convergence_report,
    fit_head,
    load_split,
    reset_convergence,
    set_trunk,
    subsample_patients,
)
from risk_control import conformal_risk_control_fnr

BASE = str(Path(__file__).resolve().parent.parent)
heads.BASE = BASE  # heads.py hardcodes a stale machine path; repoint it here
OUT = f"{BASE}/results"


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def _load_frozen_config(sweep_tag):
    """Read a sweep result JSON back for its trunk/C/seeds/per-label best policy."""
    with open(f"{OUT}/{sweep_tag}.json") as f:
        sweep = json.load(f)
    policy_by_label = {
        lab: r.get("_best", WORK1_POLICY.get(lab, "u_ignore"))
        for lab, r in sweep.get("results", {}).items()
    }
    unc_fraction = {
        lab: st.get("unc_fraction", 0.0)
        for lab, st in sweep.get("label_stats", {}).items()
    }
    return sweep, policy_by_label, unc_fraction


def _refit_probs(x_tr, y_tr_raw, rows, policy, C, max_iter, x_b, x_e):
    idx, y, w = apply_policy(y_tr_raw[rows], policy)
    clf = fit_head(x_tr[rows][idx], y, w, C=C, max_iter=max_iter)
    if clf is None:
        return None, None
    return clf.predict_proba(x_b)[:, 1], clf.predict_proba(x_e)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default="densenet121", choices=list(trunks.TRUNKS))
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=0.8)
    ap.add_argument(
        "--sweep-tag", default=None, help="sweep result JSON to freeze the config from"
    )
    ap.add_argument("--min-pos", type=int, default=30)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    set_trunk(args.trunk)
    reset_convergence()
    suffix = "" if args.trunk == "densenet121" else f"_{args.trunk}"
    sweep_tag = args.sweep_tag or f"sweep_train_c{suffix}_C0.1"
    # output always carries the trunk; sweep inputs keep the legacy bare-densenet names
    tag = args.tag or f"phase5_conformal_{args.trunk}"

    print(f"PHASE 5 - CONFORMAL LAYER  [trunk: {args.trunk}, alpha: {args.alpha}]")
    t_start = time.time()

    section("LOAD")
    sweep, policy_by_label, unc_fraction = _load_frozen_config(sweep_tag)
    seeds = sweep.get("seeds") or list(range(42, 42 + args.seeds))
    C = sweep.get("C", args.C)

    train, trainb, evalset = (
        load_split("train_a"),
        load_split("train_b"),
        load_split("train_c"),
    )
    names = train["label_names"]

    scaler = Standardizer().fit(train["emb_backbone"])
    x_tr_all = scaler.transform(train["emb_backbone"])
    x_b_all = scaler.transform(trainb["emb_backbone"])
    x_e_all = scaler.transform(evalset["emb_backbone"])

    row_sets = {s: subsample_patients(train["paths"], s, args.fraction) for s in seeds}

    print(f"Frozen config from : {sweep_tag}.json  (C={C}, seeds={seeds})")
    print(f"Labels              : {list(policy_by_label)}")

    section("CONFORMAL")
    results = {}
    for lab, policy in policy_by_label.items():
        if lab not in names:
            continue
        li = names.index(lab)
        y_tr_raw = train["labels"][:, li]

        y_b_raw = trainb["labels"][:, li]
        b_mask = y_b_raw != -1
        y_b = np.nan_to_num(y_b_raw[b_mask], nan=0.0)
        x_b = x_b_all[b_mask]

        y_e_raw = evalset["labels"][:, li]
        e_mask = y_e_raw != -1
        y_e = np.nan_to_num(y_e_raw[e_mask], nan=0.0)
        x_e = x_e_all[e_mask]

        if (
            int(y_e.sum()) < args.min_pos
            or len(np.unique(y_e)) < 2
            or len(np.unique(y_b)) < 2
        ):
            print(f"{lab:<28} skipped - insufficient positives or single class")
            continue

        split_sizes, mondrian_sizes, fnrs, coverages = [], [], [], []
        for s in seeds:
            p_b, p_e = _refit_probs(
                x_tr_all, y_tr_raw, row_sets[s], policy, C, args.max_iter, x_b, x_e
            )
            if p_b is None:
                continue

            cal_scores = binary_nonconformity(p_b)[np.arange(len(y_b)), y_b.astype(int)]
            test_scores = binary_nonconformity(p_e)

            sc = split_conformal(cal_scores, test_scores, alpha=args.alpha)
            mc = mondrian_conformal(
                cal_scores, y_b.astype(int), test_scores, alpha=args.alpha
            )
            rc = conformal_risk_control_fnr(p_b, y_b, p_e, alpha=args.alpha)

            split_sizes.append(sc["avg_set_size"])
            mondrian_sizes.append(mc["avg_set_size"])
            fnrs.append(rc["empirical_fnr"])
            coverages.append(rc["coverage"])

        if not split_sizes:
            print(f"{lab:<28} skipped - no seed produced a usable fit")
            continue

        results[lab] = {
            "policy": policy,
            "unc_fraction": unc_fraction.get(lab, 0.0),
            "split_avg_set_size_mean": float(np.mean(split_sizes)),
            "mondrian_avg_set_size_mean": float(np.mean(mondrian_sizes)),
            "risk_control_fnr_mean": float(np.mean(fnrs)),
            "risk_control_coverage_mean": float(np.mean(coverages)),
            "n_seeds_used": len(split_sizes),
        }
        print(
            f"{lab:<28} split_size={results[lab]['split_avg_set_size_mean']:.3f} "
            f"fnr={results[lab]['risk_control_fnr_mean']:.3f}"
        )

    section("MECHANISM LINK")
    labels_with_results = [l for l in results if results[l]["n_seeds_used"] > 0]
    rho, p_value = float("nan"), float("nan")
    if len(labels_with_results) >= 3:
        x_mech = [results[l]["unc_fraction"] for l in labels_with_results]
        y_mech = [results[l]["split_avg_set_size_mean"] for l in labels_with_results]
        rho, p_value = stats.spearmanr(x_mech, y_mech)
        print(f"Spearman(unc_fraction, split_set_size) = {rho:.3f}  p = {p_value:.3f}")
    else:
        print("Too few labels with results for a Spearman test.")

    os.makedirs(OUT, exist_ok=True)
    payload = {
        "phase": "conformal",
        "trunk": args.trunk,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "C": C,
        "alpha": args.alpha,
        "seeds": seeds,
        "sweep_tag": sweep_tag,
        "results": results,
        "mechanism_link": {"spearman_rho": rho, "p_value": p_value},
    }
    path = f"{OUT}/{tag}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    section("DONE")
    print(f"Written {path}")
    print(convergence_report())
    print(f"Total  : {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
