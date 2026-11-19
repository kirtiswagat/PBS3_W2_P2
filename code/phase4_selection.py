"""Phase 4 runner - feature selection under H2 (equivalence) and H3 (selection rule).

WHY THIS ORDER PER (DISEASE, FRACTION, SEED)
    filters.py preselects a redundancy-aware candidate pool (mRMR is too slow
    over the full 1024/2048-d space), group_lasso.py sparsifies that pool to
    the target size via binary search on C, and only then is a head fit and
    scored - mirroring sweep.py's own fit -> calibrate-on-Train-B -> score
    structure so this run is directly comparable to it.

H2 AS EQUIVALENCE
    "p = 0.06" is not evidence the reduced subset is as good as the full
    feature set; a TOST (two one-sided tests) verdict of "equivalent within
    Δ=0.02" is. Work-1's tost_equivalence_analysis.py (referenced by
    TODO_CODE.md) is not present on this machine, so a minimal TOST -
    two one-sided Welch t-tests on the seed-level AUC difference - is
    implemented here directly. Delta and alpha match the TODO_CODE spec.

H3 - SELECTION RULE COMPARISON
    Rule A picks the matched-size candidate subset (L1 result, top-k MI, and a
    few random draws) with the best cross-validated Train-B AUC. Rule B picks
    the one with the best worst-environment (AP/PA x age-band) Train-B AUC.
    Both then get scored once, out of sample, on Train-C - the only fair way
    to see whether Rule B's internal robustness carries over externally.

CONTROLS AT MATCHED SIZE
    top-k MI, PCA (k components), and n_random draws are all fit at exactly
    the L1 subset's size k, so a spread between them and the L1 result is
    attributable to the selector, not to subset size.

PATH-AGNOSTIC BASE
    heads.py / sweep.py hardcode BASE for a different machine. Rather than
    edit those modules, this script computes its own BASE from its file
    location and repoints heads.BASE before calling heads.set_trunk().
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

import controls
import efficiency
import export_results
import filters
import group_lasso
import heads
import selection_rules
import trunks
from calibration import apply_temperature, fit_temperature, fit_threshold
from heads import (
    HEAD_CLASSES,
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
from metrics import per_class_metrics

BASE = str(Path(__file__).resolve().parent.parent)
heads.BASE = BASE  # heads.py hardcodes a stale machine path; repoint it here
OUT = f"{BASE}/results"

TOST_DELTA = 0.02
TOST_ALPHA = 0.05


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def tost_equivalence(a, b, delta=TOST_DELTA, alpha=TOST_ALPHA):
    """Two one-sided Welch t-tests on mean(a) - mean(b) against +/- delta.

    Returns {"verdict", "diff", "p_tost"}. "equivalent" means the seed-level
    AUC difference is statistically confined inside [-delta, +delta];
    "inconclusive" covers degenerate cases (fewer than 2 seeds either side).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return {"verdict": "inconclusive", "diff": None, "p_tost": None}

    diff = float(np.mean(a) - np.mean(b))
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(var_a / na + var_b / nb)
    if se == 0:
        return {
            "verdict": "equivalent" if abs(diff) < delta else "not equivalent",
            "diff": diff,
            "p_tost": 0.0 if abs(diff) < delta else 1.0,
        }
    dof = (var_a / na + var_b / nb) ** 2 / (
        (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    )
    p_lower = 1 - stats.t.cdf((diff + delta) / se, dof)  # H0: diff <= -delta
    p_upper = stats.t.cdf((diff - delta) / se, dof)  # H0: diff >= +delta
    p_tost = float(max(p_lower, p_upper))
    verdict = "equivalent" if p_tost < alpha else "not equivalent"
    return {"verdict": verdict, "diff": diff, "p_tost": p_tost}


def _mask_env_indices(env_indices, row_mask):
    """Remap environment row indices (into the unmasked split) through a boolean mask."""
    remap = np.full(len(row_mask), -1, dtype=int)
    remap[np.flatnonzero(row_mask)] = np.arange(row_mask.sum())
    out = {}
    for key, idx in env_indices.items():
        mapped = remap[idx]
        mapped = mapped[mapped >= 0]
        if len(mapped) >= 10:
            out[key] = mapped
    return out


def _score(x_tr, y, w, x_eval, y_eval, indices, C, max_iter, threshold=None):
    clf = fit_head(x_tr[:, indices], y, w, C=C, max_iter=max_iter)
    if clf is None:
        return None, None
    p = clf.predict_proba(x_eval[:, indices])[:, 1]
    m = per_class_metrics(
        y_eval, p, threshold=threshold if threshold is not None else 0.5
    )
    return (m["auc"] if m else None), clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default="densenet121", choices=list(trunks.TRUNKS))
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fractions", default="0.10,0.25,0.50")
    ap.add_argument(
        "--fraction",
        type=float,
        default=0.8,
        help="patient subsample of Train-A, as in sweep.py",
    )
    ap.add_argument(
        "--labels",
        default=None,
        help="comma-separated subset, or 'all' for every label; default is the 5 head classes",
    )
    ap.add_argument("--min-pos", type=int, default=30)
    ap.add_argument("--n-candidates", type=int, default=500)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    set_trunk(args.trunk)
    reset_convergence()
    suffix = "" if args.trunk == "densenet121" else f"_{args.trunk}"
    tag = args.tag or f"phase4_selection{suffix}"

    fractions = [float(f) for f in args.fractions.split(",")]
    seeds = list(range(42, 42 + args.seeds))

    print(f"PHASE 4 - FEATURE SELECTION  [trunk: {args.trunk}]")
    t_start = time.time()

    section("LOAD")
    train, trainb, evalset = (
        load_split("train_a"),
        load_split("train_b"),
        load_split("train_c"),
    )
    names = train["label_names"]
    if not args.labels:
        labels = list(HEAD_CLASSES)
    elif args.labels.strip().lower() == "all":
        labels = list(names)
    else:
        labels = args.labels.split(",")
    labels = [l for l in labels if l in names]

    scaler = Standardizer().fit(train["emb_backbone"])
    x_tr_all = scaler.transform(train["emb_backbone"])
    x_b_all = scaler.transform(trainb["emb_backbone"])
    x_e_all = scaler.transform(evalset["emb_backbone"])

    envs_full = selection_rules.build_environments(
        trainb["paths"], f"{BASE}/splits/train_b.csv"
    )
    row_sets = {s: subsample_patients(train["paths"], s, args.fraction) for s in seeds}

    print(f"Labels    : {labels}")
    print(f"Fractions : {fractions}")
    print(f"Seeds     : {seeds}")
    print(f"C         : {args.C} (fixed across every method)")
    print(f"Environments (Train-B): {list(envs_full)}")

    section("SELECTION")
    results = {}
    for lab in labels:
        li = names.index(lab)
        policy = WORK1_POLICY.get(lab, "u_ignore")
        y_tr_raw = train["labels"][:, li]

        y_b_raw = trainb["labels"][:, li]
        b_mask = y_b_raw != -1
        y_b = np.nan_to_num(y_b_raw[b_mask], nan=0.0)
        x_b = x_b_all[b_mask]
        envs_b = _mask_env_indices(envs_full, b_mask)

        y_e_raw = evalset["labels"][:, li]
        e_mask = y_e_raw != -1
        y_e = np.nan_to_num(y_e_raw[e_mask], nan=0.0)
        x_e = x_e_all[e_mask]
        n_pos_eval = int(y_e.sum())

        if len(np.unique(y_e)) < 2 or n_pos_eval < args.min_pos:
            print(
                f"{lab:<28} skipped - only {n_pos_eval} positives in train_c (min {args.min_pos})"
            )
            continue
        if len(np.unique(y_b)) < 2:
            print(
                f"{lab:<28} skipped - single class in Train-B after masking uncertain rows"
            )
            continue

        results[lab] = {"policy": policy, "fractions": {}}
        print(f"\n{lab}  (policy={policy})")

        # Full-feature baseline per seed - the reference H2 equivalence is judged against.
        baseline_aucs = []
        for s in seeds:
            rows = row_sets[s]
            idx, y, w = apply_policy(y_tr_raw[rows], policy)
            x_tr = x_tr_all[rows][idx]
            auc, clf = _score(
                x_tr, y, w, x_b, y_b, np.arange(x_tr.shape[1]), args.C, args.max_iter
            )
            if clf is not None:
                p_b = clf.predict_proba(x_b)[:, 1]
                t = fit_temperature(y_b, p_b)
                thr = fit_threshold(y_b, apply_temperature(p_b, t))
                p_e = clf.predict_proba(x_e)[:, 1]
                m = per_class_metrics(y_e, p_e, threshold=thr)
                if m:
                    baseline_aucs.append(m["auc"])

        for frac in fractions:
            method_aucs = {
                "l1": [],
                "top_k_mi": [],
                "pca": [],
                "random": [],
                "rule_a": [],
                "rule_b": [],
            }
            # kept so phase6_h3_external.py can re-score the same subsets on NIH without re-selecting
            chosen_idx = {"l1": {}, "rule_a": {}, "rule_b": {}}
            k_used, eff = None, None

            for s in seeds:
                rows = row_sets[s]
                idx, y, w = apply_policy(y_tr_raw[rows], policy)
                x_tr = x_tr_all[rows][idx]

                cand_pool = filters.rank_mrmr(
                    x_tr,
                    y,
                    top_k=args.n_candidates,
                    n_candidates=min(args.n_candidates * 2, x_tr.shape[1]),
                    seed=s,
                )
                sel = group_lasso.select_l1(
                    x_tr[:, cand_pool],
                    y,
                    w,
                    target_fraction=frac,
                    max_iter=args.max_iter,
                )
                subset = cand_pool[sel["indices"]]
                k = len(subset)
                if k == 0:
                    continue
                k_used = k

                def calibrated_auc(indices):
                    a, clf = _score(
                        x_tr, y, w, x_b, y_b, indices, args.C, args.max_iter
                    )
                    if clf is None:
                        return None
                    p_b = clf.predict_proba(x_b[:, indices])[:, 1]
                    t = fit_temperature(y_b, p_b)
                    thr = fit_threshold(y_b, apply_temperature(p_b, t))
                    p_e = clf.predict_proba(x_e[:, indices])[:, 1]
                    m = per_class_metrics(y_e, p_e, threshold=thr)
                    return m["auc"] if m else None

                auc_l1 = calibrated_auc(subset)
                if auc_l1 is not None:
                    method_aucs["l1"].append(auc_l1)
                chosen_idx["l1"][str(s)] = [int(i) for i in subset]
                if eff is None:
                    eff = efficiency.benchmark_subset(
                        x_tr, y, w, subset, args.C, max_iter=args.max_iter
                    )

                topk_idx = controls.top_k_mi(x_tr, y, k)
                auc_topk = calibrated_auc(topk_idx)
                if auc_topk is not None:
                    method_aucs["top_k_mi"].append(auc_topk)

                x_tr_pca, x_e_pca = controls.pca_projection(x_tr, x_e, k)
                clf_pca = fit_head(x_tr_pca, y, w, C=args.C, max_iter=args.max_iter)
                if clf_pca is not None:
                    p = clf_pca.predict_proba(x_e_pca)[:, 1]
                    m = per_class_metrics(y_e, p)
                    if m:
                        method_aucs["pca"].append(m["auc"])

                rand_aucs = [
                    a
                    for r in controls.random_subsets(
                        x_tr.shape[1], k, n_draws=args.n_random, seed=s
                    )
                    if (a := calibrated_auc(r)) is not None
                ]
                if rand_aucs:
                    method_aucs["random"].append(float(np.mean(rand_aucs)))

                # H3: Rule A vs Rule B choose among a small matched-size candidate pool,
                # evaluated for the decision on Train-B, then scored (once) on Train-C.
                candidate_sets = [subset, topk_idx] + controls.random_subsets(
                    x_tr.shape[1], k, n_draws=5, seed=1000 + s
                )
                rule_a_choice = selection_rules.rule_a_auc(
                    x_b, y_b, np.ones(len(y_b)), candidate_sets, args.C
                )
                auc_rule_a = calibrated_auc(rule_a_choice)
                if auc_rule_a is not None:
                    method_aucs["rule_a"].append(auc_rule_a)
                chosen_idx["rule_a"][str(s)] = [int(i) for i in rule_a_choice]
                if envs_b:
                    rule_b_choice = selection_rules.rule_b_worst_env(
                        x_b, y_b, np.ones(len(y_b)), candidate_sets, envs_b, args.C
                    )
                    auc_rule_b = calibrated_auc(rule_b_choice)
                    if auc_rule_b is not None:
                        method_aucs["rule_b"].append(auc_rule_b)
                    chosen_idx["rule_b"][str(s)] = [int(i) for i in rule_b_choice]

            tost = tost_equivalence(method_aucs["l1"], baseline_aucs)

            def summarise(vals):
                if not vals:
                    return None
                return {
                    "auc_mean": float(np.mean(vals)),
                    "auc_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "auc_per_seed": [float(v) for v in vals],
                }

            results[lab]["fractions"][str(frac)] = {
                "n_features": k_used,
                "methods": {m: summarise(v) for m, v in method_aucs.items()},
                "selected_indices": chosen_idx,
                "baseline_full": summarise(baseline_aucs),
                "tost_vs_full": tost,
                "efficiency": eff,
            }
            l1_mean = method_aucs["l1"] and np.mean(method_aucs["l1"])
            print(
                f"  frac={frac:<5} k={k_used:<5} "
                f"l1_auc={l1_mean if l1_mean else float('nan'):.4f} "
                f"tost={tost['verdict']}"
            )

    os.makedirs(OUT, exist_ok=True)
    payload = {
        "phase": "selection",
        "trunk": args.trunk,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "C": args.C,
        "seeds": seeds,
        "fraction": args.fraction,
        "fractions": fractions,
        "subsample_fraction": args.fraction,
        "policy_source": "WORK1_POLICY",
        "tost_delta": TOST_DELTA,
        "tost_alpha": TOST_ALPHA,
        "results": results,
    }
    path = f"{OUT}/{tag}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    section("DONE")
    print(f"Written {path}")
    print(convergence_report())
    print(f"Total  : {time.time() - t_start:.1f}s")

    export_results.export()


if __name__ == "__main__":
    main()
