"""Sl 22-24 / T18-T21 - Weeks 4-6: the uncertainty-policy sweep.

This is the core experiment. Five uncertainty policies are applied to each of
the 14 CheXpert labels independently, on the frozen emb_1024 features.

WHY 25 (OR 70) FITS AND NOT 3,125
    Under plain-concatenation Binary Relevance the heads are independent by
    construction: disease i's policy cannot affect disease j's per-class AUC.
    The 5^5 combination grid therefore factorises completely. Only
    policies x diseases x seeds need to be fitted; all 3,125 global
    combinations can be assembled analytically afterwards for the joint
    metrics, which are the only place the combination matters.

WHY C IS FIXED
    Tuning regularisation per policy would confound the comparison. Holding C
    fixed means any AUC difference between policies is attributable to the
    policy alone - the structural answer to Work-1's tuning-asymmetry
    objection. Tier 1 showed macro AUC moves by 0.0003 across C = 0.1 to 10,
    so nothing is lost by fixing it.

WHY emb_1024 AND NOT emb_512
    emb_512 is the output of a layer trained on 5 labels under one fixed
    uncertainty policy, so using it here would partially confound H1.
    emb_1024 is the generic pooled backbone output and treats all 14 labels
    on equal footing.

Calibration and thresholds are fitted on Train-B only. The expert set is
scored but never tuned on.

Usage:
    python sweep.py                    # full run, 5 seeds, 14 labels
    python sweep.py --seeds 2 --labels Atelectasis,Pneumonia   # smoke test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from calibration import apply_temperature, fit_temperature, fit_threshold
from heads import (
    POLICIES,
    Standardizer,
    apply_policy,
    convergence_report,
    fit_head,
    load_split,
    reset_convergence,
    set_trunk,
    subsample_patients,
)
from metrics import expected_calibration_error, per_class_metrics
import trunks

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
EMB = "emb_backbone"

# The trunk was fine-tuned on these five labels under a fixed policy, so they
# are potentially contaminated. The other nine were never in its objective and
# serve as the control group in mechanism.py.
TRUNK_SEEN = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def fmt_hms(s):
    s = int(max(s, 0))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


def label_stats(labels, idx):
    c = labels[:, idx]
    pos = int((c == 1).sum())
    neg = int((c == 0).sum())
    unc = int((c == -1).sum())
    nan = int(np.isnan(c).sum())
    n = len(c)
    return {
        "n": n,
        "pos": pos,
        "neg_explicit": neg,
        "unmentioned": nan,
        "uncertain": unc,
        # Fraction of training rows whose label actually changes when the
        # policy flips. This is the mechanism's predictor: if almost no rows
        # change, no policy can matter much.
        "unc_fraction": unc / n,
        "unc_over_mentioned": unc / (pos + neg + unc) if (pos + neg + unc) else 0.0,
        "unc_over_effective_neg": unc / (neg + nan) if (neg + nan) else 0.0,
    }


def run_cell(x_tr, y_tr_raw, rows, x_b, y_b, x_e, y_e, policy, C, max_iter=2000):
    """One (disease, policy, seed) cell: fit, calibrate on Train-B, score expert."""
    idx, y, w = apply_policy(y_tr_raw[rows], policy)
    clf = fit_head(x_tr[rows][idx], y, w, C=C, max_iter=max_iter)
    if clf is None:
        return None

    p_b = clf.predict_proba(x_b)[:, 1]
    p_e = clf.predict_proba(x_e)[:, 1]

    t = fit_temperature(y_b, p_b)
    thr = fit_threshold(y_b, apply_temperature(p_b, t))
    p_e_cal = apply_temperature(p_e, t)

    m = per_class_metrics(y_e, p_e, threshold=thr)
    if m is None:
        return None
    m["ece_calibrated"] = expected_calibration_error(y_e, p_e_cal)
    m["temperature"] = t
    m["threshold"] = thr
    m["n_train_rows"] = int(len(idx))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=0.8)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--labels", default=None, help="comma-separated subset")
    ap.add_argument(
        "--eval-on",
        default="train_c",
        choices=["train_c", "expert"],
        help="evaluation split. train_c (18,692 rows) has the power to "
        "measure all 13 uncertainty-bearing labels; the 202-row expert "
        "set has <=1 positive for three of them and stays terminal.",
    )
    ap.add_argument(
        "--min-pos",
        type=int,
        default=30,
        help="skip labels with fewer positives than this in the eval split",
    )
    ap.add_argument("--trunk", default="densenet121", choices=list(trunks.TRUNKS))
    ap.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="lbfgs iteration cap; 2048-d features need more than 1024-d",
    )
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    set_trunk(args.trunk)
    reset_convergence()
    suffix = "" if args.trunk == "densenet121" else f"_{args.trunk}"
    tag = args.tag or f"sweep_{args.eval_on}{suffix}"

    print(f"UNCERTAINTY POLICY SWEEP  [trunk: {args.trunk}]")
    t_start = time.time()

    section("LOAD")
    train, trainb = load_split("train_a"), load_split("train_b")
    evalset = load_split(args.eval_on)
    names = train["label_names"]
    labels = args.labels.split(",") if args.labels else names
    labels = [l for l in labels if l in names]
    seeds = list(range(42, 42 + args.seeds))

    scaler = Standardizer().fit(train[EMB])
    x_tr = scaler.transform(train[EMB])
    x_b = scaler.transform(trainb[EMB])
    x_e_all = scaler.transform(evalset[EMB])

    n_cells = len(labels) * len(POLICIES) * len(seeds)
    print(f"Embedding : {EMB} ({x_tr.shape[1]}-d)")
    print(f"Eval on   : {args.eval_on} ({len(x_e_all):,} rows)")
    print(f"Labels    : {len(labels)}  (min {args.min_pos} positives required)")
    print(f"Policies  : {', '.join(POLICIES)}")
    print(f"Seeds     : {seeds}")
    print(f"C         : {args.C} (fixed across every policy)")
    print(f"Fits      : {n_cells}")

    row_sets = {s: subsample_patients(train["paths"], s, args.fraction) for s in seeds}

    section("SWEEP")
    results, stats, done = {}, {}, 0
    for lab in labels:
        li = names.index(lab)
        stats[lab] = label_stats(train["labels"], li)
        y_tr_raw = train["labels"][:, li]
        # Calibration and thresholds are fitted against certain labels only.
        # Applying the policy under test to Train-B as well would let it
        # influence its own calibration and make the comparison circular.
        y_b_raw = trainb["labels"][:, li]
        b_mask = y_b_raw != -1
        y_b = np.nan_to_num(y_b_raw[b_mask], nan=0.0)
        x_b_eval = x_b[b_mask]

        # Train-C carries uncertain labels of its own; score only rows whose
        # label is trustworthy. The expert set has none, so this is a no-op there.
        y_e_raw = evalset["labels"][:, li]
        e_mask = y_e_raw != -1
        y_e = np.nan_to_num(y_e_raw[e_mask], nan=0.0)
        x_e = x_e_all[e_mask]
        n_pos_eval = int(y_e.sum())

        if len(np.unique(y_e)) < 2:
            print(f"{lab:<28} skipped - single class in {args.eval_on}")
            done += len(POLICIES) * len(seeds)
            continue

        # Underpowered labels produce spreads that are pure noise: on the expert
        # set Pleural Other had 1 positive and reported the largest spread in
        # the table.
        if n_pos_eval < args.min_pos:
            print(
                f"{lab:<28} skipped - only {n_pos_eval} positives in {args.eval_on} "
                f"(min {args.min_pos})"
            )
            done += len(POLICIES) * len(seeds)
            continue

        if len(np.unique(y_b)) < 2:
            print(
                f"{lab:<28} skipped - single class in Train-B after masking uncertain rows"
            )
            done += len(POLICIES) * len(seeds)
            continue

        results[lab] = {}
        stats[lab]["eval_n"] = int(len(y_e))
        stats[lab]["eval_n_pos"] = n_pos_eval
        line = f"{lab:<28}"
        for pol in POLICIES:
            cells = [
                m
                for s in seeds
                if (
                    m := run_cell(
                        x_tr,
                        y_tr_raw,
                        row_sets[s],
                        x_b_eval,
                        y_b,
                        x_e,
                        y_e,
                        pol,
                        args.C,
                        args.max_iter,
                    )
                )
            ]
            done += len(seeds)
            if not cells:
                results[lab][pol] = None
                line += f"{'n/a':>11}"
                continue
            aucs = [c["auc"] for c in cells]
            results[lab][pol] = {
                "auc_mean": float(np.mean(aucs)),
                "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "auc_per_seed": [float(a) for a in aucs],
                "ap_mean": float(np.mean([c["ap"] for c in cells])),
                "ece_mean": float(np.mean([c["ece"] for c in cells])),
                "ece_calibrated_mean": float(
                    np.mean([c["ece_calibrated"] for c in cells])
                ),
                "brier_mean": float(np.mean([c["brier"] for c in cells])),
                "recall_mean": float(np.mean([c["recall"] for c in cells])),
                "temperature_mean": float(np.mean([c["temperature"] for c in cells])),
                "threshold_mean": float(np.mean([c["threshold"] for c in cells])),
                "n_train_rows": int(np.mean([c["n_train_rows"] for c in cells])),
            }
            line += f"{np.mean(aucs):>11.4f}"

        best = max(
            (p for p in POLICIES if results[lab][p]),
            key=lambda p: results[lab][p]["auc_mean"],
        )
        worst = min(
            (p for p in POLICIES if results[lab][p]),
            key=lambda p: results[lab][p]["auc_mean"],
        )
        spread = results[lab][best]["auc_mean"] - results[lab][worst]["auc_mean"]
        results[lab]["_best"] = best
        results[lab]["_spread"] = float(spread)
        print(line + f"   best={best:<9} spread={spread:.4f}   [{done}/{n_cells}]")

    section("MAIN-EFFECTS TABLE")
    header = f"{'Pathology':<28}" + "".join(f"{p:>11}" for p in POLICIES)
    print(header + f"{'best':>11}{'spread':>9}{'unc frac':>10}")
    print("-" * len(header + " " * 30))
    for lab in results:
        row = f"{lab:<28}"
        for p in POLICIES:
            r = results[lab][p]
            row += f"{r['auc_mean']:>11.4f}" if r else f"{'n/a':>11}"
        row += f"{results[lab]['_best']:>11}{results[lab]['_spread']:>9.4f}"
        row += f"{stats[lab]['unc_fraction']:>10.4f}"
        print(row)

    os.makedirs(OUT, exist_ok=True)
    payload = {
        "phase": "sweep",
        "trunk": args.trunk,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "embedding": EMB,
        "eval_on": args.eval_on,
        "min_pos": args.min_pos,
        "C": args.C,
        "seeds": seeds,
        "fraction": args.fraction,
        "policies": POLICIES,
        "trunk_seen": TRUNK_SEEN,
        "label_stats": stats,
        "results": results,
    }
    path = f"{OUT}/{tag}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    section("DONE")
    print(f"Written {path}")
    print(convergence_report())
    print(f"Total  : {fmt_hms(time.time() - t_start)}")
    print("\nNext: python mechanism.py  - does the uncertainty fraction predict")
    print("the policy spread, and does it hold on labels the trunk never saw?")


if __name__ == "__main__":
    main()
