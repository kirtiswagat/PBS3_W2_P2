"""Sl 17-21 / T13-T17 - Week 3 Tier 1 baseline and GATE 2.

Trains five binary-relevance heads on the frozen embeddings, replicating
Work-1's per-disease uncertainty policy, and checks the result against the
published macro AUC of 0.7892.

Two embeddings are compared here and the winner is used for every later phase:

    emb_512   task-adapted. The published head is an affine map of it, so a
              linear probe should reproduce 0.7934 almost exactly. This is the
              honest reproduction, and it validates the cache.

    emb_1024  generic pooled backbone features. Expected to score slightly
              lower, but it is far less contaminated by Work-1's fixed
              uncertainty policy and covers all 14 labels on equal footing.
              This is the substrate the Week 4-6 mechanism test should use.

Choosing between them now costs one extra run. Choosing in Week 5 would mean
repeating the entire sweep.

Usage:
    python tier1.py --benchmark            # time a single fit first
    python tier1.py                        # full run, 5 seeds
    python tier1.py --seeds 2 --fraction 0.3   # quick smoke test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from heads import (
    HEAD_CLASSES,
    WORK1_POLICY,
    Standardizer,
    apply_policy,
    fit_head,
    load_split,
    set_trunk,
    subsample_patients,
)
import trunks
from metrics import bootstrap_ci, joint_metrics, per_class_metrics, summarise

GATE2_TARGET = 0.7892
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def section(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def fmt_hms(s):
    s = int(max(s, 0))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h {m:02d}m {sec:02d}s" if h else f"{m}m {sec:02d}s"


def benchmark(train, emb_key, C):
    section("SINGLE-FIT BENCHMARK")
    names = train["label_names"]
    scaler = Standardizer().fit(train[emb_key])
    x = scaler.transform(train[emb_key])
    y_raw = train["labels"][:, names.index("Atelectasis")]
    idx, y, w = apply_policy(y_raw, "u_ones")

    t = time.time()
    fit_head(x[idx], y, w, C=C)
    dt = time.time() - t

    print(f"embedding        : {emb_key} ({x.shape[1]}-d, {len(idx):,} rows)")
    print(f"one fit          : {fmt_hms(dt)}")
    print(f"Tier 1 (50 fits) : {fmt_hms(dt * 50)}")
    print(
        f"Week 4-6 sweep   : {fmt_hms(dt * 5 * 14 * 5)}   (5 policies x 14 diseases x 5 seeds)"
    )
    print("\nIf the sweep projection is unacceptable, reduce seeds or raise the")
    print("lbfgs tolerance before Week 4 - not during it.")


def run(train, expert, emb_key, seeds, fraction, C):
    section(f"TIER 1 - {emb_key}")
    names = train["label_names"]

    scaler = Standardizer().fit(train[emb_key])
    x_train_all = scaler.transform(train[emb_key])
    x_expert = scaler.transform(expert[emb_key])

    per_seed_probs = []
    for seed in seeds:
        rows = subsample_patients(train["paths"], seed, fraction)
        probs = np.zeros((len(x_expert), len(HEAD_CLASSES)), np.float32)
        t0 = time.time()

        for c, cls in enumerate(HEAD_CLASSES):
            y_raw = train["labels"][rows, names.index(cls)]
            idx, y, w = apply_policy(y_raw, WORK1_POLICY[cls])
            clf = fit_head(x_train_all[rows][idx], y, w, C=C)
            probs[:, c] = clf.predict_proba(x_expert)[:, 1]

        per_seed_probs.append(probs)
        aucs = []
        for c, cls in enumerate(HEAD_CLASSES):
            y = np.nan_to_num(expert["labels"][:, names.index(cls)], nan=0.0)
            m = per_class_metrics(y, probs[:, c])
            if m:
                aucs.append(m["auc"])
        print(
            f"  seed {seed:<5} macro AUC {np.mean(aucs):.4f}   ({fmt_hms(time.time() - t0)}, {len(rows):,} rows)"
        )

    mean_probs = np.mean(per_seed_probs, axis=0)

    print(
        f"\n{'Pathology':<20}{'AUC':>8}{'SD':>8}{'95% CI':>18}{'AP':>8}{'ECE':>8}{'Work-1':>9}"
    )
    print("-" * 79)
    work1 = {
        "Atelectasis": 0.7495,
        "Cardiomegaly": 0.7609,
        "Consolidation": 0.7093,
        "Edema": 0.8948,
        "Pleural Effusion": 0.8314,
    }

    results, macro_per_seed = {}, []
    for c, cls in enumerate(HEAD_CLASSES):
        y = np.nan_to_num(expert["labels"][:, names.index(cls)], nan=0.0)
        seeds_m = [per_class_metrics(y, p[:, c]) for p in per_seed_probs]
        s = summarise([m for m in seeds_m if m])
        lo, hi = bootstrap_ci(y, mean_probs[:, c])
        results[cls] = {**s, "ci95": [lo, hi]}
        print(
            f"{cls:<20}{s['auc']['mean']:>8.4f}{s['auc']['sd']:>8.4f}"
            f"{f'[{lo:.3f}, {hi:.3f}]':>18}{s['ap']['mean']:>8.4f}"
            f"{s['ece']['mean']:>8.4f}{work1[cls]:>9.4f}"
        )

    for p in per_seed_probs:
        ms = [
            per_class_metrics(
                np.nan_to_num(expert["labels"][:, names.index(c)], nan=0.0), p[:, i]
            )
            for i, c in enumerate(HEAD_CLASSES)
        ]
        macro_per_seed.append(np.mean([m["auc"] for m in ms if m]))

    macro, macro_sd = (
        float(np.mean(macro_per_seed)),
        float(np.std(macro_per_seed, ddof=1) if len(seeds) > 1 else 0.0),
    )
    y_mat = np.column_stack(
        [
            np.nan_to_num(expert["labels"][:, names.index(c)], nan=0.0)
            for c in HEAD_CLASSES
        ]
    )
    jm = joint_metrics(y_mat, mean_probs)

    print("-" * 79)
    print(
        f"{'MACRO':<20}{macro:>8.4f}{macro_sd:>8.4f}{'':>18}{'':>8}{'':>8}{GATE2_TARGET:>9.4f}"
    )
    print(
        f"\nJoint: Hamming loss {jm['hamming_loss']:.4f}  exact-match {jm['exact_match']:.4f}"
    )

    return {
        "macro_auc": macro,
        "macro_sd": macro_sd,
        "per_class": results,
        "joint": jm,
        "macro_per_seed": macro_per_seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--fraction", type=float, default=0.8, help="patient subsample per seed"
    )
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--trunk", default="densenet121", choices=list(trunks.TRUNKS))
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument(
        "--tag",
        default=None,
        help="result filename stem; use for sensitivity runs so they do not overwrite tier1.json",
    )
    args = ap.parse_args()
    tag = args.tag or (
        "tier1" if args.trunk == "densenet121" else f"tier1_{args.trunk}"
    )

    set_trunk(args.trunk)
    trunk_cfg = trunks.get(args.trunk)

    print(f"TIER 1 BASELINE / GATE 2  [trunk: {args.trunk}]")
    t_start = time.time()

    section("LOAD")
    train, expert = load_split("train_a"), load_split("expert")
    print(f"Train-A : {len(train['labels']):,} rows")
    print(f"Expert  : {len(expert['labels']):,} rows")
    print(f"Policy  : " + ", ".join(f"{k}={v}" for k, v in WORK1_POLICY.items()))
    print(f"C       : {args.C} (fixed across all policies in later phases)")
    print(f"Target  : {trunk_cfg['work1_macro_auc']} (this trunk's Work-1 macro AUC)")

    if args.benchmark:
        benchmark(train, "emb_backbone", args.C)
        return

    seeds = list(range(42, 42 + args.seeds))
    out = {}
    for emb in ("emb_dense", "emb_backbone"):
        out[emb] = run(train, expert, emb, seeds, args.fraction, args.C)

    section("GATE 2")
    # Each trunk is judged against its own Work-1 published macro AUC.
    target = trunk_cfg["work1_macro_auc"]
    best = max(out, key=lambda k: out[k]["macro_auc"])
    for emb, r in out.items():
        mark = "PASS" if r["macro_auc"] >= target else "below target"
        print(
            f"{emb:<14} macro AUC {r['macro_auc']:.4f} +/- {r['macro_sd']:.4f}   {mark}"
        )

    passed = out[best]["macro_auc"] >= target
    print(f"\nBest embedding for reproduction : {best}")
    print("Substrate for the sweep         : emb_backbone (less policy-contaminated)")

    if passed:
        print(
            f"\nGATE 2: PASS - {out[best]['macro_auc']:.4f} >= {target} ({args.trunk})"
        )
        print("Proceed to the policy sweep.")
    else:
        print(f"\nGATE 2: FAIL - best {out[best]['macro_auc']:.4f} < {target}")
        print("Fallbacks, in order:")
        print("  1. raise --C or --max-iter; confirm lbfgs converged")
        print("  2. unfreeze the trunk's final dense block and re-extract")
        print("  3. try the other trunk")

    import os

    os.makedirs(OUT, exist_ok=True)
    json_path = f"{OUT}/{tag}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "phase": "tier1",
                "trunk": args.trunk,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "gate2_passed": bool(passed),
                "target": target,
                "seeds": seeds,
                "fraction": args.fraction,
                "C": args.C,
                "results": out,
            },
            f,
            indent=2,
        )
    print(f"\nWritten {json_path}")

    # Rebuild the CSVs from every result JSON so they never drift.
    from export_results import export

    export()
    print(f"Total: {fmt_hms(time.time() - t_start)}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
