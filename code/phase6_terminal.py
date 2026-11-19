"""Phase 6 - terminal evaluation (ONE-WAY DOOR).

Opens expert.npz for the first and only time (invariant 1) and runs every
number the paper will quote: NIH retention, coverage under shift, fairness.
Refuses to run without --confirm-terminal, and prints a banner recording that
the terminal sets were opened, with a timestamp - once this executes, no
further tuning of anything upstream (Phases 3-5) is legitimate.

FROZEN CONFIGURATION
    Per-label policy and C are read back from a sweep result JSON, exactly as
    phase5_conformal.py does, rather than re-decided here. Terminal evaluation
    freezes a decision; it does not make one.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import heads
import trunks
from calibration import fit_threshold
from coverage_shift import coverage_under_shift
from external import evaluate_external, load_nih_cached, retention
from fairness import (
    DECLARED_NARROWING,
    chexpert_groups,
    equalised_odds_gap,
    nih_groups,
    subgroup_auc,
)
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
)
from metrics import per_class_metrics

BASE = str(Path(__file__).resolve().parent.parent)
PHD = str(Path(BASE).parent)
heads.BASE = BASE  # heads.py hardcodes a stale machine path; repoint it here
OUT = f"{BASE}/results"
DEFAULT_EXPERT_CSV = f"{PHD}/Dataset/CheXpert-v1.0-small/valid.csv"
DEFAULT_NIH_CSV = f"{PHD}/Dataset/NIH/Data_Entry_2017.csv"


def banner(trunk):
    ts = datetime.now().isoformat(timespec="seconds")
    print("\n" + "#" * 78)
    print(f"# TERMINAL EVALUATION - trunk={trunk} - opened at {ts}")
    print("# expert.npz and the NIH external set are being used for the FIRST and")
    print("# ONLY time in this study. No further tuning after this point is valid.")
    print("#" * 78 + "\n")
    return ts


def _load_frozen_config(sweep_tag):
    with open(f"{OUT}/{sweep_tag}.json") as f:
        sweep = json.load(f)
    policy_by_label = {
        lab: r.get("_best", WORK1_POLICY.get(lab, "u_ignore"))
        for lab, r in sweep.get("results", {}).items()
    }
    return sweep, policy_by_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default="densenet121", choices=list(trunks.TRUNKS))
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--sweep-tag", default=None)
    ap.add_argument(
        "--chexpert-csv",
        default=None,
        help="metadata for the internal eval split; default splits/train_c.csv",
    )
    ap.add_argument(
        "--expert-csv",
        default=None,
        help=f"metadata for expert.npz; default {DEFAULT_EXPERT_CSV}",
    )
    ap.add_argument(
        "--nih-entry-csv",
        default=None,
        help=f"Data_Entry_2017.csv path; default {DEFAULT_NIH_CSV}",
    )
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--confirm-terminal",
        action="store_true",
        help="required; refuses to run otherwise",
    )
    args = ap.parse_args()

    if not args.confirm_terminal:
        raise SystemExit(
            "REFUSING TO RUN: this is the one-way door (Phase 6). "
            "Pass --confirm-terminal only once every upstream phase is frozen."
        )

    set_trunk(args.trunk)
    reset_convergence()
    suffix = "" if args.trunk == "densenet121" else f"_{args.trunk}"
    sweep_tag = args.sweep_tag or f"sweep_train_c{suffix}_C0.1"
    # output always carries the trunk; sweep inputs keep the legacy bare-densenet names
    tag = args.tag or f"phase6_terminal_{args.trunk}"
    # must describe the split actually being scored, or every subgroup join silently returns None
    chexpert_csv = args.chexpert_csv or f"{BASE}/splits/train_c.csv"
    expert_csv = args.expert_csv or DEFAULT_EXPERT_CSV
    nih_csv = args.nih_entry_csv or DEFAULT_NIH_CSV

    opened_at = banner(args.trunk)
    t_start = time.time()

    sweep, policy_by_label = _load_frozen_config(sweep_tag)
    C = sweep.get("C", args.C)

    train = load_split("train_a")
    trainb = load_split("train_b")
    evalset = load_split("train_c")
    expert = load_split("expert")  # FIRST AND ONLY USE
    nih = load_nih_cached(trunks.emb_dir(BASE, args.trunk))
    names = train["label_names"]

    scaler = Standardizer().fit(train["emb_backbone"])
    x_tr_all = scaler.transform(train["emb_backbone"])
    x_b_all = scaler.transform(trainb["emb_backbone"])
    x_e_all = scaler.transform(evalset["emb_backbone"])
    x_expert = scaler.transform(expert["emb_backbone"])
    x_nih = scaler.transform(nih["emb_backbone"])

    results = {"per_label": {}}
    models = {}
    for lab, policy in policy_by_label.items():
        if lab not in names:
            continue
        li = names.index(lab)
        idx, y, w = apply_policy(train["labels"][:, li], policy)
        clf = fit_head(x_tr_all[idx], y, w, C=C, max_iter=args.max_iter)
        models[lab] = clf
        if clf is None:
            continue

        y_e_raw = evalset["labels"][:, li]
        e_mask = y_e_raw != -1
        internal_m = per_class_metrics(
            np.nan_to_num(y_e_raw[e_mask], nan=0.0),
            clf.predict_proba(x_e_all[e_mask])[:, 1],
        )

        y_expert = expert["labels"][:, li]
        expert_m = (
            per_class_metrics(y_expert, clf.predict_proba(x_expert)[:, 1])
            if len(np.unique(y_expert)) > 1
            else None
        )

        results["per_label"][lab] = {
            "policy": policy,
            "internal_train_c": internal_m,
            "expert": expert_m,
        }

    print("\nNIH RETENTION")
    nih_metrics = evaluate_external(models, {**nih, "emb_backbone": x_nih})
    for lab, m in nih_metrics.items():
        internal_auc = (
            results["per_label"].get(lab, {}).get("internal_train_c") or {}
        ).get("auc")
        r = retention(internal_auc, m["auc"]) if internal_auc else None
        results["per_label"].setdefault(lab, {})["nih"] = {**m, "retention": r}
        print(f"  {lab:<20} internal={internal_auc} nih={m['auc']:.4f} retention={r}")
    for c in external_caveats_note():
        print(f"  caveat: {c}")

    print("\nCOVERAGE UNDER SHIFT")
    for lab in HEAD_CLASSES:
        li = names.index(lab)
        clf = models.get(lab)
        if clf is None or lab not in nih["label_names"]:
            continue
        y_b_raw = trainb["labels"][:, li]
        b_mask = y_b_raw != -1
        p_b = clf.predict_proba(x_b_all[b_mask])[:, 1]
        y_b = np.nan_to_num(y_b_raw[b_mask], nan=0.0)

        y_e_raw = evalset["labels"][:, li]
        e_mask = y_e_raw != -1
        p_int = clf.predict_proba(x_e_all[e_mask])[:, 1]
        y_int = np.nan_to_num(y_e_raw[e_mask], nan=0.0)

        nih_li = nih["label_names"].index(lab)
        p_ext = clf.predict_proba(x_nih)[:, 1]
        y_ext = nih["labels"][:, nih_li]

        cov = coverage_under_shift(
            (p_b, y_b),
            (p_int, y_int),
            (p_ext, y_ext),
            alpha=args.alpha,
            x_cal=x_b_all[b_mask],
            x_external=x_nih,
        )
        results["per_label"][lab]["coverage_shift"] = cov
        print(
            f"  {lab:<20} internal_cov={cov['internal_coverage']:.3f} external_cov={cov['external_coverage']:.3f} "
            f"violated={cov['coverage_violated']} recovered={cov['weighted_mitigation']['recovered_external_coverage']:.3f}"
        )

    print("\nFAIRNESS")
    print(f"  {DECLARED_NARROWING}")
    groups = {
        "train_c": chexpert_groups(evalset["paths"], chexpert_csv),
        "expert": chexpert_groups(expert["paths"], expert_csv),
        "nih": nih_groups(nih["paths"], nih_csv) if os.path.exists(nih_csv) else None,
    }
    for nm, g in groups.items():
        if g is None:
            print(f"  {nm:<8} metadata CSV absent - subgroup analysis skipped")
            continue
        matched = sum(x is not None for x in g)
        print(f"  {nm:<8} metadata matched {matched}/{len(g)} rows")
        if matched == 0:
            raise SystemExit(
                f"FAIRNESS ABORT: no {nm} row joined its metadata CSV. "
                "An empty subgroup table would be indistinguishable from a real null."
            )

    for lab, clf in models.items():
        if clf is None or lab not in results["per_label"]:
            continue
        li = names.index(lab)

        y_b_raw = trainb["labels"][:, li]
        b_mask = y_b_raw != -1
        thr = fit_threshold(
            np.nan_to_num(y_b_raw[b_mask], nan=0.0),
            clf.predict_proba(x_b_all[b_mask])[:, 1],
        )
        block = {"threshold": float(thr)}

        for pname, split, xmat in (
            ("train_c", evalset, x_e_all),
            ("expert", expert, x_expert),
        ):
            y_raw = split["labels"][:, li]
            m = y_raw != -1
            block[pname] = _subgroup_block(
                np.nan_to_num(y_raw[m], nan=0.0),
                clf.predict_proba(xmat[m])[:, 1],
                [groups[pname][i] for i in np.flatnonzero(m)],
                thr,
            )

        if groups["nih"] is not None and lab in nih["label_names"]:
            block["nih"] = _subgroup_block(
                nih["labels"][:, nih["label_names"].index(lab)],
                clf.predict_proba(x_nih)[:, 1],
                groups["nih"],
                thr,
            )

        results["per_label"][lab]["fairness"] = block
        sex_auc = block["train_c"]["auc_by_sex"]
        gap = block["train_c"]["equalised_odds_by_sex"]["tpr_gap"]
        print(f"  {lab:<28} groups={len(sex_auc)} sex_tpr_gap={gap:.4f}")

    os.makedirs(OUT, exist_ok=True)
    payload = {
        "phase": "terminal",
        "trunk": args.trunk,
        "opened_at": opened_at,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "C": C,
        "sweep_tag": sweep_tag,
        "declared_narrowing": DECLARED_NARROWING,
        "results": results["per_label"],
    }
    path = f"{OUT}/{tag}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: None)

    print(f"\nWritten {path}")
    print(convergence_report())
    print(f"Total  : {time.time() - t_start:.1f}s")


def _subgroup_block(y, p, groups, threshold):
    sexes = [g["sex"] if g else None for g in groups]
    ages = [g["age_band"] if g else None for g in groups]
    return {
        "n": int(len(y)),
        "n_pos": int(np.nansum(y)),
        "auc_by_sex": subgroup_auc(y, p, sexes),
        "auc_by_age": subgroup_auc(y, p, ages),
        "equalised_odds_by_sex": equalised_odds_gap(y, p, sexes, threshold),
        "equalised_odds_by_age": equalised_odds_gap(y, p, ages, threshold),
    }


def external_caveats_note():
    from external import MAPPING_CAVEATS

    return MAPPING_CAVEATS


if __name__ == "__main__":
    main()
