"""Export result JSONs to tidy CSVs for analysis and for the paper.

Three files, all append-free (rebuilt from the JSONs each time, so they can
never drift out of sync with the source of truth):

    runs.csv        one row per (run, embedding)  - headline numbers
    per_class.csv   one row per (run, embedding, pathology)
    per_seed.csv    one row per (run, embedding, seed)

Long format rather than wide, so sensitivity runs (different C, different seed
counts) stack as extra rows and can be pivoted in Excel or pandas without
reshaping anything by hand.

Usage:
    python export_results.py              # rebuild from every results/*.json
    python export_results.py --show       # rebuild and print a summary
"""

import argparse
import csv
import glob
import json
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = str(Path(__file__).resolve().parent.parent / "results")

# Published Work-1 per-class AUCs, for the delta columns.
WORK1_AUC = {
    "Atelectasis": 0.7495,
    "Cardiomegaly": 0.7609,
    "Consolidation": 0.7093,
    "Edema": 0.8948,
    "Pleural Effusion": 0.8314,
}
WORK1_MACRO = 0.7892

RUNS_COLS = [
    "run_id",
    "timestamp",
    "phase",
    "embedding",
    "C",
    "n_seeds",
    "subsample_fraction",
    "macro_auc",
    "macro_sd",
    "work1_macro",
    "delta_vs_work1",
    "hamming_loss",
    "exact_match",
    "gate2_target",
    "gate2_passed",
]

PER_CLASS_COLS = [
    "run_id",
    "embedding",
    "pathology",
    "auc_mean",
    "auc_sd",
    "ci95_lo",
    "ci95_hi",
    "ap_mean",
    "ap_sd",
    "recall_mean",
    "ece_mean",
    "brier_mean",
    "n_pos",
    "n",
    "work1_auc",
    "delta_vs_work1",
]

PER_SEED_COLS = ["run_id", "embedding", "seed", "macro_auc"]


def _get(metric_block, key, stat="mean"):
    """per_class entries look like {'auc': {'mean': x, 'sd': y}, 'ci95': [lo, hi]}."""
    v = metric_block.get(key)
    if isinstance(v, dict):
        return round(v.get(stat, float("nan")), 6)
    return v


def collect(pattern="*.json"):
    runs, per_class, per_seed = [], [], []

    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        run_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)

        if "results" not in data:
            continue  # not a run file

        # Sweep files share the "results" key but have a different shape;
        # they are handled by collect_sweep(). Runs written before the phase
        # field existed are tier1 by default.
        if data.get("phase", "tier1") != "tier1":
            continue

        # Older runs predate the embedded timestamp; fall back to file mtime.
        ts = data.get("timestamp") or datetime.fromtimestamp(
            os.path.getmtime(path)
        ).isoformat(timespec="seconds")

        seeds = data.get("seeds", [])
        for emb, r in data.get("results", {}).items():
            runs.append(
                {
                    "run_id": run_id,
                    "timestamp": ts,
                    "phase": data.get("phase", "tier1"),
                    "embedding": emb,
                    "C": data.get("C"),
                    "n_seeds": len(seeds),
                    "subsample_fraction": data.get("fraction"),
                    "macro_auc": round(r["macro_auc"], 6),
                    "macro_sd": round(r["macro_sd"], 6),
                    "work1_macro": WORK1_MACRO,
                    "delta_vs_work1": round(r["macro_auc"] - WORK1_MACRO, 6),
                    "hamming_loss": round(r["joint"]["hamming_loss"], 6),
                    "exact_match": round(r["joint"]["exact_match"], 6),
                    "gate2_target": data.get("target"),
                    "gate2_passed": data.get("gate2_passed"),
                }
            )

            for cls, m in r["per_class"].items():
                ci = m.get("ci95", [None, None])
                auc = _get(m, "auc")
                per_class.append(
                    {
                        "run_id": run_id,
                        "embedding": emb,
                        "pathology": cls,
                        "auc_mean": auc,
                        "auc_sd": _get(m, "auc", "sd"),
                        "ci95_lo": round(ci[0], 6) if ci[0] is not None else None,
                        "ci95_hi": round(ci[1], 6) if ci[1] is not None else None,
                        "ap_mean": _get(m, "ap"),
                        "ap_sd": _get(m, "ap", "sd"),
                        "recall_mean": _get(m, "recall"),
                        "ece_mean": _get(m, "ece"),
                        "brier_mean": _get(m, "brier"),
                        "n_pos": _get(m, "n_pos"),
                        "n": _get(m, "n"),
                        "work1_auc": WORK1_AUC.get(cls),
                        "delta_vs_work1": round(auc - WORK1_AUC[cls], 6)
                        if cls in WORK1_AUC
                        else None,
                    }
                )

            for seed, macro in zip(seeds, r.get("macro_per_seed", [])):
                per_seed.append(
                    {
                        "run_id": run_id,
                        "embedding": emb,
                        "seed": seed,
                        "macro_auc": round(macro, 6),
                    }
                )

    return runs, per_class, per_seed


SWEEP_COLS = [
    "run_id",
    "trunk",
    "C",
    "eval_on",
    "pathology",
    "policy",
    "is_best",
    "auc_mean",
    "auc_sd",
    "ap_mean",
    "ece_mean",
    "ece_calibrated_mean",
    "brier_mean",
    "recall_mean",
    "temperature_mean",
    "threshold_mean",
    "n_train_rows",
    "eval_n",
    "eval_n_pos",
    "unc_fraction",
    "unc_over_mentioned",
    "uncertain",
    "trunk_seen",
    "spread",
]

SWEEP_SEED_COLS = ["run_id", "eval_on", "pathology", "policy", "seed", "auc"]


def collect_sweep(pattern="sweep*.json"):
    """Long format: one row per (run, pathology, policy).

    Wide 5x14 tables are for reading; long tables are for pivoting. The
    main-effects table in the paper is a pivot of this.
    """
    cells, seed_rows = [], []

    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        run_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)
        if data.get("phase") != "sweep":
            continue

        eval_on = data.get("eval_on", "expert")
        seeds = data.get("seeds", [])
        trunk_seen = set(data.get("trunk_seen", []))

        for lab, r in data["results"].items():
            st = data["label_stats"].get(lab, {})
            for pol in data["policies"]:
                m = r.get(pol)
                if not m:
                    continue
                cells.append(
                    {
                        "run_id": run_id,
                        "trunk": data.get("trunk", "densenet121"),
                        "C": data.get("C"),
                        "eval_on": eval_on,
                        "pathology": lab,
                        "policy": pol,
                        "is_best": pol == r.get("_best"),
                        "auc_mean": round(m["auc_mean"], 6),
                        "auc_sd": round(m["auc_sd"], 6),
                        "ap_mean": round(m["ap_mean"], 6),
                        "ece_mean": round(m["ece_mean"], 6),
                        "ece_calibrated_mean": round(m["ece_calibrated_mean"], 6),
                        "brier_mean": round(m["brier_mean"], 6),
                        "recall_mean": round(m["recall_mean"], 6),
                        "temperature_mean": round(m["temperature_mean"], 4),
                        "threshold_mean": round(m["threshold_mean"], 4),
                        "n_train_rows": m["n_train_rows"],
                        "eval_n": st.get("eval_n"),
                        "eval_n_pos": st.get("eval_n_pos"),
                        "unc_fraction": round(st.get("unc_fraction", 0), 6),
                        "unc_over_mentioned": round(st.get("unc_over_mentioned", 0), 6),
                        "uncertain": st.get("uncertain"),
                        "trunk_seen": lab in trunk_seen,
                        "spread": round(r.get("_spread", 0), 6),
                    }
                )
                for seed, auc in zip(seeds, m.get("auc_per_seed", [])):
                    seed_rows.append(
                        {
                            "run_id": run_id,
                            "eval_on": eval_on,
                            "pathology": lab,
                            "policy": pol,
                            "seed": seed,
                            "auc": round(auc, 6),
                        }
                    )

    return cells, seed_rows


MECH_COLS = [
    "run_id",
    "trunk",
    "C",
    "group",
    "n_labels",
    "spearman_rho",
    "p_value",
    "supports_h1",
]


def collect_mechanism(pattern="mechanism_*.json"):
    """One row per (run, correlation group) so the replication table is a pivot."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        run_id = os.path.basename(path).replace("mechanism_", "").replace(".json", "")
        with open(path) as f:
            data = json.load(f)

        # The sweep file this analysis came from carries the trunk and C.
        src = os.path.join(RESULTS_DIR, f"{run_id}.json")
        trunk, c = "densenet121", None
        if os.path.exists(src):
            with open(src) as f:
                s = json.load(f)
            trunk, c = s.get("trunk", "densenet121"), s.get("C")

        for group, r in (data.get("correlations") or {}).items():
            if not r:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "trunk": trunk,
                    "C": c,
                    "group": group,
                    "n_labels": r["n"],
                    "spearman_rho": round(r["rho"], 4),
                    "p_value": round(r["p"], 4),
                    "supports_h1": r["rho"] > 0 and r["p"] < 0.05,
                }
            )
    return rows


SELECTION_COLS = [
    "run_id",
    "trunk",
    "C",
    "pathology",
    "policy",
    "fraction",
    "method",
    "n_features",
    "auc_mean",
    "auc_sd",
    "baseline_full_auc_mean",
    "tost_verdict",
    "tost_diff",
    "tost_p",
    "fit_seconds",
    "predict_ms_per_1k",
    "memory_mb",
]


def collect_selection(pattern="phase4_selection*.json"):
    """One row per (run, pathology, fraction, method) - the H2/H3 selection table."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        run_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)
        if data.get("phase") != "selection":
            continue

        for lab, lab_r in data.get("results", {}).items():
            policy = lab_r.get("policy")
            for frac, frac_r in lab_r.get("fractions", {}).items():
                baseline = frac_r.get("baseline_full") or {}
                tost = frac_r.get("tost_vs_full") or {}
                eff = frac_r.get("efficiency") or {}
                for method, m in frac_r.get("methods", {}).items():
                    if not m:
                        continue
                    rows.append(
                        {
                            "run_id": run_id,
                            "trunk": data.get("trunk", "densenet121"),
                            "C": data.get("C"),
                            "pathology": lab,
                            "policy": policy,
                            "fraction": frac,
                            "method": method,
                            "n_features": frac_r.get("n_features"),
                            "auc_mean": round(m["auc_mean"], 6),
                            "auc_sd": round(m["auc_sd"], 6),
                            "baseline_full_auc_mean": round(baseline["auc_mean"], 6)
                            if baseline
                            else None,
                            "tost_verdict": tost.get("verdict"),
                            "tost_diff": tost.get("diff"),
                            "tost_p": tost.get("p_tost"),
                            "fit_seconds": eff.get("fit_seconds"),
                            "predict_ms_per_1k": eff.get("predict_ms_per_1k"),
                            "memory_mb": eff.get("memory_mb"),
                        }
                    )
    return rows


CONFORMAL_COLS = [
    "run_id",
    "trunk",
    "C",
    "alpha",
    "pathology",
    "policy",
    "unc_fraction",
    "split_avg_set_size_mean",
    "mondrian_avg_set_size_mean",
    "risk_control_fnr_mean",
    "risk_control_coverage_mean",
    "n_seeds_used",
]

TERMINAL_COLS = [
    "run_id",
    "trunk",
    "C",
    "pathology",
    "policy",
    "split",
    "auc",
    "ap",
    "recall",
    "ece",
    "brier",
    "n_pos",
    "n",
    "retention",
]

FAIRNESS_COLS = [
    "run_id",
    "trunk",
    "pathology",
    "split",
    "attribute",
    "group",
    "auc",
    "n",
    "n_pos",
    "underpowered",
    "tpr",
    "fpr",
    "tpr_gap",
    "fpr_gap",
]

MIN_POS = 30  # invariant 7: below this a subgroup AUC is noise, not a measurement


def collect_conformal(pattern="phase5_conformal*.json"):
    """One row per (run, pathology) - the H5 prediction-set table."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        run_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)
        if data.get("phase") != "conformal":
            continue
        for lab, r in data.get("results", {}).items():
            rows.append(
                {
                    "run_id": run_id,
                    "trunk": data.get("trunk"),
                    "C": data.get("C"),
                    "alpha": data.get("alpha"),
                    "pathology": lab,
                    **{k: r.get(k) for k in CONFORMAL_COLS[5:]},
                }
            )
    return rows


def collect_terminal(pattern="phase6_terminal*.json"):
    """Long-format terminal metrics and subgroup fairness, one row per split/group."""
    metric_rows, fair_rows = [], []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        run_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)
        if data.get("phase") != "terminal":
            continue
        trunk, C = data.get("trunk"), data.get("C")

        for lab, r in data.get("results", {}).items():
            for split in ("internal_train_c", "expert", "nih"):
                m = r.get(split)
                if not m:
                    continue
                metric_rows.append(
                    {
                        "run_id": run_id,
                        "trunk": trunk,
                        "C": C,
                        "pathology": lab,
                        "policy": r.get("policy"),
                        "split": split,
                        **{
                            k: m.get(k)
                            for k in (
                                "auc",
                                "ap",
                                "recall",
                                "ece",
                                "brier",
                                "n_pos",
                                "n",
                            )
                        },
                        "retention": m.get("retention"),
                    }
                )

            fair = r.get("fairness") or {}
            for split, block in fair.items():
                if not isinstance(block, dict) or "auc_by_sex" not in block:
                    continue
                for attr, auc_key, eo_key in (
                    ("sex", "auc_by_sex", "equalised_odds_by_sex"),
                    ("age", "auc_by_age", "equalised_odds_by_age"),
                ):
                    eo = block.get(eo_key) or {}
                    per_group = eo.get("per_group", {})
                    for group, g in (block.get(auc_key) or {}).items():
                        fair_rows.append(
                            {
                                "run_id": run_id,
                                "trunk": trunk,
                                "pathology": lab,
                                "split": split,
                                "attribute": attr,
                                "group": group,
                                "auc": round(g["auc"], 6),
                                "n": g["n"],
                                "n_pos": g["n_pos"],
                                "underpowered": g["n_pos"] < MIN_POS,
                                "tpr": (per_group.get(group) or {}).get("tpr"),
                                "fpr": (per_group.get(group) or {}).get("fpr"),
                                "tpr_gap": eo.get("tpr_gap"),
                                "fpr_gap": eo.get("fpr_gap"),
                            }
                        )
    return metric_rows, fair_rows


H3_COLS = [
    "run_id",
    "trunk",
    "C",
    "pathology",
    "fraction",
    "n_features",
    "rule_a_internal_auc",
    "rule_a_external_auc",
    "rule_a_retention",
    "rule_b_internal_auc",
    "rule_b_external_auc",
    "rule_b_retention",
    "retention_delta_b_minus_a",
    "p_raw",
    "p_bh",
    "significant_bh",
]


def collect_h3(pattern="phase6_h3_external*.json"):
    """One row per (run, pathology, fraction) - the H3 external-retention test."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        run_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)
        if data.get("phase") != "h3_external":
            continue
        for c in data.get("results", []):
            a, b = c.get("rule_a", {}), c.get("rule_b", {})
            rows.append(
                {
                    "run_id": run_id,
                    "trunk": data.get("trunk"),
                    "C": data.get("C"),
                    "pathology": c.get("pathology"),
                    "fraction": c.get("fraction"),
                    "n_features": c.get("n_features"),
                    "rule_a_internal_auc": a.get("internal_auc_mean"),
                    "rule_a_external_auc": a.get("external_auc_mean"),
                    "rule_a_retention": a.get("retention_mean"),
                    "rule_b_internal_auc": b.get("internal_auc_mean"),
                    "rule_b_external_auc": b.get("external_auc_mean"),
                    "rule_b_retention": b.get("retention_mean"),
                    "retention_delta_b_minus_a": c.get("retention_delta_b_minus_a"),
                    "p_raw": c.get("p_raw"),
                    "p_bh": c.get("p_bh"),
                    "significant_bh": c.get("significant_bh"),
                }
            )
    return rows


JOINT_COLS = [
    "run_id",
    "trunk",
    "C",
    "label_set",
    "n_labels",
    "n_rows",
    "configuration",
    "hamming_loss",
    "hamming_sd",
    "exact_match",
    "exact_match_sd",
    "mean_per_label_accuracy",
    "br_gap",
]


def collect_joint(pattern="joint_assembly*.json"):
    """One row per (run, label set, configuration) - the T23 joint-metric table."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        run_id = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)
        if data.get("phase") != "joint":
            continue

        for set_name, block in data.get("results", {}).items():
            if set_name == "assembly" or not isinstance(block, dict):
                continue
            configs = {f"global:{p}": m for p, m in block.get("global", {}).items()}
            if block.get("per_label_best"):
                configs["per_label_best"] = block["per_label_best"]
            for cfg, m in configs.items():
                rows.append(
                    {
                        "run_id": run_id,
                        "trunk": data.get("trunk"),
                        "C": data.get("C"),
                        "label_set": set_name,
                        "n_labels": block.get("n_labels"),
                        "n_rows": block.get("n_rows"),
                        "configuration": cfg,
                        "hamming_loss": round(m["hamming_loss"]["mean"], 6),
                        "hamming_sd": round(m["hamming_loss"]["sd"], 6),
                        "exact_match": round(m["exact_match"]["mean"], 6),
                        "exact_match_sd": round(m["exact_match"]["sd"], 6),
                        "mean_per_label_accuracy": round(
                            m["mean_per_label_accuracy"]["mean"], 6
                        ),
                        "br_gap": round(
                            m["mean_per_label_accuracy"]["mean"]
                            - m["exact_match"]["mean"],
                            6,
                        ),
                    }
                )
    return rows


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  {os.path.basename(path):<20} {len(rows):>4} rows")


def export(show=False):
    runs, per_class, per_seed = collect()
    sweep_cells, sweep_seeds = collect_sweep()

    if not runs and not sweep_cells:
        print("No result JSONs found.")
        return

    print(f"Exporting to {RESULTS_DIR}/")
    if runs:
        write_csv(f"{RESULTS_DIR}/runs.csv", RUNS_COLS, runs)
        write_csv(f"{RESULTS_DIR}/per_class.csv", PER_CLASS_COLS, per_class)
        write_csv(f"{RESULTS_DIR}/per_seed.csv", PER_SEED_COLS, per_seed)
    if sweep_cells:
        write_csv(f"{RESULTS_DIR}/sweep_cells.csv", SWEEP_COLS, sweep_cells)
        write_csv(f"{RESULTS_DIR}/sweep_per_seed.csv", SWEEP_SEED_COLS, sweep_seeds)

    mech = collect_mechanism()
    if mech:
        write_csv(f"{RESULTS_DIR}/mechanism_all.csv", MECH_COLS, mech)

    selection = collect_selection()
    if selection:
        write_csv(f"{RESULTS_DIR}/selection_cells.csv", SELECTION_COLS, selection)

    conformal = collect_conformal()
    if conformal:
        write_csv(f"{RESULTS_DIR}/conformal_cells.csv", CONFORMAL_COLS, conformal)

    terminal, fairness = collect_terminal()
    if terminal:
        write_csv(f"{RESULTS_DIR}/terminal_cells.csv", TERMINAL_COLS, terminal)
    if fairness:
        write_csv(f"{RESULTS_DIR}/fairness_cells.csv", FAIRNESS_COLS, fairness)

    h3 = collect_h3()
    if h3:
        write_csv(f"{RESULTS_DIR}/h3_external_cells.csv", H3_COLS, h3)

    joint = collect_joint()
    if joint:
        write_csv(f"{RESULTS_DIR}/joint_cells.csv", JOINT_COLS, joint)

    if show:
        print(
            f"\n{'run_id':<20}{'embedding':<11}{'C':>6}{'seeds':>7}"
            f"{'macro AUC':>11}{'SD':>9}{'vs Work-1':>11}{'gate2':>8}"
        )
        print("-" * 84)
        for r in runs:
            print(
                f"{r['run_id']:<20}{r['embedding']:<11}{str(r['C']):>6}"
                f"{r['n_seeds']:>7}{r['macro_auc']:>11.4f}{r['macro_sd']:>9.4f}"
                f"{r['delta_vs_work1']:>+11.4f}"
                f"{('PASS' if r['gate2_passed'] else 'FAIL'):>8}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    export(show=ap.parse_args().show)
