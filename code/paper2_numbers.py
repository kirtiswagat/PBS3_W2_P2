"""Collect every number Paper 2 (efficiency, shift, coverage, fairness) will quote."""

import collections
import csv
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def head(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


head("H2 - EQUIVALENCE BY FRACTION (xception, l1)")
rows = list(csv.DictReader(open(R / "selection_cells.csv")))
x = [r for r in rows if r["trunk"] == "xception" and r["method"] == "l1"]
for frac in ("0.1", "0.25", "0.5"):
    sub = [r for r in x if r["fraction"] == frac]
    eq = sum(1 for r in sub if r["tost_verdict"] == "equivalent")
    ks = [int(r["n_features"]) for r in sub]
    print(f"  fraction {frac}: {eq}/{len(sub)} equivalent, k range {min(ks)}-{max(ks)}")
    for r in sub:
        if r["tost_verdict"] != "equivalent":
            print(
                f"      NOT EQUIV: {r['pathology']:<24} k={r['n_features']:<5} "
                f"auc={r['auc_mean']} full={r['baseline_full_auc_mean']} diff={r['tost_diff']}"
            )

head("H2 - METHOD COMPARISON at fraction 0.25 (xception)")
agg = collections.defaultdict(list)
for r in rows:
    if r["trunk"] == "xception" and r["fraction"] == "0.25":
        agg[r["method"]].append(float(r["auc_mean"]))
for m, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    print(f"  {m:<10} mean AUC {sum(v) / len(v):.4f}  (n={len(v)})")

head("H2 - EFFICIENCY (xception l1)")
for frac in ("0.1", "0.25", "0.5"):
    sub = [r for r in x if r["fraction"] == frac]
    lat = [float(r["predict_ms_per_1k"]) for r in sub]
    mem = [float(r["memory_mb"]) for r in sub]
    fit = [float(r["fit_seconds"]) for r in sub]
    k = [int(r["n_features"]) for r in sub]
    print(
        f"  frac {frac}: k_mean={sum(k) / len(k):.0f} fit={sum(fit) / len(fit):.3f}s "
        f"predict={sum(lat) / len(lat):.4f}ms/1k mem={sum(mem) / len(mem):.1f}MB"
    )
full = [
    r
    for r in rows
    if r["trunk"] == "xception" and r["method"] == "l1" and r["fraction"] == "0.5"
]
print("  (full-dimension reference: 2048 features)")

head("H3 - EXTERNAL RETENTION")
d = json.loads((R / "phase6_h3_external_xception.json").read_text())
print(" verdict:", json.dumps(d["verdict"]))
print(
    f"\n{'pathology':<20}{'frac':>6}{'k':>6}{'retA':>9}{'retB':>9}{'delta':>10}{'p_bh':>9}"
)
for c in sorted(d["results"], key=lambda z: (z["pathology"], z["fraction"])):
    pb = c.get("p_bh")
    print(
        f"{c['pathology']:<20}{c['fraction']:>6}{c['n_features']:>6}"
        f"{c['rule_a']['retention_mean']:>9.4f}{c['rule_b']['retention_mean']:>9.4f}"
        f"{c['retention_delta_b_minus_a']:>+10.4f}{(f'{pb:.3f}' if pb else 'na'):>9}"
    )

head("H5 - COVERAGE UNDER SHIFT")
for r in csv.DictReader(open(R / "coverage_shift_cells.csv")):
    print(
        f"  {r['trunk']:<12}{r['pathology']:<20}int={float(r['internal_coverage']):.4f} "
        f"ext={float(r['external_coverage']):.4f} violated={r['coverage_violated']} "
        f"recovered={float(r['recovered_external_coverage']):.4f}"
    )

head("H5 - CONFORMAL SET SIZE AND FNR")
for r in csv.DictReader(open(R / "conformal_cells.csv")):
    if r["trunk"] == "xception":
        print(
            f"  {r['pathology']:<28}unc={float(r['unc_fraction']):.3f} "
            f"split={float(r['split_avg_set_size_mean']):.3f} "
            f"mondrian={float(r['mondrian_avg_set_size_mean']):.3f} "
            f"fnr={float(r['risk_control_fnr_mean']):.4f} "
            f"cov={float(r['risk_control_coverage_mean']):.4f}"
        )
m = json.loads((R / "phase5_conformal_xception.json").read_text())["mechanism_link"]
print(" set-size vs uncertainty:", m)

head("TERMINAL - NIH RETENTION")
for r in csv.DictReader(open(R / "terminal_cells.csv")):
    if r["split"] == "nih":
        print(
            f"  {r['trunk']:<12}{r['pathology']:<20}nih_auc={float(r['auc']):.4f} "
            f"retention={float(r['retention']):.4f} n_pos={r['n_pos']}"
        )

head("FAIRNESS - powered subgroups, largest gaps")
fr = [
    r
    for r in csv.DictReader(open(R / "fairness_cells.csv"))
    if r["underpowered"] == "False"
]
print(
    "powered rows:", len(fr), "| by split:", collections.Counter(r["split"] for r in fr)
)
by = collections.defaultdict(dict)
for r in fr:
    by[(r["trunk"], r["pathology"], r["attribute"], r["split"])][r["group"]] = float(
        r["auc"]
    )
gaps = []
for k, v in by.items():
    if len(v) > 1:
        gaps.append((max(v.values()) - min(v.values()), k, v))
for g, k, v in sorted(gaps, reverse=True)[:12]:
    worst = min(v, key=v.get)
    best = max(v, key=v.get)
    print(
        f"  {k[0]:<12}{k[1]:<26}{k[2]:<5}{k[3]:<10}gap={g:.4f}  "
        f"{best}={v[best]:.3f} vs {worst}={v[worst]:.3f}"
    )
