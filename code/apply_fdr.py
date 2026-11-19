"""Apply BH-FDR to every reported comparison family and rewrite the CSVs in place.

Run after export_results.py. The exporters rebuild CSVs from the result JSONs,
so correction has to be re-applied whenever they are regenerated - this script
is idempotent and safe to re-run.

FAMILIES CORRECTED
    selection_cells.csv    TOST equivalence, one family per run_id
    tost_policy_cells.csv  policy contrasts, one family per (run_id, contrast_type)

Mechanism and H3 are already corrected inside their own runners and are left
alone; correcting them twice would be wrong.
"""

import argparse
import csv
import json
import os
from pathlib import Path

from multiplicity import DEFAULT_ALPHA, correct_within_families

RESULTS = str(Path(__file__).resolve().parent.parent / "results")

TARGETS = [
    {
        "file": "selection_cells.csv",
        "p_key": "tost_p",
        "family_keys": ["run_id"],
        "label": "selection TOST (H2)",
    },
    {
        "file": "tost_policy_cells.csv",
        "p_key": "p_tost",
        "family_keys": ["run_id", "contrast_type"],
        "label": "policy contrasts (H1)",
    },
]


def apply_to(path, p_key, family_keys, alpha):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    if p_key not in rows[0]:
        print(f"  skipped - no '{p_key}' column in {os.path.basename(path)}")
        return None

    summary = correct_within_families(rows, p_key, family_keys, alpha=alpha)

    cols = list(rows[0].keys())
    for extra in ("p_bh", "significant_bh"):
        if extra not in cols:
            cols.append(extra)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--results-dir", default=RESULTS)
    args = ap.parse_args()

    print(f"BH-FDR at alpha={args.alpha}\n")
    report = {"alpha": args.alpha, "families": {}}

    for t in TARGETS:
        path = os.path.join(args.results_dir, t["file"])
        if not os.path.exists(path):
            print(f"{t['file']:<24} not found, skipping")
            continue
        print(f"{t['file']}  [{t['label']}]")
        summary = apply_to(path, t["p_key"], t["family_keys"], args.alpha)
        if not summary:
            continue
        for fam, s in sorted(summary.items()):
            print(f"  {fam:<46} {s['n_rejected']:>3}/{s['n_tests']:<3} survive BH")
        report["families"][t["file"]] = summary
        print()

    out = os.path.join(args.results_dir, "multiplicity_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Written {out}")


if __name__ == "__main__":
    main()
