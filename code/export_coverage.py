"""Export terminal coverage results to a CSV for F4 and paper tables."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run_id",
    "trunk",
    "pathology",
    "alpha",
    "nominal_coverage",
    "internal_coverage",
    "external_coverage",
    "coverage_violated",
    "recovered_external_coverage",
]


def _rows_from(source):
    data = json.loads(Path(source).read_text())
    run_id = Path(source).stem
    trunk = data.get("trunk")
    rows = []
    for pathology, result in data.get("results", {}).items():
        coverage = result.get("coverage_shift")
        if coverage is None:
            continue
        mitigation = coverage.get("weighted_mitigation", {})
        rows.append(
            {
                "run_id": run_id,
                "trunk": trunk,
                "pathology": pathology,
                "alpha": coverage["alpha"],
                "nominal_coverage": coverage["nominal_coverage"],
                "internal_coverage": coverage["internal_coverage"],
                "external_coverage": coverage["external_coverage"],
                "coverage_violated": coverage["coverage_violated"],
                "recovered_external_coverage": mitigation.get(
                    "recovered_external_coverage"
                ),
            }
        )
    return rows


def export_coverage(sources, output):
    output = Path(output)
    rows = []
    for source in sources:
        rows.extend(_rows_from(source))

    if not rows:
        raise ValueError(f"No coverage_shift results found in {list(sources)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    trunks = sorted({r["trunk"] for r in rows})
    print(f"Wrote {output} with {len(rows)} rows across {trunks}")


def main():
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs="*",
        default=None,
        type=Path,
        help="terminal JSONs; defaults to every results/phase6_terminal_*.json",
    )
    parser.add_argument(
        "--output",
        default=base / "results" / "coverage_shift_cells.csv",
        type=Path,
    )
    args = parser.parse_args()
    sources = args.source or sorted((base / "results").glob("phase6_terminal_*.json"))
    export_coverage(sources, args.output)


if __name__ == "__main__":
    main()
