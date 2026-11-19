"""Sl 5 / T2 - freeze Train-A / Train-B / Train-C.

The existing train_split.csv / val_split.csv pair is already a clean
patient-level 80/20 split (verified: zero patient and study overlap, union
reconstructs train.csv). What it lacks is a third split and a provenance
record, so this script:

  Train-A = train_split.csv                      (fit the heads)
  Train-B = val_split.csv, patient half 1        (search, calibrate, threshold)
  Train-C = val_split.csv, patient half 2        (final retrain, used once)

Splitting is by PATIENT. CheXpert has 64,534 patients across 191,027 frontal
images - up to 91 images for a single patient - so a row-level split would
place the same patient on both sides and silently inflate every result.

Usage:
    python splits.py            # build and freeze
    python splits.py --verify   # re-check an existing splits.json
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

PHD = os.environ.get("PHD_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = f"{PHD}/Dataset/CheXpert-v1.0-small"
OUT_DIR = f"{PHD}/Problem_Statement_3/splits"

TRAIN_CSV = f"{DATA_DIR}/train.csv"
SOURCE_A = f"{DATA_DIR}/train_split.csv"
SOURCE_BC = f"{DATA_DIR}/val_split.csv"

SEED = 42
LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def add_keys(df):
    df = df.copy()
    df["patient"] = df["Path"].str.extract(r"(patient\d+)")
    df["study"] = df["Path"].str.extract(r"(patient\d+/study\d+)")
    return df


def frontal(df):
    return df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)


def hash_paths(df):
    joined = "\n".join(sorted(df["Path"].tolist()))
    return hashlib.sha256(joined.encode()).hexdigest()


def build():
    section("1. LOAD SOURCES")
    for p in (TRAIN_CSV, SOURCE_A, SOURCE_BC):
        if not os.path.exists(p):
            print(f"FAIL - missing {p}")
            sys.exit(1)

    full = frontal(add_keys(pd.read_csv(TRAIN_CSV)))
    a = frontal(add_keys(pd.read_csv(SOURCE_A)))
    bc = frontal(add_keys(pd.read_csv(SOURCE_BC)))
    print(
        f"train.csv        frontal rows : {len(full):,}  patients: {full['patient'].nunique():,}"
    )
    print(
        f"train_split.csv  frontal rows : {len(a):,}  patients: {a['patient'].nunique():,}"
    )
    print(
        f"val_split.csv    frontal rows : {len(bc):,}  patients: {bc['patient'].nunique():,}"
    )

    section("2. SPLIT val_split BY PATIENT INTO B AND C")
    patients = np.array(sorted(bc["patient"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(patients)
    cut = len(patients) // 2
    pb, pc = set(patients[:cut]), set(patients[cut:])

    b = bc[bc["patient"].isin(pb)].reset_index(drop=True)
    c = bc[bc["patient"].isin(pc)].reset_index(drop=True)
    print(f"seed = {SEED}")
    print(f"Train-B patients: {len(pb):,}  rows: {len(b):,}")
    print(f"Train-C patients: {len(pc):,}  rows: {len(c):,}")

    return full, a, b, c


def check(full, a, b, c):
    section("3. INTEGRITY ASSERTIONS")
    splits = {"Train-A": a, "Train-B": b, "Train-C": c}
    ok = True

    for k1, k2 in [
        ("Train-A", "Train-B"),
        ("Train-A", "Train-C"),
        ("Train-B", "Train-C"),
    ]:
        for key in ("patient", "study", "Path"):
            n = len(set(splits[k1][key]) & set(splits[k2][key]))
            ok &= n == 0
            print(
                f"{k1} vs {k2:<8} {key:<8} overlap : {n:>6}  {'OK' if n == 0 else 'LEAKAGE'}"
            )

    total = len(a) + len(b) + len(c)
    exact = set(a["Path"]) | set(b["Path"]) | set(c["Path"]) == set(full["Path"])
    print(f"\nrows A+B+C = {total:,}   train.csv frontal = {len(full):,}")
    print(f"union reconstructs train.csv : {exact}")
    ok &= exact and total == len(full)

    valid = add_keys(pd.read_csv(f"{DATA_DIR}/valid.csv"))
    bleed = len(set(full["patient"]) & set(valid["patient"]))
    print(
        f"train patients also in valid.csv : {bleed}  {'OK' if bleed == 0 else 'LEAKAGE'}"
    )
    ok &= bleed == 0

    if not ok:
        print("\nFAIL - integrity checks did not pass. Splits not written.")
        sys.exit(1)
    print("\nAll integrity checks passed.")


def prevalence(a, b, c):
    section("4. LABEL PREVALENCE ACROSS SPLITS")
    print(
        f"{'Label':<28}{'A pos%':>9}{'B pos%':>9}{'C pos%':>9}{'B n_pos':>9}{'C n_pos':>9}{'max gap':>9}"
    )
    print("-" * 74)
    warn = []
    for lab in LABELS:
        pa, pb, pc = [100 * (d[lab] == 1).mean() for d in (a, b, c)]
        nb, nc = (b[lab] == 1).sum(), (c[lab] == 1).sum()
        gap = max(pa, pb, pc) - min(pa, pb, pc)
        print(f"{lab:<28}{pa:8.2f}%{pb:8.2f}%{pc:8.2f}%{nb:9,}{nc:9,}{gap:8.2f}pp")
        if min(nb, nc) < 100:
            warn.append(f"{lab}: only {min(nb, nc)} positives in the smaller of B/C")
        if gap > 2.0:
            warn.append(f"{lab}: prevalence gap {gap:.2f}pp across splits")

    if warn:
        print("\nWARNINGS")
        for w in warn:
            print(f"  - {w}")
        print("  If these matter for your analysis, switch to iterative")
        print("  stratification at patient level rather than a plain shuffle.")
    else:
        print("\nNo warnings: all labels balanced and adequately populated.")
    return warn


def write(a, b, c, warn):
    section("5. FREEZE AND HASH")
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "scope": "frontal only",
        "source": {
            "train_a": os.path.basename(SOURCE_A),
            "train_bc": os.path.basename(SOURCE_BC),
            "note": "B and C are a patient-level 50/50 division of val_split.csv",
        },
        "splits": {},
    }

    cols = [c_ for c_ in a.columns if c_ not in ("patient", "study")]
    for name, df in [("train_a", a), ("train_b", b), ("train_c", c)]:
        path = f"{OUT_DIR}/{name}.csv"
        df[cols].to_csv(path, index=False)
        h = hash_paths(df)
        manifest["splits"][name] = {
            "csv": os.path.basename(path),
            "rows": int(len(df)),
            "patients": int(df["patient"].nunique()),
            "studies": int(df["study"].nunique()),
            "sha256_paths": h,
        }
        print(
            f"{name:<9} rows={len(df):>7,}  patients={df['patient'].nunique():>6,}  sha256={h[:16]}..."
        )

    combined = hashlib.sha256(
        "".join(
            manifest["splits"][k]["sha256_paths"]
            for k in ("train_a", "train_b", "train_c")
        ).encode()
    ).hexdigest()
    manifest["sha256_combined"] = combined
    manifest["warnings"] = warn

    with open(f"{OUT_DIR}/splits.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\ncombined sha256 : {combined}")
    print(f"written to      : {OUT_DIR}/")
    print("\nThese splits are now FROZEN. Do not regenerate them. Quote the")
    print("combined hash in the paper to prove Week 10 used the same splits as")
    print("Week 3.")


def verify():
    section("VERIFY EXISTING SPLITS")
    mpath = f"{OUT_DIR}/splits.json"
    if not os.path.exists(mpath):
        print(f"FAIL - {mpath} not found. Run without --verify first.")
        sys.exit(1)

    manifest = json.load(open(mpath))
    ok = True
    for name, meta in manifest["splits"].items():
        df = add_keys(pd.read_csv(f"{OUT_DIR}/{meta['csv']}"))
        h = hash_paths(df)
        match = h == meta["sha256_paths"]
        ok &= match and len(df) == meta["rows"]
        print(
            f"{name:<9} rows={len(df):>7,} (expected {meta['rows']:,})  hash {'MATCH' if match else 'MISMATCH'}"
        )

    print(f"\ncombined sha256 : {manifest['sha256_combined']}")
    print(f"created         : {manifest['created']}")
    print("\nVERIFIED - splits unchanged." if ok else "\nFAIL - splits have drifted.")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    print("Sl 5 / T2 - FREEZE TRAIN-A / TRAIN-B / TRAIN-C")
    if args.verify:
        verify()

    full, a, b, c = build()
    check(full, a, b, c)
    warn = prevalence(a, b, c)
    write(a, b, c, warn)

    section("DONE")
    print("Week 1 complete. Next: Week 2 embedding extraction.")


if __name__ == "__main__":
    main()
