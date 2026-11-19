"""Sl 14-16 / T10-T12 - Phase 1 embedding extraction.

Runs every image through the frozen trunk exactly once and caches the feature
vectors. After this completes, no image is ever loaded again: the 350-fit policy
sweep in Weeks 4-6 operates purely on these cached matrices, which is what makes
the December deadline reachable.

Trunk choice rationale: see the docstring of bench_extract.py.

Three design decisions worth understanding before changing anything:

  1. LABELS ARE STORED RAW. The -1 uncertain values are preserved verbatim.
     Applying U-Ones/U-Zeros here would bake one uncertainty policy into the
     cache and make the entire Work-2 contribution impossible to compute.

  2. BOTH EMBEDDINGS ARE EXTRACTED. The trunk exposes a generic 1024-d pooled
     backbone output and a task-adapted 512-d dense output. Taking both costs
     one extra output tap on the same forward pass; taking only one risks a
     full re-extraction if Gate 2 prefers the other.

  3. ALL 14 CHEXPERT LABELS ARE STORED, not just the 5 competition classes.
     The mechanism test needs 13 pathologies to obtain the 89x spread in
     uncertainty burden that H1 rests on.

Usage:
    python extract.py                  # everything
    python extract.py --only expert    # single dataset
    python extract.py --verify         # re-check existing cache
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trunks
from preprocessing import load_image, resolve_path

PHD = os.environ.get("PHD_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CHEXPERT_DIR = f"{PHD}/Dataset/CheXpert-v1.0-small"
NIH_DIR = f"{PHD}/Dataset/NIH"
SPLITS_DIR = f"{PHD}/Problem_Statement_3/splits"
BASE = f"{PHD}/Problem_Statement_3"

# Set from --trunk in main(); module-level so the extraction helpers can see them.
TRUNK = trunks.get("densenet121")
TRUNK_NAME = "densenet121"
OUT_DIR = trunks.emb_dir(BASE, TRUNK_NAME)
IMG_SIZE = TRUNK["img_size"]
PREPROCESS = TRUNK["preprocess"]

BATCH_SIZE = 128
SHARD_SIZE = 20_000
EXPECTED_EXPERT_AUC = 0.7934
EMB_KEY = "emb_backbone"
AUC_TOLERANCE = 0.01

CHEXPERT_LABELS = [
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

# The five classes the frozen head predicts, in its output order.
HEAD_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# NIH uses different finding names; only these five map onto CheXpert classes.
# "Effusion" is not strictly identical to "Pleural Effusion" and "Infiltration"
# partially overlaps Consolidation - both caveats belong in the paper.
NIH_TO_CHEXPERT = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Pleural Effusion",
}


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# Progress bookkeeping. GLOBAL_* span the whole run so the ETA reflects every
# remaining dataset, not just the one being processed.
RUN_START = time.time()
GLOBAL_DONE = 0
GLOBAL_TOTAL = 0
TIMINGS = {}


def fmt_hms(seconds):
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:d}m {s:02d}s"


def progress(name, done_ds, total_ds, processed_this_run, elapsed_this_run):
    """Single refreshing status line.

    Rate uses only images actually pushed through the model in this invocation;
    counting resumed shards would inflate it and make the ETA meaningless.
    """
    rate = processed_this_run / elapsed_this_run if elapsed_this_run > 0 else 0.0
    remaining = (GLOBAL_TOTAL - GLOBAL_DONE) / rate if rate > 0 else 0.0
    finish = datetime.now().timestamp() + remaining
    pct_ds = 100 * done_ds / total_ds if total_ds else 0
    pct_all = 100 * GLOBAL_DONE / GLOBAL_TOTAL if GLOBAL_TOTAL else 0

    sys.stdout.write(
        f"\r  {name:<8} {done_ds:>7,}/{total_ds:<7,} {pct_ds:5.1f}%"
        f" | all {pct_all:5.1f}% | {rate:5.1f} img/s"
        f" | elapsed {fmt_hms(time.time() - RUN_START)}"
        f" | left {fmt_hms(remaining)}"
        f" | ETC {datetime.fromtimestamp(finish).strftime('%H:%M:%S')}   "
    )
    sys.stdout.flush()


def build_feature_model():
    """Tap the two embedding layers and the classifier head in one graph.

    Keras 3 will not let us read `model.get_layer('densenet121').output`
    directly: for a nested Functional model that tensor belongs to the nested
    graph, not the outer one. So we re-apply the loaded layer objects - same
    weights, same order - onto a fresh Input to obtain connected tensors.

    Extracting the head's predictions alongside the embeddings is free and
    enables the faithfulness check at the end without rebuilding the head.
    """
    import tensorflow as tf

    model = tf.keras.models.load_model(TRUNK["ckpt"], compile=False)

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inp
    emb_backbone = emb_dense = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x)
        if layer.name == TRUNK["backbone_layer"]:
            emb_backbone = x
        elif layer.name == TRUNK["dense_layer"]:
            emb_dense = x

    if emb_backbone is None or emb_dense is None:
        print(
            f"FAIL - expected layers '{TRUNK['backbone_layer']}' and "
            f"'{TRUNK['dense_layer']}' not found."
        )
        print("Layers present:", [l.name for l in model.layers])
        sys.exit(1)

    if emb_backbone.shape[-1] != TRUNK["expected_backbone_dim"]:
        print(
            f"FAIL - backbone dim {emb_backbone.shape[-1]} != expected "
            f"{TRUNK['expected_backbone_dim']}. Wrong checkpoint?"
        )
        sys.exit(1)

    feat = tf.keras.Model(inp, [emb_backbone, emb_dense, x])
    print(f"Trunk    : {TRUNK_NAME} ({os.path.basename(TRUNK['ckpt'])})")
    print(f"Input    : {IMG_SIZE}x{IMG_SIZE}  preprocess: {PREPROCESS}")
    print(
        f"Outputs  : {emb_backbone.shape[-1]}-d, {emb_dense.shape[-1]}-d, {x.shape[-1]} predictions"
    )
    return feat


# ---------------------------------------------------------------- data sources


def chexpert_source(csv_path):
    """Frontal rows with raw labels. NaN means the pathology was never mentioned."""
    df = pd.read_csv(csv_path)
    df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
    labels = (
        df[CHEXPERT_LABELS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    )
    paths = [resolve_path(CHEXPERT_DIR, p) for p in df["Path"]]
    return paths, labels, CHEXPERT_LABELS, df["Path"].tolist()


def nih_source():
    """Official NIH test split, mapped onto the five shared CheXpert classes."""
    entry = pd.read_csv(f"{NIH_DIR}/Data_Entry_2017.csv")
    with open(f"{NIH_DIR}/test_list.txt") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    # Images are spread across images_001/images .. images_012/images.
    index = {}
    for i in range(1, 13):
        d = f"{NIH_DIR}/images_{i:03d}/images"
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith(".png"):
                    index[fn] = os.path.join(d, fn)

    entry = entry[entry["Image Index"].isin(test_ids)].reset_index(drop=True)
    entry = entry[entry["Image Index"].isin(index)].reset_index(drop=True)

    names = [NIH_TO_CHEXPERT[k] for k in NIH_TO_CHEXPERT]
    labels = np.zeros((len(entry), len(names)), np.float32)
    for r, findings in enumerate(entry["Finding Labels"]):
        present = set(str(findings).split("|"))
        for c, nih_name in enumerate(NIH_TO_CHEXPERT):
            labels[r, c] = 1.0 if nih_name in present else 0.0

    paths = [index[n] for n in entry["Image Index"]]
    return paths, labels, names, entry["Image Index"].tolist()


# ------------------------------------------------------------------ extraction


def load_images(paths):
    x = np.zeros((len(paths), IMG_SIZE, IMG_SIZE, 3), np.float32)
    for i, p in enumerate(paths):
        x[i] = load_image(p, IMG_SIZE, PREPROCESS)
    return x


def extract_dataset(feat_model, name, paths, labels, label_names, rel_paths):
    section(f"EXTRACT: {name}  ({len(paths):,} images)")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{name}.npz"

    if os.path.exists(out_path):
        cached = np.load(out_path, allow_pickle=True)
        if len(cached[EMB_KEY]) == len(paths):
            print("Already cached with matching row count - skipping.")
            return out_path
        print("Cached row count differs; re-extracting.")

    shard_dir = f"{OUT_DIR}/.shards_{name}"
    os.makedirs(shard_dir, exist_ok=True)

    n = len(paths)
    n_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
    t0 = time.time()
    processed = 0  # images actually run through the model in this invocation

    global GLOBAL_DONE

    for s in range(n_shards):
        shard_file = f"{shard_dir}/{s:04d}.npz"
        lo, hi = s * SHARD_SIZE, min((s + 1) * SHARD_SIZE, n)

        if os.path.exists(shard_file):
            GLOBAL_DONE += hi - lo
            print(f"  shard {s + 1}/{n_shards}: cached ({hi - lo:,} images)")
            continue

        e1024, e512, preds = [], [], []
        for b in range(lo, hi, BATCH_SIZE):
            batch = paths[b : min(b + BATCH_SIZE, hi)]
            a, c, p = feat_model.predict(
                load_images(batch), batch_size=BATCH_SIZE, verbose=0
            )
            e1024.append(a.astype(np.float32))
            e512.append(c.astype(np.float32))
            preds.append(p.astype(np.float32))

            processed += len(batch)
            GLOBAL_DONE += len(batch)
            progress(name, min(b + len(batch), hi), n, processed, time.time() - t0)

        np.savez_compressed(
            shard_file,
            emb_backbone=np.concatenate(e1024),
            emb_dense=np.concatenate(e512),
            preds=np.concatenate(preds),
        )
        print(
            f"\n  shard {s + 1}/{n_shards} written  ({fmt_hms(time.time() - t0)} into this dataset)"
        )

    parts = [np.load(f"{shard_dir}/{s:04d}.npz") for s in range(n_shards)]
    np.savez_compressed(
        out_path,
        emb_backbone=np.concatenate([p["emb_backbone"] for p in parts]),
        emb_dense=np.concatenate([p["emb_dense"] for p in parts]),
        preds=np.concatenate([p["preds"] for p in parts]),
        labels=labels,
        label_names=np.array(label_names),
        paths=np.array(rel_paths),
    )
    for p in parts:
        p.close()
    for s in range(n_shards):
        os.remove(f"{shard_dir}/{s:04d}.npz")
    os.rmdir(shard_dir)

    elapsed = time.time() - t0
    TIMINGS[name] = {"seconds": elapsed, "rows": n, "processed": processed}
    rate = processed / elapsed if processed else 0.0
    print(
        f"Written {os.path.basename(out_path)}  {n:,} rows in {fmt_hms(elapsed)}"
        + (f"  ({rate:.1f} img/s)" if processed else "  (all shards cached)")
    )
    return out_path


# ----------------------------------------------------------------- validation


def _emb(d, which="backbone"):
    """The DenseNet cache predates the trunk-agnostic key names."""
    new, old = (
        ("emb_backbone", "emb_1024")
        if which == "backbone"
        else ("emb_dense", "emb_512")
    )
    return d[new] if new in d.files else d[old]


def check_cache(name):
    d = np.load(f"{OUT_DIR}/{name}.npz", allow_pickle=True)
    a, c, lab = _emb(d, "backbone"), _emb(d, "dense"), d["labels"]
    issues = []
    if np.isnan(a).any() or np.isnan(c).any():
        issues.append("NaN in embeddings")
    if (np.abs(a).sum(1) == 0).any():
        issues.append(f"{int((np.abs(a).sum(1) == 0).sum())} all-zero 1024-d rows")
    if len(a) != len(lab):
        issues.append("embedding/label row mismatch")

    status = "OK" if not issues else "; ".join(issues)
    print(
        f"{name:<10} {len(a):>7,} rows  1024-d {a.shape[1]}  512-d {c.shape[1]}  {status}"
    )
    return not issues


def check_faithfulness():
    """Prove extraction preserved the trunk: predictions cached during the pass
    must reproduce the AUC that t0c_verify_checkpoint.py measured directly."""
    section("FAITHFULNESS CHECK")
    from sklearn.metrics import roc_auc_score

    d = np.load(f"{OUT_DIR}/expert.npz", allow_pickle=True)
    preds = d["preds"]
    names = list(d["label_names"])
    labels = d["labels"]

    aucs = []
    for i, cls in enumerate(HEAD_CLASSES):
        y = labels[:, names.index(cls)]
        y = np.nan_to_num(y, nan=0.0)
        if len(np.unique(y)) > 1:
            aucs.append(roc_auc_score(y, preds[:, i]))

    macro = float(np.mean(aucs))
    ref = TRUNK.get("expected_expert_auc")
    if ref is not None:
        tol, src = AUC_TOLERANCE, "measured directly in T0c"
    else:
        # No independent reproduction for this trunk; fall back to the Work-1
        # published figure with a looser tolerance. This is a sanity check that
        # preprocessing is right, not an exact reproduction.
        ref, tol, src = TRUNK["work1_macro_auc"], 0.05, "Work-1 published"

    delta = macro - ref
    print(f"macro AUC from cached predictions : {macro:.4f}")
    print(f"{src:<34}: {ref:.4f}")
    print(f"delta                             : {delta:+.4f}  (tolerance {tol})")

    if abs(delta) <= tol:
        print("\nPASS - extraction is faithful to the trunk.")
        return True, macro
    print("\nFAIL - extraction diverged. Do NOT proceed to Phase 2.")
    print("Check the trunk's img_size and preprocess mode in trunks.py.")
    return False, macro


def write_manifest(names, macro):
    section("MANIFEST")
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "trunk": TRUNK_NAME,
        "checkpoint": os.path.basename(TRUNK["ckpt"]),
        "img_size": IMG_SIZE,
        "preprocess": PREPROCESS,
        "batch_size": BATCH_SIZE,
        "expert_macro_auc": macro,
        "datasets": {},
    }
    for n in names:
        p = f"{OUT_DIR}/{n}.npz"
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        h = hashlib.sha256(_emb(d).tobytes()).hexdigest()
        manifest["datasets"][n] = {
            "rows": int(len(_emb(d))),
            "labels": [str(x) for x in d["label_names"]],
            "size_mb": round(os.path.getsize(p) / 1e6, 1),
            "sha256_emb_backbone": h,
        }
        print(
            f"{n:<10} {len(_emb(d)):>7,} rows  {os.path.getsize(p) / 1e6:>7.1f} MB  {h[:16]}..."
        )

    with open(f"{OUT_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWritten {OUT_DIR}/manifest.json")


def verify():
    section("VERIFY EXISTING CACHE")
    mpath = f"{OUT_DIR}/manifest.json"
    if not os.path.exists(mpath):
        print("FAIL - no manifest. Run extraction first.")
        sys.exit(1)

    manifest = json.load(open(mpath))
    ok = True
    for n, meta in manifest["datasets"].items():
        d = np.load(f"{OUT_DIR}/{n}.npz", allow_pickle=True)
        h = hashlib.sha256(_emb(d).tobytes()).hexdigest()
        ref = meta.get("sha256_emb_backbone") or meta.get("sha256_emb_1024")
        match = h == ref and len(_emb(d)) == meta["rows"]
        ok &= match
        print(f"{n:<10} {len(_emb(d)):>7,} rows  {'MATCH' if match else 'MISMATCH'}")
    print("\nVERIFIED - cache unchanged." if ok else "\nFAIL - cache has drifted.")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trunk",
        default="densenet121",
        choices=list(trunks.TRUNKS),
        help="which Work-1 checkpoint to freeze and extract from",
    )
    ap.add_argument("--only", default=None, help="train_a|train_b|train_c|expert|nih")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    global TRUNK, TRUNK_NAME, OUT_DIR, IMG_SIZE, PREPROCESS
    TRUNK_NAME = args.trunk
    TRUNK = trunks.get(TRUNK_NAME)
    OUT_DIR = trunks.emb_dir(BASE, TRUNK_NAME)
    IMG_SIZE = TRUNK["img_size"]
    PREPROCESS = TRUNK["preprocess"]

    print(f"PHASE 1 - EMBEDDING EXTRACTION  [trunk: {TRUNK_NAME}]")
    if args.verify:
        verify()

    sources = {
        "train_a": lambda: chexpert_source(f"{SPLITS_DIR}/train_a.csv"),
        "train_b": lambda: chexpert_source(f"{SPLITS_DIR}/train_b.csv"),
        "train_c": lambda: chexpert_source(f"{SPLITS_DIR}/train_c.csv"),
        "expert": lambda: chexpert_source(f"{CHEXPERT_DIR}/valid.csv"),
        "nih": nih_source,
    }
    selected = [args.only] if args.only else list(sources)

    section("SETUP")
    feat_model = build_feature_model()

    # Resolve every source first so the run-wide ETA is correct from batch one.
    global GLOBAL_TOTAL
    resolved = {}
    for name in selected:
        resolved[name] = sources[name]()
        GLOBAL_TOTAL += len(resolved[name][0])
    print(f"Datasets : {', '.join(selected)}")
    print(f"Images   : {GLOBAL_TOTAL:,} total")
    print(f"Started  : {datetime.now().strftime('%H:%M:%S')}")

    for name in selected:
        paths, labels, label_names, rel = resolved[name]
        extract_dataset(feat_model, name, paths, labels, label_names, rel)

    section("CACHE INTEGRITY")
    all_ok = all(
        check_cache(n) for n in selected if os.path.exists(f"{OUT_DIR}/{n}.npz")
    )

    macro = None
    if os.path.exists(f"{OUT_DIR}/expert.npz"):
        faithful, macro = check_faithfulness()
        all_ok &= faithful

    write_manifest(selected, macro)

    section("TIMING SUMMARY")
    print(f"{'Dataset':<12}{'Rows':>10}{'Extracted':>12}{'Time':>14}{'Rate':>12}")
    print("-" * 74)
    for name, t in TIMINGS.items():
        rate = (
            f"{t['processed'] / t['seconds']:.1f} img/s" if t["processed"] else "cached"
        )
        print(
            f"{name:<12}{t['rows']:>10,}{t['processed']:>12,}{fmt_hms(t['seconds']):>14}{rate:>12}"
        )

    total_elapsed = time.time() - RUN_START
    total_processed = sum(t["processed"] for t in TIMINGS.values())
    print("-" * 74)
    print(
        f"{'TOTAL':<12}{GLOBAL_TOTAL:>10,}{total_processed:>12,}{fmt_hms(total_elapsed):>14}"
        f"{(total_processed / total_elapsed if total_elapsed else 0):>7.1f} img/s"
    )
    print(
        f"\nStarted  : {datetime.fromtimestamp(RUN_START).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total    : {fmt_hms(total_elapsed)}")

    section("DONE" if all_ok else "DONE WITH FAILURES")
    if all_ok:
        print("Phase 1 complete. Everything downstream is CPU-cheap.")
        print("Next: Week 3 Tier-1 baseline, Gate 2 target macro AUC >= 0.7892.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
