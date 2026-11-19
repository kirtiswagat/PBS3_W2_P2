"""Preprocessing replicated bit-for-bit from Work-1 DenseNet121_Adam.py.

Do not "improve" anything here. Any deviation silently invalidates the frozen
trunk's embeddings and will surface later as an unexplained Gate-2 failure.
"""

import os

import numpy as np
from PIL import Image

IMG_SIZE = 224

# PyTorch-style ImageNet statistics, as used in Work-1 (not keras preprocess_input).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASSES_5 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# Work-1 already used a per-disease uncertainty policy (Irvin et al. heuristic),
# not a single global rule. Tier 1 must reproduce this to match 0.7892.
WORK1_U_ONE = ["Atelectasis", "Edema"]
WORK1_U_ZERO = ["Cardiomegaly", "Consolidation", "Pleural Effusion"]


def preprocess_array(img_array, mode="imagenet_std"):
    """Work-1 used a different scheme per backbone; see trunks.py."""
    if mode == "imagenet_std":
        img = img_array / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    elif mode == "tf_scale":
        img = img_array / 127.5 - 1.0
    else:
        raise ValueError(f"unknown preprocess mode: {mode}")
    return img.astype(np.float32)


def load_image(path, img_size=IMG_SIZE, mode="imagenet_std"):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))
    return preprocess_array(np.array(img, dtype=np.float32), mode)


def resolve_path(root_dir, rel_path):
    if rel_path.startswith("CheXpert-v1.0-small/"):
        rel_path = rel_path.replace("CheXpert-v1.0-small/", "", 1)
    return os.path.join(root_dir, rel_path)


def apply_work1_label_policy(df, classes=CLASSES_5):
    import pandas as pd

    df = df.copy()
    for c in classes:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        if c in WORK1_U_ONE:
            df[c] = df[c].replace(-1, 1)
        elif c in WORK1_U_ZERO:
            df[c] = df[c].replace(-1, 0)
        df[c] = df[c].astype(float)
    return df


def load_batch(df, root_dir, classes=CLASSES_5, img_size=IMG_SIZE, mode="imagenet_std"):
    n = len(df)
    x = np.zeros((n, img_size, img_size, 3), dtype=np.float32)
    y = np.zeros((n, len(classes)), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        x[i] = load_image(resolve_path(root_dir, row["Path"]), img_size, mode)
        y[i] = row[classes].values.astype(np.float32)
    return x, y
