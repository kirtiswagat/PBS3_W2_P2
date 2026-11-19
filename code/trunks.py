"""Trunk registry.

Each Work-1 checkpoint was trained with its own input size and its own
preprocessing, and those choices are not interchangeable. DenseNet121 used
224x224 with PyTorch-style ImageNet normalisation; Xception used 299x299 scaled
to [-1, 1]. Getting this wrong produces embeddings that look plausible and are
silently wrong, so the values below are copied verbatim from the Work-1
training scripts rather than inferred.

Sources:
    DenseNet121  Journal_Submission/source_scripts/cnn_benchmark/DenseNet121_Adam.py
    Xception     Journal_Submission/source_scripts/cnn_benchmark/Xception_Adam.py
"""

import os

# Dataset and checkpoint roots live outside the repository; override with PHD_ROOT.
PHD = os.environ.get("PHD_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

TRUNKS = {
    "densenet121": {
        "ckpt": f"{PHD}/Result_Analysis/DenseNet_Adam/Adam_best_model.h5",
        "img_size": 224,
        "preprocess": "imagenet_std",  # /255 then (x - mean) / std
        "backbone_layer": "densenet121",
        "dense_layer": "dense",
        "expected_backbone_dim": 1024,
        "work1_macro_auc": 0.7892,
        "expected_expert_auc": 0.7934,  # independently reproduced by t0c_verify_checkpoint.py
        "work1_nih_retention": 0.842,
    },
    "xception": {
        "ckpt": f"{PHD}/Result_Analysis/Xception_Adam/Xception_Adam_best_model.h5",
        "img_size": 299,
        "preprocess": "tf_scale",  # x / 127.5 - 1.0
        "backbone_layer": "xception",
        "dense_layer": "dense",
        "expected_backbone_dim": 2048,
        "work1_macro_auc": 0.7213,
        "work1_nih_retention": 0.960,  # transfers far better than DenseNet
    },
}


def get(name):
    if name not in TRUNKS:
        raise ValueError(f"unknown trunk '{name}'. Available: {list(TRUNKS)}")
    return TRUNKS[name]


def emb_dir(base, trunk):
    """DenseNet keeps the original path so existing results stay valid."""
    return (
        f"{base}/embeddings" if trunk == "densenet121" else f"{base}/embeddings_{trunk}"
    )
