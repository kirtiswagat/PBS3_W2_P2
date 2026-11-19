"""Regenerate every figure in Paper 2 from hard-coded values.

Self-contained: reads nothing from results/, needs only numpy and matplotlib.
All values were transcribed from the analysis outputs when the manuscript was
prepared, so the figures are equivalent in content to the published versions.

    Figure_1_feature_equivalence.png   AUC vs subset size with equivalence band
    Figure_2_efficiency.png            compute/memory cost and selector comparison
    Figure_3_external_retention.png    shift-aware vs AUC-only selection on NIH
    Figure_4_coverage_shift.png        conformal coverage, internal vs external
    Figure_5_risk_control.png          FNR control and the set-size null
    Figure_6_fairness.png              subgroup AUC gaps by age and sex

Usage:
    python generate_paper2_figures.py
    python generate_paper2_figures.py --figure 1 4 --dpi 300 --format tiff
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#1f6fb4"
ORANGE = "#e07b1a"
TEAL = "#0a7d72"
RED = "#b3312c"
VIOLET = "#6a4c9c"
GREY = "#5a6a78"
GREEN = "#2e8b45"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "axes.edgecolor": "#39444d",
        "axes.labelcolor": "#17212b",
        "text.color": "#17212b",
        "xtick.color": "#39444d",
        "ytick.color": "#39444d",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

FULL_DIM = 2048  # Xception penultimate width
TOST_DELTA = 0.02  # prespecified equivalence margin, AUC units

# --------------------------------------------------------------------------
# Figure 1 - feature selection: (pathology, full-dim AUC, then per fraction
# (k, AUC, TOST verdict)) for fractions 0.10 / 0.25 / 0.50.
# --------------------------------------------------------------------------
SELECTION = [
    (
        "No Finding",
        0.841256,
        (50, 0.833354, "eq"),
        (119, 0.836407, "eq"),
        (260, 0.839217, "eq"),
    ),
    (
        "Enlarged Cardiomediastinum",
        0.630247,
        (55, 0.607560, "ne"),
        (135, 0.617053, "eq"),
        (266, 0.618922, "eq"),
    ),
    (
        "Cardiomegaly",
        0.846140,
        (54, 0.836440, "eq"),
        (117, 0.839784, "eq"),
        (252, 0.843390, "eq"),
    ),
    (
        "Lung Opacity",
        0.701614,
        (55, 0.697557, "eq"),
        (115, 0.698601, "eq"),
        (234, 0.700253, "eq"),
    ),
    (
        "Lung Lesion",
        0.734602,
        (50, 0.727540, "eq"),
        (133, 0.732014, "eq"),
        (258, 0.732296, "eq"),
    ),
    (
        "Edema",
        0.832008,
        (53, 0.826530, "eq"),
        (136, 0.829489, "eq"),
        (263, 0.831225, "eq"),
    ),
    (
        "Consolidation",
        0.729066,
        (45, 0.731250, "eq"),
        (124, 0.733982, "eq"),
        (264, 0.734146, "eq"),
    ),
    (
        "Pneumonia",
        0.718011,
        (50, 0.709669, "eq"),
        (131, 0.718388, "eq"),
        (266, 0.721177, "eq"),
    ),
    (
        "Atelectasis",
        0.679104,
        (50, 0.678557, "eq"),
        (124, 0.679804, "eq"),
        (239, 0.680290, "eq"),
    ),
    (
        "Pneumothorax",
        0.768152,
        (50, 0.704596, "ne"),
        (118, 0.733743, "ne"),
        (260, 0.743455, "ne"),
    ),
    (
        "Pleural Effusion",
        0.859103,
        (48, 0.853537, "eq"),
        (125, 0.855822, "eq"),
        (255, 0.857175, "eq"),
    ),
    (
        "Pleural Other",
        0.765792,
        (54, 0.762624, "eq"),
        (118, 0.769287, "eq"),
        (273, 0.769078, "eq"),
    ),
    (
        "Fracture",
        0.691033,
        (46, 0.674009, "ne"),
        (129, 0.679204, "eq"),
        (229, 0.683429, "eq"),
    ),
    (
        "Support Devices",
        0.776928,
        (53, 0.738120, "ne"),
        (121, 0.750366, "ne"),
        (225, 0.757340, "ne"),
    ),
]

# --------------------------------------------------------------------------
# Figure 2 - cost per retained fraction, and mean AUC by selection method
# at the 25% operating point (14 pathologies each).
# --------------------------------------------------------------------------
EFFICIENCY = [
    # (fraction, mean k, fit seconds, predict ms per 1k, memory MB)
    (0.10, 51, 0.068, 0.0109, 23.9),
    (0.25, 125, 0.200, 0.0202, 60.3),
    (0.50, 253, 0.512, 0.0390, 122.6),
]
METHOD_AUC = [
    ("L1 (group-sparse)", 0.7481, BLUE),
    ("Rule A (best Train-B AUC)", 0.7467, TEAL),
    ("PCA", 0.7456, VIOLET),
    ("Rule B (worst-environment)", 0.7447, ORANGE),
    ("Random subsets", 0.7426, GREY),
    ("Top-k mutual information", 0.7419, RED),
]

# --------------------------------------------------------------------------
# Figure 3 - external retention on NIH. Rule A selects on internal AUC,
# Rule B on worst internal environment; both at matched subset size.
# --------------------------------------------------------------------------
H3 = [
    # (pathology, fraction, k, retention Rule A, retention Rule B, BH p or None)
    ("Atelectasis", 0.10, 50, 1.0219, 1.0219, None),
    ("Atelectasis", 0.25, 124, 1.0193, 1.0239, 0.529),
    ("Atelectasis", 0.50, 239, 1.0253, 1.0229, 0.529),
    ("Cardiomegaly", 0.10, 54, 0.9703, 0.9703, None),
    ("Cardiomegaly", 0.25, 117, 0.9713, 0.9713, None),
    ("Cardiomegaly", 0.50, 252, 0.9635, 0.9724, 0.020),
    ("Consolidation", 0.10, 45, 0.9308, 0.9320, 0.631),
    ("Consolidation", 0.25, 124, 0.9305, 0.9320, 0.631),
    ("Consolidation", 0.50, 264, 0.9306, 0.9327, 0.529),
    ("Edema", 0.10, 53, 0.9241, 0.9244, 0.529),
    ("Edema", 0.25, 136, 0.9303, 0.9284, 0.529),
    ("Edema", 0.50, 263, 0.9280, 0.9272, 0.631),
    ("Pleural Effusion", 0.10, 48, 0.8957, 0.8958, 0.529),
    ("Pleural Effusion", 0.25, 125, 0.8969, 0.8978, 0.529),
    ("Pleural Effusion", 0.50, 255, 0.8965, 0.8962, 0.529),
]
H3_MIN_EFFECT = 0.01  # prespecified smallest retention gain worth claiming

# --------------------------------------------------------------------------
# Figure 4 - conformal coverage, nominal 0.90, internal vs NIH, with the
# weighted-conformal mitigation.
# --------------------------------------------------------------------------
COVERAGE = {
    "DenseNet121": [
        ("Cardiomegaly", 0.9043, 0.9572, 0.9989),
        ("Edema", 0.9032, 0.9644, 0.9999),
        ("Consolidation", 0.8971, 0.7843, 0.9994),
        ("Atelectasis", 0.9006, 0.9156, 0.9990),
        ("Pleural Effusion", 0.9038, 0.8738, 1.0000),
    ],
    "Xception": [
        ("Cardiomegaly", 0.9063, 0.9512, 0.9151),
        ("Edema", 0.9068, 0.9694, 0.9717),
        ("Consolidation", 0.8961, 0.8206, 0.9445),
        ("Atelectasis", 0.9012, 0.9111, 0.9105),
        ("Pleural Effusion", 0.9085, 0.8124, 0.7895),
    ],
}
NOMINAL = 0.90

# --------------------------------------------------------------------------
# Figure 5 - conformal risk control and prediction-set size (Xception).
# --------------------------------------------------------------------------
CONFORMAL = [
    # (pathology, uncertain fraction, split set size, Mondrian set size, FNR)
    ("No Finding", 0.000, 0.975, 1.311, 0.0983),
    ("Enlarged Cardiomediastinum", 0.054, 0.945, 1.724, 0.0974),
    ("Cardiomegaly", 0.036, 1.028, 1.333, 0.0984),
    ("Lung Opacity", 0.023, 1.594, 1.595, 0.0986),
    ("Lung Lesion", 0.006, 0.921, 1.496, 0.0974),
    ("Edema", 0.062, 1.279, 1.355, 0.0982),
    ("Consolidation", 0.128, 0.966, 1.543, 0.0973),
    ("Pneumonia", 0.084, 0.912, 1.583, 0.0981),
    ("Atelectasis", 0.156, 1.265, 1.603, 0.0990),
    ("Pneumothorax", 0.014, 0.994, 1.504, 0.0987),
    ("Pleural Effusion", 0.050, 1.312, 1.313, 0.0988),
    ("Pleural Other", 0.009, 0.906, 1.481, 0.0954),
    ("Fracture", 0.003, 0.933, 1.620, 0.0988),
    ("Support Devices", 0.005, 1.471, 1.484, 0.0985),
]
SETSIZE_RHO, SETSIZE_P = 0.099, 0.737
TARGET_FNR = 0.10

# --------------------------------------------------------------------------
# Figure 6 - largest adequately powered subgroup AUC gaps (n+ >= 30).
# --------------------------------------------------------------------------
FAIRNESS = [
    # (trunk, pathology, attribute, split, best group, best AUC, worst group, worst AUC)
    ("DenseNet121", "Lung Opacity", "age", "expert", "60-75", 0.932, ">75", 0.743),
    ("DenseNet121", "Fracture", "age", "internal", "40-60", 0.776, ">75", 0.663),
    (
        "DenseNet121",
        "Enl. Cardiomediastinum",
        "age",
        "expert",
        ">75",
        0.692,
        "60-75",
        0.583,
    ),
    ("Xception", "Fracture", "age", "internal", "40-60", 0.750, ">75", 0.655),
    ("Xception", "Lung Opacity", "age", "internal", "<40", 0.753, ">75", 0.660),
    ("DenseNet121", "Pleural Other", "age", "internal", "<40", 0.824, ">75", 0.731),
    ("DenseNet121", "Cardiomegaly", "age", "NIH", ">75", 0.860, "60-75", 0.768),
    ("Xception", "No Finding", "age", "internal", "40-60", 0.851, ">75", 0.760),
    ("Xception", "Lung Lesion", "age", "internal", ">75", 0.763, "<40", 0.673),
    ("DenseNet121", "No Finding", "age", "internal", "40-60", 0.847, ">75", 0.760),
    ("DenseNet121", "Atelectasis", "age", "internal", "<40", 0.731, ">75", 0.645),
    ("Xception", "Atelectasis", "age", "internal", "<40", 0.729, ">75", 0.643),
]


def _save(fig, outdir, name, dpi, fmt):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def figure_1(outdir, dpi, fmt):
    """AUC against retained dimensions, with the prespecified equivalence band."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    cmap = plt.get_cmap("tab20")
    for i, row in enumerate(SELECTION):
        lab, full = row[0], row[1]
        ks = [row[2][0], row[3][0], row[4][0]]
        aucs = [row[2][1], row[3][1], row[4][1]]
        failed = any(row[j][2] == "ne" for j in (2, 3, 4))
        deltas = [a - full for a in aucs]
        color = RED if failed else cmap(i % 20)
        ax1.plot(
            ks,
            deltas,
            "-o",
            color=color,
            linewidth=2.2 if failed else 1.4,
            markersize=6 if failed else 4.5,
            alpha=1.0 if failed else 0.75,
            zorder=4 if failed else 2,
            label=lab if failed else None,
        )

    ax1.axhspan(-TOST_DELTA, TOST_DELTA, color=TEAL, alpha=0.14, zorder=0)
    ax1.axhline(0, color=GREY, linewidth=1.1, linestyle="-")
    ax1.axhline(-TOST_DELTA, color=TEAL, linewidth=1.3, linestyle="--")
    ax1.text(
        268,
        -TOST_DELTA + 0.0016,
        f"equivalence margin  Δ = {TOST_DELTA}",
        fontsize=8.6,
        color=TEAL,
        ha="right",
    )
    ax1.set_xlabel(f"Retained dimensions (of {FULL_DIM:,})", fontsize=10.5)
    ax1.set_ylabel("AUC change vs full representation", fontsize=10.5)
    ax1.set_title(
        "(a) Reduced representations stay inside the margin\nfor 12 of 14 findings",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(alpha=0.28, linestyle=":")
    ax1.set_axisbelow(True)
    ax1.legend(
        fontsize=8.4,
        title="Not equivalent",
        title_fontsize=8.6,
        loc="lower right",
        framealpha=0.95,
    )

    fracs = ["10%", "25%", "50%"]
    eq = [sum(1 for r in SELECTION if r[j][2] == "eq") for j in (2, 3, 4)]
    ne = [14 - e for e in eq]
    yy = np.arange(3)
    ax2.barh(yy, eq, color=TEAL, height=0.55, label="Equivalent")
    ax2.barh(yy, ne, left=eq, color=RED, height=0.55, label="Not equivalent")
    for i, (e, n) in enumerate(zip(eq, ne)):
        ax2.text(
            e / 2,
            i,
            f"{e}",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        if n:
            ax2.text(
                e + n / 2,
                i,
                f"{n}",
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                fontweight="bold",
            )
    ax2.set_yticks(yy)
    ax2.set_yticklabels(
        [f"{f}\n(k ≈ {int(k)})" for f, (_, k, *_rest) in zip(fracs, EFFICIENCY)],
        fontsize=9.5,
    )
    ax2.set_xlabel("Number of findings (of 14)", fontsize=10.5)
    ax2.set_title(
        "(b) Equivalence verdicts by retained fraction", fontsize=11, fontweight="bold"
    )
    ax2.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax2.grid(axis="x", alpha=0.28, linestyle=":")
    ax2.set_axisbelow(True)
    ax2.set_xlim(0, 15)

    fig.tight_layout()
    return _save(fig, outdir, "Figure_1_feature_equivalence", dpi, fmt)


def figure_2(outdir, dpi, fmt):
    """Compute and memory cost, and how little selector choice matters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.8))

    ks = [e[1] for e in EFFICIENCY]
    mem = [e[4] for e in EFFICIENCY]
    lat = [e[3] for e in EFFICIENCY]
    fitt = [e[2] for e in EFFICIENCY]

    ax1.plot(
        ks, mem, "-o", color=BLUE, linewidth=2.2, markersize=8, label="Memory (MB)"
    )
    ax1.set_xlabel("Retained dimensions", fontsize=10.5)
    ax1.set_ylabel("Feature-matrix memory (MB)", color=BLUE, fontsize=10.5)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.grid(alpha=0.28, linestyle=":")
    ax1.set_axisbelow(True)

    axb = ax1.twinx()
    axb.plot(
        ks,
        lat,
        "-s",
        color=ORANGE,
        linewidth=2.2,
        markersize=8,
        label="Inference latency",
    )
    axb.plot(
        ks, fitt, "-^", color=TEAL, linewidth=2.2, markersize=8, label="Fit time (s)"
    )
    axb.set_ylabel("Latency (ms / 1k) and fit time (s)", color=ORANGE, fontsize=10.5)
    axb.tick_params(axis="y", labelcolor=ORANGE)

    for k, m, p in zip(ks, mem, EFFICIENCY):
        ax1.annotate(
            f"{int(p[0] * 100)}%",
            (k, m),
            textcoords="offset points",
            xytext=(6, -14),
            fontsize=9,
            color=GREY,
        )

    lines = ax1.get_lines() + axb.get_lines()
    ax1.legend(
        lines,
        [l.get_label() for l in lines],
        fontsize=8.8,
        loc="upper left",
        framealpha=0.95,
    )
    ax1.set_title(
        "(a) Cost scales linearly with retained dimensions",
        fontsize=11,
        fontweight="bold",
    )

    names = [m[0] for m in METHOD_AUC][::-1]
    vals = [m[1] for m in METHOD_AUC][::-1]
    cols = [m[2] for m in METHOD_AUC][::-1]
    yy = np.arange(len(names))
    ax2.barh(yy, vals, color=cols, height=0.62)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(names, fontsize=9.2)
    lo = min(vals) - 0.004
    ax2.set_xlim(lo, max(vals) + 0.002)
    for i, v in enumerate(vals):
        ax2.text(v + 0.0002, i, f"{v:.4f}", va="center", fontsize=8.8)
    rand = dict((m[0], m[1]) for m in METHOD_AUC)["Random subsets"]
    ax2.axvline(rand, color=GREY, linestyle="--", linewidth=1.4)
    ax2.text(
        rand - 0.0004,
        0.15,
        "random baseline",
        rotation=90,
        fontsize=8.4,
        color=GREY,
        va="bottom",
        ha="right",
    )
    ax2.set_xlabel("Mean AUC across 14 findings (25% of dimensions)", fontsize=10.5)
    ax2.set_title(
        "(b) Selector choice spans only 0.006 AUC", fontsize=11, fontweight="bold"
    )
    ax2.grid(axis="x", alpha=0.28, linestyle=":")
    ax2.set_axisbelow(True)

    fig.tight_layout()
    return _save(fig, outdir, "Figure_2_efficiency", dpi, fmt)


def figure_3(outdir, dpi, fmt):
    """Shift-aware selection against AUC-only selection, on the external set."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.4, 6.2), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    labels = [f"{p}\n{int(f * 100)}% (k={k})" for p, f, k, *_ in H3]
    y = np.arange(len(H3))
    ra = [h[3] for h in H3]
    rb = [h[4] for h in H3]

    for i, (a, b) in enumerate(zip(ra, rb)):
        ax1.plot([a, b], [i, i], color=GREY, linewidth=1.3, zorder=1, alpha=0.7)
    ax1.scatter(
        ra,
        y,
        s=72,
        color=BLUE,
        zorder=3,
        label="Rule A (internal AUC)",
        edgecolor="white",
        linewidth=1.0,
    )
    ax1.scatter(
        rb,
        y,
        s=72,
        color=ORANGE,
        zorder=3,
        marker="D",
        label="Rule B (worst environment)",
        edgecolor="white",
        linewidth=1.0,
    )
    ax1.axvline(1.0, color=GREEN, linestyle=":", linewidth=1.5)
    ax1.text(1.002, -0.7, "full retention", fontsize=8.2, color=GREEN)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=7.8)
    ax1.set_xlabel("External retention (NIH AUC ÷ internal AUC)", fontsize=10.5)
    ax1.set_title(
        "(a) The two selection rules are indistinguishable",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(fontsize=8.8, loc="lower left", framealpha=0.95)
    ax1.grid(axis="x", alpha=0.28, linestyle=":")
    ax1.set_axisbelow(True)

    deltas = [h[4] - h[3] for h in H3]
    cols = [TEAL if d > 0 else (RED if d < 0 else GREY) for d in deltas]
    ax2.barh(y, deltas, color=cols, height=0.62)
    ax2.axvline(0, color="black", linewidth=1.1)
    ax2.axvline(H3_MIN_EFFECT, color=VIOLET, linestyle="--", linewidth=1.8)
    ax2.text(
        H3_MIN_EFFECT + 0.0004,
        len(H3) - 1.2,
        f"prespecified minimum\neffect ({H3_MIN_EFFECT})",
        fontsize=8.2,
        color=VIOLET,
        va="top",
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Retention difference (Rule B − Rule A)", fontsize=10.5)
    ax2.set_title(
        "(b) No comparison reaches the minimum effect", fontsize=11, fontweight="bold"
    )
    ax2.set_xlim(-0.004, H3_MIN_EFFECT + 0.004)
    ax2.grid(axis="x", alpha=0.28, linestyle=":")
    ax2.set_axisbelow(True)

    mean_d = float(np.mean(deltas))
    fig.suptitle(
        f"Shift-aware feature selection does not improve external retention "
        f"(mean difference {mean_d:+.4f}, 8 of 15 favour Rule B)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return _save(fig, outdir, "Figure_3_external_retention", dpi, fmt)


def figure_4(outdir, dpi, fmt):
    """Conformal coverage internally and under institutional shift."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0), sharey=True)

    for ax, (trunk, rows) in zip(axes, COVERAGE.items()):
        labs = [r[0] for r in rows]
        x = np.arange(len(rows))
        internal = [r[1] for r in rows]
        external = [r[2] for r in rows]
        recovered = [r[3] for r in rows]

        ax.bar(x - 0.26, internal, width=0.25, color=TEAL, label="Internal")
        ext_cols = [RED if e < NOMINAL else BLUE for e in external]
        ax.bar(x, external, width=0.25, color=ext_cols, label="External (NIH)")
        ax.bar(
            x + 0.26,
            recovered,
            width=0.25,
            color=ORANGE,
            alpha=0.9,
            label="External, weighted conformal",
        )

        ax.axhline(NOMINAL, color="black", linestyle="--", linewidth=1.6)
        ax.text(
            len(rows) - 0.4,
            NOMINAL + 0.004,
            "nominal 1 − α = 0.90",
            fontsize=8.6,
            ha="right",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=28, ha="right", fontsize=9)
        ax.set_ylim(0.72, 1.04)
        ax.set_title(trunk, fontsize=11.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.28, linestyle=":")
        ax.set_axisbelow(True)
        for i, e in enumerate(external):
            if e < NOMINAL:
                ax.text(
                    i,
                    e - 0.022,
                    "under-\ncovers",
                    ha="center",
                    fontsize=7.6,
                    color=RED,
                    fontweight="bold",
                )

    axes[0].set_ylabel("Empirical coverage", fontsize=10.5)
    axes[0].legend(fontsize=8.6, loc="lower left", framealpha=0.95)
    fig.suptitle(
        "Conformal coverage holds internally but fails for two findings under shift",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _save(fig, outdir, "Figure_4_coverage_shift", dpi, fmt)


def figure_5(outdir, dpi, fmt):
    """FNR risk control, and the absent link between set size and uncertainty."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.8))

    labs = [c[0] for c in CONFORMAL]
    y = np.arange(len(CONFORMAL))
    fnr = [c[4] for c in CONFORMAL]
    ax1.barh(y, fnr, color=TEAL, height=0.62)
    ax1.axvline(TARGET_FNR, color=RED, linestyle="--", linewidth=1.8)
    ax1.text(
        TARGET_FNR + 0.0004, -0.9, f"target α = {TARGET_FNR}", fontsize=8.8, color=RED
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(labs, fontsize=8.6)
    ax1.set_xlim(0.090, 0.1025)
    ax1.set_xlabel("Empirical false-negative rate", fontsize=10.5)
    ax1.set_title(
        "(a) Risk control holds for every finding", fontsize=11, fontweight="bold"
    )
    ax1.grid(axis="x", alpha=0.28, linestyle=":")
    ax1.set_axisbelow(True)
    for i, v in enumerate(fnr):
        ax1.text(v + 0.00008, i, f"{v:.4f}", va="center", fontsize=7.6)

    unc = [c[1] for c in CONFORMAL]
    split = [c[2] for c in CONFORMAL]
    mond = [c[3] for c in CONFORMAL]
    ax2.scatter(
        unc,
        split,
        s=82,
        color=BLUE,
        edgecolor="white",
        linewidth=1.0,
        label="Split conformal",
        zorder=3,
    )
    ax2.scatter(
        unc,
        mond,
        s=82,
        color=VIOLET,
        marker="^",
        edgecolor="white",
        linewidth=1.0,
        label="Mondrian conformal",
        zorder=3,
    )
    z = np.polyfit(unc, split, 1)
    xs = np.linspace(0, max(unc), 40)
    ax2.plot(xs, np.poly1d(z)(xs), color=GREY, linestyle="--", linewidth=1.6)
    ax2.set_xlabel("Uncertain fraction of training mentions", fontsize=10.5)
    ax2.set_ylabel("Mean prediction-set size", fontsize=10.5)
    ax2.set_title(
        f"(b) Set size does not track uncertainty\n"
        f"ρ = {SETSIZE_RHO}, p = {SETSIZE_P} (null)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.legend(fontsize=8.8, framealpha=0.95)
    ax2.grid(alpha=0.28, linestyle=":")
    ax2.set_axisbelow(True)
    for lab, u, s in zip(labs, unc, split):
        if u > 0.08 or s > 1.4:
            ax2.annotate(
                lab,
                (u, s),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=7.8,
                color=GREY,
            )

    fig.tight_layout()
    return _save(fig, outdir, "Figure_5_risk_control", dpi, fmt)


def figure_6(outdir, dpi, fmt):
    """Largest adequately powered subgroup AUC disparities."""
    rows = sorted(FAIRNESS, key=lambda r: r[5] - r[7])
    labs = [f"{r[1]}\n{r[0]} · {r[3]}" for r in rows]
    y = np.arange(len(rows))
    best = [r[5] for r in rows]
    worst = [r[7] for r in rows]
    gaps = [b - w for b, w in zip(best, worst)]

    fig, ax = plt.subplots(figsize=(10.6, 7.4))
    for i, (b, w) in enumerate(zip(best, worst)):
        ax.plot([w, b], [i, i], color=GREY, linewidth=2.0, zorder=1, alpha=0.65)
    ax.scatter(
        best,
        y,
        s=88,
        color=TEAL,
        zorder=3,
        edgecolor="white",
        linewidth=1.0,
        label="Best-performing age band",
    )
    ax.scatter(
        worst,
        y,
        s=88,
        color=RED,
        zorder=3,
        marker="v",
        edgecolor="white",
        linewidth=1.0,
        label="Worst-performing age band",
    )

    for i, r in enumerate(rows):
        ax.text(
            r[5] + 0.006,
            i,
            f"Δ {gaps[i]:.3f}   ({r[4]} vs {r[6]})",
            va="center",
            fontsize=8.2,
            color=GREY,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8.4)
    ax.set_xlabel("Subgroup AUC", fontsize=10.5)
    ax.set_xlim(0.55, 1.07)
    ax.set_title(
        "Age-related performance disparities\n"
        "adequately powered subgroups only (≥ 30 positive cases)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.28, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, outdir, "Figure_6_fairness", dpi, fmt)


FIGURES = {1: figure_1, 2: figure_2, 3: figure_3, 4: figure_4, 5: figure_5, 6: figure_6}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", default="manuscript_figures")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--format", default="png", choices=["png", "tiff", "jpg", "pdf"])
    ap.add_argument("--figure", type=int, nargs="*", choices=sorted(FIGURES))
    args = ap.parse_args()

    wanted = args.figure or sorted(FIGURES)
    print(f"Generating {len(wanted)} figure(s) at {args.dpi} dpi -> {args.outdir}/")
    for n in wanted:
        FIGURES[n](args.outdir, args.dpi, args.format)
    print("done.")


if __name__ == "__main__":
    main()
