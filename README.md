# Feature Efficiency, Conformal Coverage and Subgroup Performance Under Institutional Shift — Analysis Code

Code accompanying *"Cross-Institutional Evaluation of Feature
Compression, Conformal Prediction, and
Age-Related Disparities in Chest Radiography"* (submitted to *Journal Name*).

Every number and figure in the manuscript is reproducible from this code plus the
two public source datasets. No images, model checkpoints or cached embeddings are
redistributed here.

---

## 1. Scope

This repository contains the scripts needed to reproduce **this paper**. The
companion paper on uncertainty-label policies shares the same frozen split and
feature-extraction pipeline but has its own release; the overlapping modules are
byte-identical between the two.

| Included | Not included |
|---|---|
| Analysis code for selection, conformal and subgroup results (`code/`) | CheXpert / NIH images — obtain from the custodians |
| Frozen split manifest with SHA-256 (`splits.json`) | Backbone checkpoints |
| Pinned dependency list (`requirements.txt`) | Embedding cache (regenerate with `extract.py`) |
| Standalone figure generator with embedded values | |

`code/generate_paper2_figures.py` is deliberately self-contained: it carries the
plotted values as literals and reproduces all six figures without the dataset,
the embedding cache or the result files. Reviewers can run it immediately.

---

## 2. Environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

All analyses ran on CPU. `tensorflow-metal` was deliberately not installed, since
its numerical divergence from the CPU path would invalidate the checkpoint
verification the pipeline depends on.

### Dataset location

Project paths are derived from the location of the code, so the repository works
wherever it is cloned. The source datasets live outside the repository and are
expected one level above it:

```
<parent>/
├── Dataset/CheXpert-v1.0-small/
├── Dataset/NIH/
└── <this repository>/code/
```

If your datasets are elsewhere, set `PHD_ROOT` to the directory containing
`Dataset/`, and optionally `PS3_BASE` to the repository root:

```bash
export PHD_ROOT=/path/to/data-parent
```

---

## 3. Layout

`code/` is flat by design. The modules import one another directly
(`import heads`), so moving files into subpackages breaks every entry point.

**Shared infrastructure**

| File | Role |
|---|---|
| `trunks.py` | Backbone registry, preprocessing spec, embedding paths |
| `preprocessing.py` | Image preprocessing, replicated from the original training scripts |
| `splits.py` | Builds and verifies the frozen patient-level partition |
| `extract.py` | One forward pass per backbone; writes the embedding cache |
| `heads.py` | Policy application, patient resampling, per-finding head fitting |
| `metrics.py` | Per-class and joint metrics, bootstrap intervals |
| `calibration.py` | Temperature scaling and threshold fitting |
| `multiplicity.py` | Benjamini–Hochberg FDR, shared by every analysis |
| `tier1.py`, `sweep.py` | Baseline heads and the policy sweep that fixes the per-finding configuration consumed downstream |

**Feature selection — manuscript Figures 1–3**

| File | Role |
|---|---|
| `filters.py` | Mutual information, ANOVA-F and approximate mRMR ranking |
| `group_lasso.py` | L1 selector at a target retained fraction |
| `controls.py` | Matched-size controls: top-*k* MI, PCA, random subsets |
| `efficiency.py` | Fit time, inference latency and memory measurement |
| `selection_rules.py` | Rule A (internal AUC) and Rule B (worst internal environment) |
| `phase4_selection.py` | Runner; equivalence testing and efficiency benchmarking |
| `phase6_h3_external.py` | External retention readout comparing the two rules |

**Conformal prediction — manuscript Figures 4–5**

| File | Role |
|---|---|
| `conformal.py` | Split and Mondrian conformal procedures |
| `risk_control.py` | False-negative-rate conformal risk control |
| `coverage_shift.py` | Coverage under institutional shift, weighted mitigation |
| `phase5_conformal.py` | Conformal runner |

**External evaluation and subgroups — manuscript Figure 6**

| File | Role |
|---|---|
| `external.py` | External set loading, retention, label-mapping caveats |
| `fairness.py` | Subgroup AUC, equalised-odds gaps, underpowered flagging |
| `fairness_audit.py` | Internal-only subgroup audit |
| `phase6_terminal.py` | Terminal evaluation; requires `--confirm-terminal` |

**Reporting**

| File | Role |
|---|---|
| `export_results.py` | Rebuilds every tidy CSV from the result JSONs |
| `export_coverage.py` | Coverage table across backbones |
| `apply_fdr.py` | Applies FDR correction to the exported comparison tables |
| `paper2_numbers.py` | Prints every value quoted in the manuscript |
| `generate_paper2_figures.py` | All six figures, standalone |

---

## 4. Reproducing the results

### Figures only — no dataset required

```bash
cd code
python generate_paper2_figures.py --dpi 600
```

### Full pipeline

Obtain CheXpert and NIH ChestX-ray14 from their custodians and set the dataset
paths, then:

```bash
cd code

python splits.py --verify                  # confirm the manifest hash matches
python extract.py --trunk densenet121      # ~1.5 h per backbone, CPU
python extract.py --trunk xception

# per-finding configuration consumed by the later stages
python tier1.py --trunk xception --C 0.1
python sweep.py --trunk xception --C 0.1 --eval-on train_c \
       --tag sweep_train_c_xception_C0.1

# Figures 1-2: selection, equivalence, efficiency  (~20 h at 14 findings)
python phase4_selection.py --trunk xception --C 0.1 --seeds 5 \
       --fractions 0.10,0.25,0.50 --labels all --n-random 5

# Figures 4-5: conformal coverage and risk control
python phase5_conformal.py --trunk xception --C 0.1 --alpha 0.1
python phase5_conformal.py --trunk densenet121 --C 0.1 --alpha 0.1

# Figure 6 and external results - terminal sets, opened once
python phase6_terminal.py --trunk xception --C 0.1 --confirm-terminal
python phase6_terminal.py --trunk densenet121 --C 0.1 --confirm-terminal

# Figure 3: external retention, Rule A vs Rule B
python phase6_h3_external.py --trunk xception --C 0.1 --confirm-terminal

python export_results.py && python apply_fdr.py && python export_coverage.py
python paper2_numbers.py
```

`paper2_numbers.py` prints every value quoted in the manuscript directly from the
result files, so each table and in-text figure can be checked against its source.

---

## 5. Protocol constraints enforced in the code

These are not stylistic preferences; relaxing any of them changes the reported
results.

1. Calibration, thresholds and **all** selection decisions use the calibration
   partition only.
2. The external set is opened once, after every internal decision is frozen. The
   terminal scripts refuse to run without `--confirm-terminal` and record the
   opening timestamp in their output.
3. `phase6_h3_external.py` reads selected feature indices back from the Phase 4
   result and **aborts rather than re-selecting**, so it cannot silently become a
   second selection pass.
4. Regularisation strength is held fixed across compared conditions.
5. Seeds resample patients, not model initialisation — the estimator is convex.
6. Subgroups below 30 positive cases are flagged `underpowered`, not dropped and
   not quoted.
7. Equivalence uses a margin fixed before analysis (Δ = 0.02 AUC); the minimum
   retention effect worth claiming is set in `phase6_h3_external.py`
   (`MIN_RETENTION_EFFECT = 0.01`). Neither should be relaxed to obtain a positive
   result.

---

## 6. Data availability and licence

CheXpert: <https://stanfordmlgroup.github.io/competitions/chexpert/>
NIH ChestX-ray14: <https://nihcc.app.box.com/v/ChestXray-NIHCC>

Both remain under their original terms of use and must be obtained from their
custodians. This repository redistributes neither images nor derived image data.

Code licence: [choose and state, e.g. MIT or Apache-2.0 — add a `LICENSE` file].

## 7. Citation

> [Kirti Swagat Mohanty]. Cross-Institutional Evaluation of Feature
Compression, Conformal Prediction, and
Age-Related Disparities in Chest Radiography. *Journal Name*. [2026]. DOI: [to be assigned]

Archived release: [Zenodo DOI once deposited].
