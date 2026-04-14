# BREATHE

**B.R.E.A.T.H.E.** — *Blended Respiratory Environmental-Aware Triage & Health Estimator*

This repo contains our midpoint experiments on **CheXpert+**: tabular models that predict **severe vs not severe** chest findings using only patient demographics, with optional comparison to imaging and multimodal work from the rest of the team.

---

## Quick start

1. **Python environment** (recommended: a dedicated conda env so dependencies stay clean):

   ```bash
   conda create -n breathe python=3.11 -y
   conda activate breathe
   pip install -r requirements.txt
   ```

2. **Download data** (not stored in git — see below) into `data/`.

3. **Run tabular Part 1 (demographics baseline)**:

   ```bash
   python midpoint/demographics/run.py \
     --csv data/df_chexpert_plus_240401.csv \
     --labels-jsonl data/chexbert_labels/impression_fixed.json \
     --output-dir midpoint/results
   ```

   You’ll see class balance, **Logistic Regression** and **Random Forest** metrics, and files will be written under `midpoint/results/`.

4. **Run imaging models (image-only) and generate embeddings**:

   ```bash
   python midpoint/imaging/run.py \
     --csv data/df_chexpert_plus_240401.csv \
     --labels-jsonl data/chexbert_labels/impression_fixed.json \
     --extractor resnet18 \
     --output-dir midpoint/results/imaging
   ```

   Notes:
   - Use `--extractor densenet121` to switch backbones.
   - If `path_to_image` values in your CSV are relative, add `--image-root /abs/path/to/images`.
   - Add `--max-samples 200` for a quick smoke run.
   - XGBoost is optional and disabled by default; enable with `--include-xgboost`.

5. **Run tabular Part 2 + early multimodal model**:

   ```bash
   python midpoint/multimodal/run.py \
     --csv data/df_chexpert_plus_240401.csv \
     --labels-jsonl data/chexbert_labels/impression_fixed.json \
     --embeddings midpoint/results/imaging/imaging_embeddings_resnet18.npz \
     --output-dir midpoint/results/multimodal
   ```

   Notes:
   - The multimodal run concatenates image embeddings with `age`/`sex`.
   - XGBoost is optional here too; enable with `--include-xgboost`.

---

## What each experiment does

All tracks predict a binary **Severe** label from CheXbert-derived conditions:

| Feature | Notes |
|--------|--------|
| **Age** | Numeric |
| **Sex** | Encoded as `1` for Male, `0` otherwise |

**Important:** The main CheXpert+ CSV does **not** ship with ready-made pathology columns for our severity definition. We build the target from **CheXbert** (report-based) predictions for four conditions — Pneumonia, Edema, Consolidation, Pleural Effusion — then define **Severe = 1** if *any* of those is positive (after dropping uncertain `-1` labels). That is a **proxy** for severity, not the original CheXpert image label matrix.

Rows are matched between the demographics table and CheXbert output using **`path_to_image`**. After cleaning, we use an **80/20 train–test split** with `random_state=42` for reproducibility.

### Tabular Part 1 (`midpoint/demographics/run.py`)

Features:
- `age`
- `sex` (encoded as `1` for Male, `0` otherwise)

Models:
- Logistic Regression (linear baseline)
- Random Forest (nonlinear baseline)

Outputs include accuracy, F1, confusion matrices, and a short RF vs logistic comparison.

### Imaging (`midpoint/imaging/run.py`)

Pipeline:
- Frozen pretrained feature extractor (`resnet18` or `densenet121`)
- Embedding classifiers: Logistic Regression, MLP, optional XGBoost

Outputs include:
- Image-only accuracy/F1 (plus ROC AUC when available)
- Confusion matrix and ROC plot
- Saved embeddings (`imaging_embeddings_<extractor>.npz`)
- Brief rationale for why imaging captures richer pathology than demographics alone

### Tabular Part 2 + Multimodal (`midpoint/multimodal/run.py`)

Tabular Part 2:
- Gradient Boosting + optional XGBoost on `age`/`sex`
- Feature importance chart (`tabular_feature_importance.png`)

Multimodal:
- Join image embeddings by `path_to_image`
- Concatenate embeddings + `age`/`sex`
- Train early combined classifier and report preliminary accuracy/F1 (or WIP notes)

---

## Data you need locally

The `data/` folder is gitignored. Download from the [Stanford AIMI CheXpert+ dataset page](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1) (or your team’s copy) and place:

- `data/df_chexpert_plus_240401.csv` — demographics + paths + report text, etc.
- `data/chexbert_labels/impression_fixed.json` — JSONL with one object per line; must include `path_to_image` and label columns for the four conditions above.

---

## Outputs (what gets generated)

After successful runs, look in **`midpoint/results/`**:

| File | What it is |
|------|------------|
| `demographics_metrics.txt` | Human-readable summary + both models |
| `demographics_metrics.json` | Same content in JSON |
| `tabular_comparison.csv` | Side-by-side accuracy + F1 for logistic vs RF |
| `demographics_accuracy_f1.png` | Bar chart comparing both models |
| `demographics_confusion_matrix_logistic.png` | Confusion matrix for logistic |
| `demographics_confusion_matrix_random_forest.png` | Confusion matrix for random forest |

From imaging (`midpoint/results/imaging/`):

| File | What it is |
|------|------------|
| `imaging_metrics.txt` | Human-readable image-only metrics |
| `imaging_metrics.json` | Same content in JSON |
| `imaging_confusion_matrix.png` | Confusion matrix for best image-only model |
| `imaging_roc_curve.png` | ROC curve for best model (if probabilities available) |
| `imaging_embeddings_resnet18.npz` | Saved embeddings keyed by `path_to_image` |

From multimodal (`midpoint/results/multimodal/`):

| File | What it is |
|------|------------|
| `multimodal_metrics.txt` | Tabular Part 2 + multimodal summary |
| `multimodal_metrics.json` | Same content in JSON |
| `tabular_feature_importance.csv` | Feature importance values |
| `tabular_feature_importance.png` | Feature importance chart |

---

## Project layout (midpoint)

- `midpoint/demographics/run.py` — tabular pipeline (this README)
- `midpoint/imaging/run.py` — imaging pipeline (frozen CNN embeddings + classifiers)
- `midpoint/multimodal/run.py` — tabular Part 2 + early multimodal pipeline
