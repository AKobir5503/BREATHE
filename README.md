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

3. **Run the tabular pipeline**:

   ```bash
   python midpoint/demographics/run.py \
     --csv data/df_chexpert_plus_240401.csv \
     --labels-jsonl data/chexbert_labels/impression_fixed.json \
     --output-dir midpoint/results
   ```

   You’ll see class balance, **Logistic Regression** and **Random Forest** metrics, and files will be written under `midpoint/results/`.

---

## What the tabular experiment does

We predict a binary **Severe** label from:

| Feature | Notes |
|--------|--------|
| **Age** | Numeric |
| **Sex** | Encoded as `1` for Male, `0` otherwise |

**Important:** The main CheXpert+ CSV does **not** ship with ready-made pathology columns for our severity definition. We build the target from **CheXbert** (report-based) predictions for four conditions — Pneumonia, Edema, Consolidation, Pleural Effusion — then define **Severe = 1** if *any* of those is positive (after dropping uncertain `-1` labels). That is a **proxy** for severity, not the original CheXpert image label matrix.

Rows are matched between the demographics table and CheXbert output using **`path_to_image`**. After cleaning, we use an **80/20 train–test split** with `random_state=42` for reproducibility.

**Models compared:** Logistic Regression (linear baseline) and Random Forest (nonlinear). Metrics include accuracy, F1, confusion matrices, and a short comparison of RF vs logistic.

---

## Data you need locally

The `data/` folder is gitignored. Download from the [Stanford AIMI CheXpert+ dataset page](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1) (or your team’s copy) and place:

- `data/df_chexpert_plus_240401.csv` — demographics + paths + report text, etc.
- `data/chexbert_labels/impression_fixed.json` — JSONL with one object per line; must include `path_to_image` and label columns for the four conditions above.

---

## Outputs (what gets generated)

After a successful run, look in **`midpoint/results/`**:

| File | What it is |
|------|------------|
| `demographics_metrics.txt` | Human-readable summary + both models |
| `demographics_metrics.json` | Same content in JSON |
| `tabular_comparison.csv` | Side-by-side accuracy + F1 for logistic vs RF |
| `demographics_accuracy_f1.png` | Bar chart comparing both models |
| `demographics_confusion_matrix_logistic.png` | Confusion matrix for logistic |
| `demographics_confusion_matrix_random_forest.png` | Confusion matrix for random forest |

---

## Project layout (midpoint)

- `midpoint/demographics/run.py` — tabular pipeline (this README)
- `midpoint/imaging/` — imaging experiments (placeholder)
- `midpoint/multimodal/` — multimodal experiments (placeholder)
