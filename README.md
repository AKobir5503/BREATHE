# BREATHE

**B.R.E.A.T.H.E.** — *Blended Respiratory Environmental-Aware Triage & Health Estimator*

This repository contains midpoint experiments on **CheXpert+** to predict a binary severity target (**Severe** vs **Not Severe**) using:
- tabular demographics models,
- image-only models from frozen CNN embeddings,
- and an early multimodal fusion baseline.

## Quick Start

1. **Create/activate environment**

```bash
conda create -n breathe python=3.11 -y
conda activate breathe
pip install -r requirements.txt
```

2. **Prepare local data** (not committed to git)

```bash
mkdir -p data/chexbert_labels
```

Place files at:
- `data/df_chexpert_plus_240401.csv`
- `data/chexbert_labels/impression_fixed.json`

3. **Run demographics (tabular baseline + expanded tabular set)**

```bash
python midpoint/demographics/run.py \
  --csv data/df_chexpert_plus_240401.csv \
  --labels-jsonl data/chexbert_labels/impression_fixed.json \
  --output-dir midpoint/results/demographics
```

4. **Run imaging (image-only, generates embeddings)**

```bash
python midpoint/imaging/run.py \
  --csv data/df_chexpert_plus_240401.csv \
  --labels-jsonl data/chexbert_labels/impression_fixed.json \
  --extractor resnet18 \
  --output-dir midpoint/results/imaging/resnet18
```

5. **Run tabular part 2 + early multimodal**

```bash
python midpoint/multimodal/run.py \
  --csv data/df_chexpert_plus_240401.csv \
  --labels-jsonl data/chexbert_labels/impression_fixed.json \
  --embeddings midpoint/results/imaging/resnet18/imaging_embeddings_resnet18.npz \
  --output-dir midpoint/results/multimodal
```

## Label Definition and Data Join

The project does **not** use a single native "severity" column from CheXpert+.
Instead, we construct a proxy target from CheXbert labels:

- `Pneumonia`
- `Edema`
- `Consolidation`
- `Pleural Effusion`

Rules:
- Drop uncertain labels (`-1`)
- Set `Severe = 1` if **any** of the four conditions is positive, else `0`
- Join records using `path_to_image`

This is a **proxy** severity definition and should be described as such in reports/presentations.

## What Each Script Does

### `midpoint/demographics/run.py`

Tabular modeling with `age` and `sex`:
- Logistic Regression
- Random Forest
- Gradient Boosting
- optional XGBoost (if available)

Produces:
- metrics text/json
- comparison table
- confusion matrices (per model + best model)
- accuracy/F1 chart
- tree-model feature importance chart
- results markdown table and short takeaway text

### `midpoint/imaging/run.py`

Image-only pipeline:
- frozen feature extractor (`resnet18`, `densenet121`, `efficientnet_b0`)
- embeddings classifiers (Logistic Regression, SVM-RBF, MLP, optional XGBoost)
- stratified CV model comparison + holdout evaluation for best CV model

Produces:
- `imaging_metrics.txt` / `imaging_metrics.json`
- confusion matrix + ROC
- CV F1 bar chart by classifier
- embeddings NPZ for reuse in multimodal

Helpful options:
- `--image-root` if CSV paths are relative
- `--max-samples` for quick runs
- `--include-xgboost` to enable XGBoost
- `--reuse-embeddings` or `--embeddings-npz` to skip extraction
- `--comparison-csv` to append extractor-level summary rows

### `midpoint/imaging/merge_summaries.py`

Merges multiple imaging run directories (each with `imaging_metrics.json`) into:
- one comparison CSV
- optional extractor comparison plot

### `midpoint/multimodal/run.py`

Tabular part 2 + early fusion:
- tabular CV + grid search for logistic/random-forest/gradient-boosting (+ optional XGBoost)
- feature importance for tree models
- optional early multimodal classifier by concatenating embeddings + `age/sex`

Produces:
- `multimodal_metrics.txt` / `multimodal_metrics.json`
- `tabular_feature_importance.csv` / `tabular_feature_importance.png`
- `tabular_age_sex_scatter_by_severity.png`
- `tabular_logistic_decision_boundary_age_sex.png`

If embeddings are missing or too few rows merge, multimodal section is marked as WIP/preliminary.

## Typical Output Locations

- `midpoint/results/demographics/` for demographics artifacts
- `midpoint/results/imaging/<extractor>/` for each imaging backbone run
- `midpoint/results/multimodal/` for tabular part 2 + multimodal outputs

## Common Commands

Run imaging with a different extractor:

```bash
python midpoint/imaging/run.py \
  --csv data/df_chexpert_plus_240401.csv \
  --labels-jsonl data/chexbert_labels/impression_fixed.json \
  --extractor densenet121 \
  --output-dir midpoint/results/imaging/densenet121
```

Enable XGBoost (optional):

```bash
python midpoint/multimodal/run.py \
  --csv data/df_chexpert_plus_240401.csv \
  --labels-jsonl data/chexbert_labels/impression_fixed.json \
  --output-dir midpoint/results/multimodal \
  --include-xgboost
```

Run imaging unit test (no full dataset needed):

```bash
pytest tests/test_imaging_eval.py -q
```
