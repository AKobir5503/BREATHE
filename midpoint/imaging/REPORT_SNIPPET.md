# Imaging section — report snippets

## Label wording (use consistently with tabular / multimodal)

**Positive class (1)** — *High-Risk Respiratory Condition*: CheXbert-derived presence of any of Pneumonia, Edema, Consolidation, or Pleural Effusion (rows with uncertain `-1` dropped).

**Negative class (0)** — *Lower-Risk Condition*: none of the above.

## Methods (one paragraph)

We extracted frozen ImageNet-pretrained CNN embeddings from frontal chest radiographs (`path_to_image`), using ResNet-18, DenseNet-121, and EfficientNet-B0 without fine-tuning. On each embedding matrix we compared linear (scaled logistic regression), RBF SVM, MLP, and optionally XGBoost using **stratified 5-fold cross-validation** with **F1** as the primary selection metric; we report mean ± SD for accuracy, F1, precision, recall, and ROC-AUC. The classifier with highest mean CV F1 was refit on an **80/20 stratified hold-out** (`random_state=42`) for the final confusion matrix and ROC curve.

## Limitations (short)

- **Proxy labels** from report-derived CheXbert outputs, not a radiologist image-level severity score.
- **Frozen embeddings** — no domain adaptation to CheXpert+; fine-tuning would likely change results.
- **Subset size and class imbalance** — performance variance across CV folds; SVM can be slow on large *n* × embedding dimension.
- **Tabular signal** — age/sex may correlate with severity proxy; image-only should be interpreted alongside demographics baselines.

## Figures to include

1. `imaging_cv_f1_by_classifier.png` — classifier comparison for the chosen backbone.
2. `imaging_confusion_matrix.png` / `imaging_roc_curve.png` — best model, hold-out.
3. Optional: `imaging_cv_f1_by_extractor.png` from `merge_summaries.py` — backbone comparison.
