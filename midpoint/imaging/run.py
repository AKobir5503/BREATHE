from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEVERE_LABELS = ["Pneumonia", "Edema", "Consolidation", "Pleural Effusion"]
JOIN_KEY = "path_to_image"
RANDOM_STATE = 42
CV_SPLITS_DEFAULT = 5

LABEL_DEFINITION = (
    "Positive class (1) = High-Risk Respiratory Condition: CheXbert-derived presence of "
    "any of Pneumonia, Edema, Consolidation, or Pleural Effusion (uncertain -1 labels dropped). "
    "Negative class (0) = Lower-Risk Condition (none of the above)."
)


def _load_metadata(csv_path: Path, labels_jsonl_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    labels = pd.read_json(labels_jsonl_path, lines=True)

    if JOIN_KEY not in df.columns:
        raise ValueError(f"Demographics CSV must include '{JOIN_KEY}'")

    missing_label_cols = [c for c in [JOIN_KEY, *SEVERE_LABELS] if c not in labels.columns]
    if missing_label_cols:
        raise ValueError(f"Labels JSONL missing columns: {missing_label_cols}")

    merged = df[[JOIN_KEY]].merge(labels[[JOIN_KEY, *SEVERE_LABELS]], on=JOIN_KEY, how="inner")
    merged = merged.dropna(subset=SEVERE_LABELS)
    for col in SEVERE_LABELS:
        merged = merged[merged[col] != -1]
    merged["Severe"] = (merged[SEVERE_LABELS].sum(axis=1) > 0).astype(int)
    return merged[[JOIN_KEY, "Severe"]]


def _resolve_image_path(path_to_image: str, image_root: Path | None) -> Path:
    p = Path(path_to_image)
    if p.exists():
        return p
    if image_root is not None:
        candidate = image_root / path_to_image
        if candidate.exists():
            return candidate
        candidate_name = image_root / p.name
        if candidate_name.exists():
            return candidate_name
    raise FileNotFoundError(f"Image file not found for path_to_image={path_to_image}")


def _build_extractor(name: str):
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
    except ImportError as exc:
        raise ImportError(
            "PyTorch/torchvision are required for imaging embeddings. "
            "Install with: pip install torch torchvision"
        ) from exc

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        embedding_dim = model.fc.in_features
        model.fc = nn.Identity()
        preprocess = weights.transforms()
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        embedding_dim = model.classifier.in_features
        model.classifier = nn.Identity()
        preprocess = weights.transforms()
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        # Last linear layer input size (1280 for B0)
        last_linear = model.classifier[-1]
        embedding_dim = int(last_linear.in_features)
        model.classifier = nn.Identity()
        preprocess = weights.transforms()
    else:
        raise ValueError(f"Unsupported extractor: {name}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, preprocess, embedding_dim, device, torch


def _extract_embeddings(
    rows: pd.DataFrame,
    image_root: Path | None,
    extractor_name: str,
    batch_size: int,
    max_samples: int | None,
) -> tuple[pd.DataFrame, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for image loading. Install with: pip install pillow") from exc

    model, preprocess, embedding_dim, device, torch = _build_extractor(extractor_name)

    sample_rows = rows.copy()
    if max_samples is not None:
        sample_rows = sample_rows.head(max_samples).copy()

    image_paths: list[str] = []
    labels: list[int] = []
    valid_files: list[Path] = []
    skipped = 0
    for _, row in sample_rows.iterrows():
        try:
            resolved = _resolve_image_path(row[JOIN_KEY], image_root)
            valid_files.append(resolved)
            image_paths.append(str(row[JOIN_KEY]))
            labels.append(int(row["Severe"]))
        except FileNotFoundError:
            skipped += 1

    if not valid_files:
        raise ValueError("No valid images found after resolving paths.")

    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(valid_files), batch_size):
            batch_files = valid_files[i : i + batch_size]
            batch_tensors = []
            for fpath in batch_files:
                img = Image.open(fpath).convert("RGB")
                batch_tensors.append(preprocess(img))
            batch = torch.stack(batch_tensors).to(device)
            emb = model(batch).detach().cpu().numpy()
            emb = emb.reshape(emb.shape[0], -1)
            all_embeddings.append(emb)

    embedding_array = np.concatenate(all_embeddings, axis=0)
    if embedding_array.shape[1] != embedding_dim:
        embedding_dim = embedding_array.shape[1]

    emb_cols = [f"emb_{i}" for i in range(embedding_dim)]
    emb_df = pd.DataFrame(embedding_array, columns=emb_cols)
    emb_df[JOIN_KEY] = image_paths
    emb_df["Severe"] = labels
    return emb_df[[JOIN_KEY, "Severe", *emb_cols]], skipped


def _load_embeddings_from_npz(
    npz_path: Path,
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Merge precomputed embeddings with label rows; skip rows missing from NPZ."""
    loaded = np.load(npz_path, allow_pickle=True)
    if "embeddings" not in loaded.files or "path_to_image" not in loaded.files:
        raise ValueError("NPZ must contain 'embeddings' and 'path_to_image' arrays.")

    paths = np.asarray(loaded["path_to_image"]).astype(str)
    embeddings = np.asarray(loaded["embeddings"], dtype=np.float64)
    if embeddings.shape[0] != len(paths):
        raise ValueError("Embeddings row count does not match path_to_image length.")

    emb_dim = embeddings.shape[1]
    emb_cols = [f"emb_{i}" for i in range(emb_dim)]
    emb_by_path: dict[str, np.ndarray] = {str(p): embeddings[i] for i, p in enumerate(paths)}

    keep_paths: list[str] = []
    keep_labels: list[int] = []
    keep_embs: list[np.ndarray] = []
    skipped_no_npz = 0

    for _, row in rows.iterrows():
        pth = str(row[JOIN_KEY])
        if pth not in emb_by_path:
            skipped_no_npz += 1
            continue
        keep_paths.append(pth)
        keep_labels.append(int(row["Severe"]))
        keep_embs.append(emb_by_path[pth])

    if not keep_paths:
        raise ValueError("No rows matched between labels and NPZ path_to_image.")

    emb_df = pd.DataFrame(np.stack(keep_embs, axis=0), columns=emb_cols)
    emb_df[JOIN_KEY] = keep_paths
    emb_df["Severe"] = keep_labels
    return emb_df[[JOIN_KEY, "Severe", *emb_cols]], skipped_no_npz


def _cross_val_metrics(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
) -> dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs: list[float] = []
    f1s: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    rocs: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        est = clone(estimator)
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        est.fit(X_tr, y_tr)
        preds = est.predict(X_val)
        accs.append(float(accuracy_score(y_val, preds)))
        f1s.append(float(f1_score(y_val, preds, zero_division=0)))
        precs.append(float(precision_score(y_val, preds, zero_division=0)))
        recs.append(float(recall_score(y_val, preds, zero_division=0)))
        if hasattr(est, "predict_proba"):
            try:
                probs = est.predict_proba(X_val)[:, 1]
                if len(np.unique(y_val)) < 2:
                    rocs.append(float("nan"))
                else:
                    rocs.append(float(roc_auc_score(y_val, probs)))
            except ValueError:
                rocs.append(float("nan"))
        else:
            rocs.append(float("nan"))

    rocs_arr = np.asarray(rocs, dtype=float)
    return {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
        "cv_precision_mean": float(np.mean(precs)),
        "cv_precision_std": float(np.std(precs)),
        "cv_recall_mean": float(np.mean(recs)),
        "cv_recall_std": float(np.std(recs)),
        "cv_roc_auc_mean": float(np.nanmean(rocs_arr)),
        "cv_roc_auc_std": float(np.nanstd(rocs_arr)),
    }


def _build_classifiers(include_xgboost: bool) -> tuple[dict[str, object], str | None]:
    """Return sklearn estimators (some wrapped in Pipeline with StandardScaler)."""
    models: dict[str, object] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 64),
                        max_iter=500,
                        random_state=RANDOM_STATE,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=15,
                    ),
                ),
            ]
        ),
    }

    xgb_note: str | None = None
    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            )
        except Exception as exc:
            note = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            xgb_note = (
                "XGBoost unavailable in this environment; install/repair xgboost and OpenMP "
                f"(macOS: `brew install libomp`). Details: {note}"
            )
    return models, xgb_note


def _evaluate_image_only(
    emb_df: pd.DataFrame,
    include_xgboost: bool,
    n_cv_splits: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray | None, str]:
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    X = emb_df[emb_cols].copy()
    y = emb_df["Severe"].copy()

    classifiers, xgb_unavailable_note = _build_classifiers(include_xgboost)

    model_cv: dict[str, dict] = {}
    for name, est in classifiers.items():
        try:
            cv_block = _cross_val_metrics(est, X, y, n_splits=n_cv_splits)
            model_cv[name] = cv_block
        except Exception as exc:
            note = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            model_cv[name] = {
                "status": "failed",
                "note": f"CV failed: {note}",
            }

    if not include_xgboost:
        model_cv["xgboost"] = {
            "status": "disabled",
            "note": "XGBoost disabled. Re-run with --include-xgboost to try it.",
        }
    elif "xgboost" not in classifiers:
        model_cv["xgboost"] = {
            "status": "unavailable",
            "note": xgb_unavailable_note
            or "xgboost package not installed; run `pip install xgboost` to enable.",
        }

    eligible = [
        n
        for n, m in model_cv.items()
        if "cv_f1_mean" in m and "status" not in m
    ]
    if not eligible:
        raise RuntimeError("No imaging classifiers produced valid CV metrics.")

    best_name = max(eligible, key=lambda n: model_cv[n]["cv_f1_mean"])
    if best_name not in classifiers:
        raise RuntimeError(f"Best model {best_name!r} not found in trained classifiers.")
    best_estimator = clone(classifiers[best_name])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    best_estimator.fit(X_train, y_train)
    preds = best_estimator.predict(X_test)
    probs: np.ndarray | None = None
    if hasattr(best_estimator, "predict_proba"):
        probs = best_estimator.predict_proba(X_test)[:, 1]

    final_metrics: dict[str, float | list | None] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    if probs is not None:
        try:
            final_metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
        except ValueError:
            final_metrics["roc_auc"] = None

    for name in list(model_cv.keys()):
        entry = model_cv[name]
        if "cv_f1_mean" in entry:
            model_cv[name] = {
                **entry,
                "selected_for_final_plots": name == best_name,
            }
        else:
            model_cv[name]["selected_for_final_plots"] = False

    summary = {
        "n_samples": int(len(emb_df)),
        "train_size_final_split": int(len(y_train)),
        "test_size_final_split": int(len(y_test)),
        "class_counts": y.value_counts().to_dict(),
        "class_distribution": y.value_counts(normalize=True).to_dict(),
        "label_definition": LABEL_DEFINITION,
        "cv_splits": int(n_cv_splits),
        "models": model_cv,
        "best_model_for_plots": best_name,
        "final_holdout_metrics": final_metrics,
    }
    return summary, y_test.to_numpy(), np.asarray(preds), probs, best_name


def _plot_confusion_and_roc(
    output_dir: Path,
    y_test: np.ndarray,
    best_preds: np.ndarray,
    best_probs: np.ndarray | None,
    best_name: str,
    extractor: str,
) -> None:
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{extractor} / {best_name}\nconfusion matrix (hold-out)")
    plt.tight_layout()
    fig.savefig(output_dir / "imaging_confusion_matrix.png")
    plt.close(fig)

    if best_probs is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, best_probs, ax=ax)
        ax.set_title(f"{extractor} / {best_name}\nROC curve (hold-out)")
        plt.tight_layout()
        fig.savefig(output_dir / "imaging_roc_curve.png")
        plt.close(fig)


def _plot_cv_f1_bar(output_dir: Path, models: dict, extractor: str, n_splits: int) -> None:
    names = [k for k, v in models.items() if "cv_f1_mean" in v]
    if not names:
        return
    means = [models[k]["cv_f1_mean"] for k in names]
    stds = [models[k].get("cv_f1_std", 0.0) for k in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Mean CV F1")
    ax.set_title(f"{extractor}: classifier comparison ({n_splits}-fold stratified CV)")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / "imaging_cv_f1_by_classifier.png")
    plt.close(fig)


def _imaging_rationale() -> str:
    return (
        "Imaging captures pathology-relevant spatial patterns (e.g., consolidation, effusion, edema) "
        "that are not encoded in age/sex alone. Frozen pretrained CNN embeddings provide a strong "
        "off-the-shelf image representation; performance can still be limited by subset size, "
        "proxy labels derived from reports, lack of fine-tuning, and class imbalance."
    )


def _append_comparison_csv(
    comparison_csv: Path,
    extractor: str,
    summary: dict,
) -> None:
    best = summary["best_model_for_plots"]
    models = summary["models"]
    row: dict[str, str | float | int | None] = {
        "extractor": extractor,
        "best_classifier": best,
        "n_samples": summary["n_samples"],
        "cv_splits": summary["cv_splits"],
    }
    if best in models and "cv_f1_mean" in models[best]:
        for k in (
            "cv_f1_mean",
            "cv_f1_std",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_precision_mean",
            "cv_recall_mean",
            "cv_roc_auc_mean",
        ):
            row[k] = models[best].get(k)
    fh = None
    try:
        file_exists = comparison_csv.exists()
        comparison_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        if file_exists:
            with comparison_csv.open("r", newline="") as rf:
                reader = csv.DictReader(rf)
                if reader.fieldnames:
                    for fn in reader.fieldnames:
                        if fn not in fieldnames:
                            fieldnames.append(fn)
        fh = comparison_csv.open("a", newline="")
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    finally:
        if fh is not None:
            fh.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Imaging experiment: frozen CNN embeddings (ResNet18, DenseNet121, EfficientNet-B0) "
            "+ classifiers with stratified k-fold CV and hold-out evaluation."
        )
    )
    parser.add_argument("--csv", type=str, required=True, help="Demographics CSV path")
    parser.add_argument(
        "--labels-jsonl",
        type=str,
        default="data/chexbert_labels/impression_fixed.json",
        help="CheXbert labels JSONL path",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="Optional root directory to prepend to path_to_image values",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="resnet18",
        choices=["resnet18", "densenet121", "efficientnet_b0"],
        help="Frozen CNN backbone used to extract embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for faster preliminary runs (0 means all rows)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="midpoint/results",
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Enable XGBoost classifier on image embeddings (optional; may require OpenMP setup).",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="If imaging_embeddings_<extractor>.npz exists under output-dir, load it instead of re-extracting.",
    )
    parser.add_argument(
        "--embeddings-npz",
        type=str,
        default="",
        help="Explicit path to embeddings NPZ (path_to_image + embeddings). Overrides extraction.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=CV_SPLITS_DEFAULT,
        help="Number of stratified CV folds (default 5).",
    )
    parser.add_argument(
        "--comparison-csv",
        type=str,
        default="",
        help="Optional CSV path to append one summary row per run (multi-backbone comparison).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    labels_path = Path(args.labels_jsonl)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels JSONL not found: {labels_path}")

    image_root = Path(args.image_root) if args.image_root.strip() else None
    if image_root is not None and not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_metadata(csv_path, labels_path)
    max_samples = args.max_samples if args.max_samples > 0 else None
    if max_samples is not None:
        rows = rows.head(max_samples).copy()

    emb_npz_default = output_dir / f"imaging_embeddings_{args.extractor}.npz"
    embeddings_npz_path = Path(args.embeddings_npz) if args.embeddings_npz.strip() else None

    skipped_missing_images = 0
    if embeddings_npz_path is not None:
        if not embeddings_npz_path.exists():
            raise FileNotFoundError(f"Embeddings NPZ not found: {embeddings_npz_path}")
        emb_df, skipped_missing_images = _load_embeddings_from_npz(embeddings_npz_path, rows)
    elif args.reuse_embeddings and emb_npz_default.exists():
        emb_df, skipped_missing_images = _load_embeddings_from_npz(emb_npz_default, rows)
    else:
        emb_df, skipped_missing_images = _extract_embeddings(
            rows=rows,
            image_root=image_root,
            extractor_name=args.extractor,
            batch_size=args.batch_size,
            max_samples=None,
        )

    np.savez_compressed(
        emb_npz_default,
        path_to_image=emb_df[JOIN_KEY].to_numpy(),
        embeddings=emb_df[[c for c in emb_df.columns if c.startswith("emb_")]].to_numpy(),
    )

    summary, y_test, best_preds, best_probs, _ = _evaluate_image_only(
        emb_df,
        include_xgboost=args.include_xgboost,
        n_cv_splits=args.cv_splits,
    )
    summary["extractor"] = args.extractor
    summary["skipped_missing_images"] = int(skipped_missing_images)
    summary["imaging_vs_demographics_summary"] = _imaging_rationale()

    _plot_confusion_and_roc(
        output_dir=output_dir,
        y_test=y_test,
        best_preds=best_preds,
        best_probs=best_probs,
        best_name=summary["best_model_for_plots"],
        extractor=args.extractor,
    )
    _plot_cv_f1_bar(output_dir, summary["models"], args.extractor, args.cv_splits)

    json_path = output_dir / "imaging_metrics.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    txt_path = output_dir / "imaging_metrics.txt"
    with txt_path.open("w") as f:
        f.write("=== Imaging Models (Image-only) ===\n")
        f.write(f"Extractor: {summary['extractor']}\n")
        f.write(f"Label definition: {summary['label_definition']}\n")
        f.write(f"N samples used: {summary['n_samples']}\n")
        f.write(f"Skipped / unmatched for NPZ or missing files: {summary['skipped_missing_images']}\n")
        f.write(f"CV folds: {summary['cv_splits']}\n")
        f.write(f"Final hold-out train size: {summary['train_size_final_split']}\n")
        f.write(f"Final hold-out test size: {summary['test_size_final_split']}\n")
        f.write(f"Best model (by mean CV F1): {summary['best_model_for_plots']}\n\n")

        f.write("--- Per-model stratified CV (mean +/- std) ---\n")
        for model_name, metrics in summary["models"].items():
            f.write(f"{model_name}:\n")
            if "cv_f1_mean" in metrics:
                f.write(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.4f} +/- {metrics['cv_accuracy_std']:.4f}\n")
                f.write(f"  CV F1: {metrics['cv_f1_mean']:.4f} +/- {metrics['cv_f1_std']:.4f}\n")
                f.write(
                    f"  CV Precision: {metrics['cv_precision_mean']:.4f} +/- {metrics['cv_precision_std']:.4f}\n"
                )
                f.write(f"  CV Recall: {metrics['cv_recall_mean']:.4f} +/- {metrics['cv_recall_std']:.4f}\n")
                f.write(f"  CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} +/- {metrics['cv_roc_auc_std']:.4f}\n")
            else:
                f.write(f"  {metrics}\n")
            f.write("\n")

        fh = summary["final_holdout_metrics"]
        f.write("--- Final stratified hold-out (80/20, seed=42) for best CV model ---\n")
        f.write(f"  Accuracy: {fh['accuracy']:.4f}\n")
        f.write(f"  F1: {fh['f1']:.4f}\n")
        f.write(f"  Precision: {fh['precision']:.4f}\n")
        f.write(f"  Recall: {fh['recall']:.4f}\n")
        if fh.get("roc_auc") is not None:
            f.write(f"  ROC AUC: {fh['roc_auc']:.4f}\n")
        f.write(f"  Confusion matrix: {fh['confusion_matrix']}\n\n")

        f.write("Analysis note:\n")
        f.write(summary["imaging_vs_demographics_summary"])
        f.write("\n")

    if args.comparison_csv.strip():
        _append_comparison_csv(Path(args.comparison_csv), args.extractor, summary)

    print(f"Wrote metrics JSON: {json_path}")
    print(f"Wrote metrics TXT: {txt_path}")
    print(f"Wrote confusion matrix plot: {output_dir / 'imaging_confusion_matrix.png'}")
    print(f"Wrote ROC plot (if probability available): {output_dir / 'imaging_roc_curve.png'}")
    print(f"Wrote CV F1 bar chart: {output_dir / 'imaging_cv_f1_by_classifier.png'}")


if __name__ == "__main__":
    main()
