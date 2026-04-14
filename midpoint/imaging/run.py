from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

SEVERE_LABELS = ["Pneumonia", "Edema", "Consolidation", "Pleural Effusion"]
JOIN_KEY = "path_to_image"


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


def _fit_and_score(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    out = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        out["probs"] = probs
        try:
            out["roc_auc"] = float(roc_auc_score(y_test, probs))
        except ValueError:
            # Can happen when a split contains one class only.
            pass
    out["preds"] = preds
    return out


def _evaluate_image_only(
    emb_df: pd.DataFrame, include_xgboost: bool
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray | None]:
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    X = emb_df[emb_cols].copy()
    y = emb_df["Severe"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "mlp": MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=250, random_state=42),
    }

    xgb_unavailable_note = None
    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
            )
        except Exception as exc:
            note = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            xgb_unavailable_note = (
                "XGBoost unavailable in this environment; install/repair xgboost and OpenMP "
                f"(macOS: `brew install libomp`). Details: {note}"
            )

    metrics: dict[str, dict] = {}
    for name, model in models.items():
        try:
            result = _fit_and_score(model, X_train, y_train, X_test, y_test)
            metrics[name] = {
                k: v for k, v in result.items() if k not in {"preds", "probs"}
            }
            metrics[name]["_preds"] = result["preds"]
            metrics[name]["_probs"] = result.get("probs")
        except Exception as exc:
            note = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            metrics[name] = {
                "status": "wip",
                "note": f"Model failed to train/evaluate in current environment: {note}",
            }

    if not include_xgboost:
        metrics["xgboost"] = {
            "status": "wip",
            "note": "XGBoost is optional and currently disabled. Re-run with --include-xgboost to try it.",
        }
    elif "xgboost" not in models:
        metrics["xgboost"] = {
            "status": "wip",
            "note": xgb_unavailable_note
            or "xgboost package not installed; run `pip install xgboost` to enable.",
        }

    eligible = [name for name in metrics if "f1" in metrics[name]]
    if not eligible:
        raise RuntimeError("No imaging classifiers produced valid metrics.")
    best_name = max(eligible, key=lambda n: metrics[n]["f1"])
    best_preds = np.asarray(metrics[best_name].pop("_preds"))
    best_probs = metrics[best_name].pop("_probs")

    for name in list(metrics.keys()):
        metrics[name].pop("_preds", None)
        metrics[name].pop("_probs", None)

    summary = {
        "n_samples": int(len(emb_df)),
        "class_counts": y.value_counts().to_dict(),
        "class_distribution": y.value_counts(normalize=True).to_dict(),
        "models": metrics,
        "best_model_for_plots": best_name,
    }
    return summary, y_test.to_numpy(), best_preds, best_probs


def _plot_confusion_and_roc(
    output_dir: Path,
    y_test: np.ndarray,
    best_preds: np.ndarray,
    best_probs: np.ndarray | None,
    best_name: str,
) -> None:
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{best_name} confusion matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "imaging_confusion_matrix.png")
    plt.close(fig)

    if best_probs is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, best_probs, ax=ax)
        ax.set_title(f"{best_name} ROC curve")
        plt.tight_layout()
        fig.savefig(output_dir / "imaging_roc_curve.png")
        plt.close(fig)


def _imaging_rationale() -> str:
    return (
        "Imaging captures richer pathology information than demographics alone because "
        "pixel-level lung patterns directly encode consolidation, effusions, edema "
        "distribution, and disease extent, while age/sex only provide population-level "
        "risk priors. Frozen CNN embeddings preserve these spatial-texture cues and "
        "therefore can separate severe vs non-severe findings more directly than tabular "
        "demographics-only models."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Midpoint imaging experiment: frozen ResNet18/DenseNet feature extractor + "
            "image-embedding classifiers."
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
        choices=["resnet18", "densenet121"],
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

    emb_df, skipped_missing_images = _extract_embeddings(
        rows=rows,
        image_root=image_root,
        extractor_name=args.extractor,
        batch_size=args.batch_size,
        max_samples=max_samples,
    )

    np.savez_compressed(
        output_dir / f"imaging_embeddings_{args.extractor}.npz",
        path_to_image=emb_df[JOIN_KEY].to_numpy(),
        embeddings=emb_df[[c for c in emb_df.columns if c.startswith("emb_")]].to_numpy(),
    )

    summary, y_test, best_preds, best_probs = _evaluate_image_only(
        emb_df, include_xgboost=args.include_xgboost
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
    )

    json_path = output_dir / "imaging_metrics.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    txt_path = output_dir / "imaging_metrics.txt"
    with txt_path.open("w") as f:
        f.write("=== Imaging Models (Image-only) ===\n")
        f.write(f"Extractor: {summary['extractor']}\n")
        f.write(f"N samples used: {summary['n_samples']}\n")
        f.write(f"Skipped missing images: {summary['skipped_missing_images']}\n")
        f.write(f"Best model for plots: {summary['best_model_for_plots']}\n\n")
        for model_name, metrics in summary["models"].items():
            f.write(f"{model_name}:\n")
            if "accuracy" in metrics:
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  F1: {metrics['f1']:.4f}\n")
                if "roc_auc" in metrics:
                    f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n")
                f.write(f"  Confusion matrix: {metrics['confusion_matrix']}\n")
            else:
                f.write(f"  {metrics}\n")
            f.write("\n")

        f.write("Why imaging is richer than demographics:\n")
        f.write(summary["imaging_vs_demographics_summary"])
        f.write("\n")

    print(f"Wrote metrics JSON: {json_path}")
    print(f"Wrote metrics TXT: {txt_path}")
    print(f"Wrote confusion matrix plot: {output_dir / 'imaging_confusion_matrix.png'}")
    print(f"Wrote ROC plot (if probability available): {output_dir / 'imaging_roc_curve.png'}")


if __name__ == "__main__":
    main()

