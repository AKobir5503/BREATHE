from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[misc, assignment]


SEVERE_LABELS = ["Pneumonia", "Edema", "Consolidation", "Pleural Effusion"]
FEATURE_COLUMNS = ["age", "sex"]
JOIN_KEY = "path_to_image"

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}

# Keep legacy filenames for the original two models (slides/docs may reference them).
CONFUSION_MATRIX_STEM = {
    "logistic_regression": "demographics_confusion_matrix_logistic",
    "random_forest": "demographics_confusion_matrix_random_forest",
    "gradient_boosting": "demographics_confusion_matrix_gradient_boosting",
    "xgboost": "demographics_confusion_matrix_xgboost",
}


def load_and_preprocess(csv_path: Path, labels_jsonl_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    labels = pd.read_json(labels_jsonl_path, lines=True)

    missing_df_cols = [c for c in [JOIN_KEY, *FEATURE_COLUMNS] if c not in df.columns]
    if missing_df_cols:
        raise ValueError(f"Demographics CSV is missing expected columns: {missing_df_cols}")

    missing_label_cols = [c for c in [JOIN_KEY, *SEVERE_LABELS] if c not in labels.columns]
    if missing_label_cols:
        raise ValueError(f"Labels JSONL is missing expected columns: {missing_label_cols}")

    df = df[[JOIN_KEY, *FEATURE_COLUMNS]]
    labels = labels[[JOIN_KEY, *SEVERE_LABELS]]

    merged = df.merge(labels, on=JOIN_KEY, how="inner")
    merged = merged.dropna(subset=[*FEATURE_COLUMNS, *SEVERE_LABELS])

    for col in SEVERE_LABELS:
        merged = merged[merged[col] != -1]

    merged["Severe"] = (merged[SEVERE_LABELS].sum(axis=1) > 0).astype(int)

    # Normalize dtypes so all models (especially XGBoost) receive numeric input.
    merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
    merged["sex"] = (
        merged["sex"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"male": 1, "female": 0})
    )
    merged = merged.dropna(subset=FEATURE_COLUMNS)
    merged["age"] = merged["age"].astype(float)
    merged["sex"] = merged["sex"].astype(int)

    X = merged[FEATURE_COLUMNS].copy()
    y = merged["Severe"].copy()
    return X, y


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    as_numpy: bool = False,
) -> dict:
    X_train_in = X_train.to_numpy(dtype=np.float32) if as_numpy else X_train
    X_test_in = X_test.to_numpy(dtype=np.float32) if as_numpy else X_test
    y_train_in = y_train.to_numpy(dtype=np.int32) if as_numpy else y_train
    y_test_in = y_test.to_numpy(dtype=np.int32) if as_numpy else y_test

    model.fit(X_train_in, y_train_in)
    preds = model.predict(X_test_in)
    result = {
        "status": "trained",
        "accuracy": float(accuracy_score(y_test_in, preds)),
        "f1": float(f1_score(y_test_in, preds)),
        "confusion_matrix": confusion_matrix(y_test_in, preds).tolist(),
    }
    if hasattr(model, "feature_importances_"):
        result["feature_importance"] = {
            feature: float(value)
            for feature, value in zip(FEATURE_COLUMNS, model.feature_importances_)
        }
    return result


def xgboost_result_unavailable(note: str) -> dict:
    return {
        "status": "unavailable",
        "note": note,
        "accuracy": None,
        "f1": None,
        "confusion_matrix": None,
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logistic = LogisticRegression(max_iter=1000)
    random_forest = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    gradient_boosting = GradientBoostingClassifier(random_state=42)

    logistic_metrics = evaluate_model(logistic, X_train, y_train, X_test, y_test)
    rf_metrics = evaluate_model(random_forest, X_train, y_train, X_test, y_test)
    gb_metrics = evaluate_model(gradient_boosting, X_train, y_train, X_test, y_test)

    if XGBClassifier is None:
        xgb_metrics = xgboost_result_unavailable("XGBoost not installed (pip install xgboost).")
    else:
        try:
            xgb = XGBClassifier(
                objective="binary:logistic",
                n_estimators=500,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
                eval_metric="logloss",
            )
            xgb_metrics = evaluate_model(
                xgb,
                X_train,
                y_train,
                X_test,
                y_test,
                as_numpy=True,
            )
        except Exception as e:  # pragma: no cover
            xgb_metrics = xgboost_result_unavailable(f"XGBoost error: {e}")

    return {
        "n_samples": int(len(y)),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "class_counts": y.value_counts().to_dict(),
        "class_counts_normalized": y.value_counts(normalize=True).to_dict(),
        "models": {
            "logistic_regression": logistic_metrics,
            "random_forest": rf_metrics,
            "gradient_boosting": gb_metrics,
            "xgboost": xgb_metrics,
        },
    }


def best_model_by_f1(models: dict) -> tuple[str | None, float | None]:
    best_key: str | None = None
    best_f1: float | None = None
    for key in MODEL_ORDER:
        m = models[key]
        if m.get("status") != "trained":
            continue
        f1 = m["f1"]
        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_key = key
    return best_key, best_f1


def save_metrics_and_plots(metrics: dict, output_dir: Path) -> None:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    models = metrics["models"]
    logistic = models["logistic_regression"]
    random_forest = models["random_forest"]

    txt_path = output_dir / "demographics_metrics.txt"
    with txt_path.open("w") as f:
        f.write("Class distribution (counts):\n")
        for k, v in metrics["class_counts"].items():
            f.write(f"  Severe={k}: {v}\n")
        f.write("\nClass distribution (normalized):\n")
        for k, v in metrics["class_counts_normalized"].items():
            f.write(f"  Severe={k}: {v:.4f}\n")

        for key in MODEL_ORDER:
            m = models[key]
            title = MODEL_DISPLAY[key]
            f.write(f"\n{title}:\n")
            f.write(f"  Status: {m['status']}\n")
            if m["status"] == "trained":
                f.write(f"  Accuracy: {m['accuracy']:.4f}\n")
                f.write(f"  F1 Score: {m['f1']:.4f}\n")
                f.write(
                    f"  Confusion matrix [[TN, FP], [FN, TP]]: {m['confusion_matrix']}\n"
                )
            else:
                f.write(f"  Note: {m.get('note', '')}\n")

        acc_delta = random_forest["accuracy"] - logistic["accuracy"]
        f1_delta = random_forest["f1"] - logistic["f1"]
        f.write("\nComparison (RF - Logistic):\n")
        f.write(f"  Accuracy delta: {acc_delta:+.4f}\n")
        f.write(f"  F1 delta: {f1_delta:+.4f}\n")
        if acc_delta > 0 or f1_delta > 0:
            f.write("  Note: Nonlinear model shows some improvement.\n")
        elif acc_delta < 0 and f1_delta < 0:
            f.write("  Note: Nonlinear model underperforms the linear baseline.\n")
        else:
            f.write("  Note: Mixed result; no clear nonlinear gain.\n")

        best_key, best_f1 = best_model_by_f1(models)
        f.write("\nBest model by F1 (among trained):\n")
        if best_key is not None:
            f.write(f"  {MODEL_DISPLAY[best_key]} ({best_key}): F1 = {best_f1:.4f}\n")
        else:
            f.write("  (none)\n")

    json_path = output_dir / "demographics_metrics.json"
    with json_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    comparison_rows = []
    for key in MODEL_ORDER:
        m = models[key]
        row = {
            "model": key,
            "status": m["status"],
            "accuracy": m.get("accuracy"),
            "f1": m.get("f1"),
        }
        if m["status"] != "trained":
            row["note"] = m.get("note", "")
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "tabular_comparison.csv", index=False)

    trained_models = [k for k in MODEL_ORDER if models[k]["status"] == "trained"]
    best_acc = max((models[k]["accuracy"] for k in trained_models), default=None)
    best_f1 = max((models[k]["f1"] for k in trained_models), default=None)

    def fmt_markdown_score(value: float | None, best: float | None) -> str:
        if value is None:
            return "N/A"
        formatted = f"{value:.4f}"
        if best is not None and value == best:
            return f"**{formatted}**"
        return formatted

    md_path = output_dir / "demographics_results_table.md"
    with md_path.open("w") as f:
        f.write("| Model | Status | Accuracy | F1 |\n")
        f.write("|---|---|---:|---:|\n")
        for key in MODEL_ORDER:
            m = models[key]
            f.write(
                f"| {MODEL_DISPLAY[key]} | {m['status']} | "
                f"{fmt_markdown_score(m.get('accuracy'), best_acc)} | "
                f"{fmt_markdown_score(m.get('f1'), best_f1)} |\n"
            )

    cmap_by_model = {
        "logistic_regression": "Blues",
        "random_forest": "Greens",
        "gradient_boosting": "Oranges",
        "xgboost": "Purples",
    }
    for key in MODEL_ORDER:
        m = models[key]
        if m["status"] != "trained" or m["confusion_matrix"] is None:
            continue
        cm = np.array(m["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap=cmap_by_model[key])
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{MODEL_DISPLAY[key]}\nconfusion matrix")
        plt.tight_layout()
        fig.savefig(output_dir / f"{CONFUSION_MATRIX_STEM[key]}.png")
        plt.close(fig)

    best_key, _ = best_model_by_f1(models)
    if best_key is not None and models[best_key]["confusion_matrix"] is not None:
        cm_best = np.array(models[best_key]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm_best, cmap="BuGn")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        for (i, j), v in np.ndenumerate(cm_best):
            ax.text(j, i, str(v), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Best model: {MODEL_DISPLAY[best_key]}\nconfusion matrix")
        plt.tight_layout()
        fig.savefig(output_dir / "demographics_confusion_matrix_best_model.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    n = len(trained_models)
    width = min(0.8 / n, 0.22) if n else 0.35
    offsets = (np.arange(n) - (n - 1) / 2) * width
    for i, key in enumerate(trained_models):
        m = models[key]
        ax.bar(
            x + offsets[i],
            [m["accuracy"], m["f1"]],
            width,
            label=MODEL_DISPLAY[key],
        )
    if best_key is not None:
        best_f1 = models[best_key]["f1"]
        ax.text(
            1,
            best_f1 + 0.02,
            f"Winner: {MODEL_DISPLAY[best_key]}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x, ["Accuracy", "F1"])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Demographics models\naccuracy and F1")
    plt.tight_layout()
    fig.savefig(output_dir / "demographics_accuracy_f1.png")
    plt.close(fig)

    tree_models = ["random_forest", "gradient_boosting", "xgboost"]
    tree_with_importance = [
        key
        for key in tree_models
        if models[key]["status"] == "trained" and "feature_importance" in models[key]
    ]
    if tree_with_importance:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(FEATURE_COLUMNS))
        n = len(tree_with_importance)
        width = min(0.8 / n, 0.22)
        offsets = (np.arange(n) - (n - 1) / 2) * width
        for i, key in enumerate(tree_with_importance):
            importances = [models[key]["feature_importance"][f] for f in FEATURE_COLUMNS]
            ax.bar(x + offsets[i], importances, width, label=MODEL_DISPLAY[key])
        ax.set_xticks(x, [c.capitalize() for c in FEATURE_COLUMNS])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Importance")
        ax.set_title("Tree model feature importance\nAge vs Sex")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(output_dir / "demographics_feature_importance_tree_models.png")
        plt.close(fig)

    dominant_feature = "age"
    if tree_with_importance:
        avg = {
            feature: float(
                np.mean(
                    [models[k]["feature_importance"][feature] for k in tree_with_importance]
                )
            )
            for feature in FEATURE_COLUMNS
        }
        dominant_feature = max(avg, key=avg.get)
    ensemble_improves = any(
        models[k]["status"] == "trained" and models[k]["f1"] > logistic["f1"]
        for k in tree_models
        if k in models
    )
    verb = "outperform" if ensemble_improves else "do not outperform"
    takeaway = (
        f"Ensemble methods {verb} logistic baseline; "
        f"{dominant_feature} contributes most predictive signal."
    )
    with (output_dir / "demographics_takeaway.txt").open("w") as f:
        f.write(takeaway + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unary demographics baseline: Age + Sex -> Severe"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to df_chexpert_plus_240401.csv",
    )
    parser.add_argument(
        "--labels-jsonl",
        type=str,
        default="data/chexbert_labels/impression_fixed.json",
        help="Path to CheXbert labels JSONL (must include path_to_image + label columns)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="midpoint/results",
        help="Directory to write metrics and plots",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    labels_path = Path(args.labels_jsonl)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels JSONL not found at: {labels_path}")

    X, y = load_and_preprocess(csv_path, labels_path)

    print("Class distribution (counts):")
    print(y.value_counts())
    print("\nClass distribution (normalized):")
    print(y.value_counts(normalize=True))

    metrics = train_and_evaluate(X, y)
    models = metrics["models"]

    logistic = models["logistic_regression"]
    random_forest = models["random_forest"]

    print("\n--- Model summary ---")
    for key in MODEL_ORDER:
        m = models[key]
        label = MODEL_DISPLAY[key]
        if m["status"] == "trained":
            print(
                f"{label}: Accuracy={m['accuracy']:.4f}, F1={m['f1']:.4f}, "
                f"status=trained"
            )
        else:
            print(f"{label}: unavailable — {m.get('note', '')}")

    best_key, best_f1 = best_model_by_f1(models)
    if best_key is not None:
        print(
            f"\nBest model by F1: {MODEL_DISPLAY[best_key]} "
            f"(F1={best_f1:.4f})"
        )
    else:
        print("\nBest model by F1: (no trained models)")

    print("\n--- Details: Logistic Regression ---")
    print("Accuracy:", logistic["accuracy"])
    print("F1 Score:", logistic["f1"])
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(logistic["confusion_matrix"]))

    print("\n--- Details: Random Forest ---")
    print("Accuracy:", random_forest["accuracy"])
    print("F1 Score:", random_forest["f1"])
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(random_forest["confusion_matrix"]))

    acc_delta = random_forest["accuracy"] - logistic["accuracy"]
    f1_delta = random_forest["f1"] - logistic["f1"]
    print("\nComparison (RF - Logistic)")
    print(f"Accuracy delta: {acc_delta:+.4f}")
    print(f"F1 delta: {f1_delta:+.4f}")
    if acc_delta > 0 or f1_delta > 0:
        print("Note: Nonlinear model shows some improvement.")
    elif acc_delta < 0 and f1_delta < 0:
        print("Note: Nonlinear model underperforms the linear baseline.")
    else:
        print("Note: Mixed result; no clear nonlinear gain.")

    save_metrics_and_plots(metrics, Path(args.output_dir))


if __name__ == "__main__":
    main()