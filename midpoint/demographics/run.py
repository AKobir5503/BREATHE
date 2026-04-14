from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


SEVERE_LABELS = ["Pneumonia", "Edema", "Consolidation", "Pleural Effusion"]
FEATURE_COLUMNS = ["age", "sex"]
JOIN_KEY = "path_to_image"


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

    merged["sex"] = (merged["sex"] == "Male").astype(int)

    X = merged[FEATURE_COLUMNS].copy()
    y = merged["Severe"].copy()
    return X, y


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logistic = LogisticRegression(max_iter=1000)
    random_forest = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )

    logistic_metrics = evaluate_model(logistic, X_train, y_train, X_test, y_test)
    rf_metrics = evaluate_model(random_forest, X_train, y_train, X_test, y_test)

    return {
        "n_samples": int(len(y)),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "class_counts": y.value_counts().to_dict(),
        "class_counts_normalized": y.value_counts(normalize=True).to_dict(),
        "models": {
            "logistic_regression": logistic_metrics,
            "random_forest": rf_metrics,
        },
    }


def save_metrics_and_plots(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logistic = metrics["models"]["logistic_regression"]
    random_forest = metrics["models"]["random_forest"]

    txt_path = output_dir / "demographics_metrics.txt"
    with txt_path.open("w") as f:
        f.write("Class distribution (counts):\n")
        for k, v in metrics["class_counts"].items():
            f.write(f"  Severe={k}: {v}\n")
        f.write("\nClass distribution (normalized):\n")
        for k, v in metrics["class_counts_normalized"].items():
            f.write(f"  Severe={k}: {v:.4f}\n")
        f.write("\nLogistic Regression:\n")
        f.write(f"  Accuracy: {logistic['accuracy']:.4f}\n")
        f.write(f"  F1 Score: {logistic['f1']:.4f}\n")
        f.write(f"  Confusion matrix [[TN, FP], [FN, TP]]: {logistic['confusion_matrix']}\n")
        f.write("\nRandom Forest:\n")
        f.write(f"  Accuracy: {random_forest['accuracy']:.4f}\n")
        f.write(f"  F1 Score: {random_forest['f1']:.4f}\n")
        f.write(
            f"  Confusion matrix [[TN, FP], [FN, TP]]: {random_forest['confusion_matrix']}\n"
        )

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

    import json

    json_path = output_dir / "demographics_metrics.json"
    with json_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    comparison_df = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "accuracy": logistic["accuracy"],
                "f1": logistic["f1"],
            },
            {
                "model": "random_forest",
                "accuracy": random_forest["accuracy"],
                "f1": random_forest["f1"],
            },
        ]
    )
    comparison_df.to_csv(output_dir / "tabular_comparison.csv", index=False)

    cm = np.array(logistic["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title("Logistic Regression\nconfusion matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "demographics_confusion_matrix_logistic.png")
    plt.close(fig)

    cm_rf = np.array(random_forest["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_rf, cmap="Greens")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm_rf):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title("Random Forest\nconfusion matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "demographics_confusion_matrix_random_forest.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    x = np.arange(2)
    width = 0.35
    ax.bar(
        x - width / 2,
        [logistic["accuracy"], logistic["f1"]],
        width,
        label="Logistic",
    )
    ax.bar(
        x + width / 2,
        [random_forest["accuracy"], random_forest["f1"]],
        width,
        label="Random Forest",
    )
    ax.set_xticks(x, ["Accuracy", "F1"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Demographics models\naccuracy and F1")
    plt.tight_layout()
    fig.savefig(output_dir / "demographics_accuracy_f1.png")
    plt.close(fig)


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

    logistic = metrics["models"]["logistic_regression"]
    random_forest = metrics["models"]["random_forest"]

    print("\nLogistic Regression")
    print("Accuracy:", logistic["accuracy"])
    print("F1 Score:", logistic["f1"])
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(logistic["confusion_matrix"]))

    print("\nRandom Forest")
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