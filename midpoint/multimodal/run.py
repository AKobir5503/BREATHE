from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEVERE_LABELS = ["Pneumonia", "Edema", "Consolidation", "Pleural Effusion"]
TABULAR_FEATURES = ["age", "sex"]
JOIN_KEY = "path_to_image"


def _load_demographics_and_labels(csv_path: Path, labels_jsonl_path: Path) -> pd.DataFrame:
    demographics = pd.read_csv(csv_path)
    labels = pd.read_json(labels_jsonl_path, lines=True)

    missing_demo = [c for c in [JOIN_KEY, *TABULAR_FEATURES] if c not in demographics.columns]
    if missing_demo:
        raise ValueError(f"Demographics CSV missing columns: {missing_demo}")

    missing_labels = [c for c in [JOIN_KEY, *SEVERE_LABELS] if c not in labels.columns]
    if missing_labels:
        raise ValueError(f"Labels JSONL missing columns: {missing_labels}")

    demographics = demographics[[JOIN_KEY, *TABULAR_FEATURES]].copy()
    labels = labels[[JOIN_KEY, *SEVERE_LABELS]].copy()
    merged = demographics.merge(labels, on=JOIN_KEY, how="inner")
    merged = merged.dropna(subset=[*TABULAR_FEATURES, *SEVERE_LABELS])
    for col in SEVERE_LABELS:
        merged = merged[merged[col] != -1]

    merged["sex"] = (merged["sex"] == "Male").astype(int)
    merged["Severe"] = (merged[SEVERE_LABELS].sum(axis=1) > 0).astype(int)
    merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
    merged = merged.dropna(subset=["age"])
    return merged[[JOIN_KEY, "age", "sex", "Severe"]]


def _load_embeddings(embeddings_path: Path) -> pd.DataFrame:
    if embeddings_path.suffix.lower() == ".npz":
        loaded = np.load(embeddings_path, allow_pickle=True)
        if "embeddings" not in loaded.files:
            raise ValueError("NPZ embeddings file must contain key: 'embeddings'")

        path_key = None
        for candidate in ["path_to_image", "paths", "image_paths"]:
            if candidate in loaded.files:
                path_key = candidate
                break
        if path_key is None:
            raise ValueError(
                "NPZ embeddings file must contain one of: path_to_image, paths, image_paths"
            )

        embeddings = np.asarray(loaded["embeddings"])
        paths = np.asarray(loaded[path_key]).astype(str)
        if embeddings.shape[0] != len(paths):
            raise ValueError("Embeddings row count does not match path list length.")

        emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols)
        emb_df[JOIN_KEY] = paths
        return emb_df[[JOIN_KEY, *emb_cols]]

    if embeddings_path.suffix.lower() == ".csv":
        emb_df = pd.read_csv(embeddings_path)
    elif embeddings_path.suffix.lower() in {".parquet", ".pq"}:
        emb_df = pd.read_parquet(embeddings_path)
    else:
        raise ValueError(
            "Unsupported embeddings file type. Use .npz, .csv, or .parquet."
        )

    if JOIN_KEY not in emb_df.columns:
        raise ValueError(f"Embeddings table must include '{JOIN_KEY}' column.")

    emb_cols = [c for c in emb_df.columns if c != JOIN_KEY]
    if not emb_cols:
        raise ValueError("Embeddings table has no feature columns.")

    for col in emb_cols:
        emb_df[col] = pd.to_numeric(emb_df[col], errors="coerce")

    emb_df = emb_df.dropna(subset=emb_cols)
    return emb_df[[JOIN_KEY, *emb_cols]]


def _train_model(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }


def _train_tabular_part2(
    X_tab: pd.DataFrame, y: pd.Series, include_xgboost: bool
) -> tuple[dict, list[dict]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X_tab, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {}
    importance_rows = []

    gb = GradientBoostingClassifier(random_state=42)
    models["gradient_boosting"] = _train_model(gb, X_train, y_train, X_test, y_test)
    for f_name, importance in zip(X_tab.columns, gb.feature_importances_):
        importance_rows.append(
            {
                "model": "gradient_boosting",
                "feature": f_name,
                "importance": float(importance),
            }
        )

    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
            )
            models["xgboost"] = _train_model(xgb, X_train, y_train, X_test, y_test)
            for f_name, importance in zip(X_tab.columns, xgb.feature_importances_):
                importance_rows.append(
                    {
                        "model": "xgboost",
                        "feature": f_name,
                        "importance": float(importance),
                    }
                )
        except Exception as exc:
            note = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            models["xgboost"] = {
                "status": "wip",
                "note": (
                    "XGBoost unavailable in this environment; install/repair xgboost and OpenMP "
                    f"(macOS: `brew install libomp`). Details: {note}"
                ),
            }
    else:
        models["xgboost"] = {
            "status": "wip",
            "note": "XGBoost is optional and currently disabled. Re-run with --include-xgboost to try it.",
        }

    return models, importance_rows


def _train_multimodal(
    merged: pd.DataFrame, embedding_cols: list[str], output: dict
) -> dict:
    X_tab = merged[TABULAR_FEATURES].copy()
    X_emb = merged[embedding_cols].copy()
    y = merged["Severe"].copy()
    X_multi = pd.concat([X_emb, X_tab], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    output["combined_random_forest"] = _train_model(clf, X_train, y_train, X_test, y_test)
    return output


def _save_importance_chart(importance_rows: list[dict], output_dir: Path) -> None:
    if not importance_rows:
        return

    imp_df = pd.DataFrame(importance_rows)
    imp_df.to_csv(output_dir / "tabular_feature_importance.csv", index=False)

    pivot = (
        imp_df.pivot(index="feature", columns="model", values="importance")
        .fillna(0.0)
        .sort_index()
    )
    ax = pivot.plot(kind="bar", figsize=(6, 4))
    ax.set_title("Tabular model feature importance")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_dir / "tabular_feature_importance.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Midpoint multimodal experiment: tabular part 2 + early combined model "
            "(image embeddings + age/sex)."
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
        "--embeddings",
        type=str,
        default="",
        help="Optional embeddings file (.npz, .csv, .parquet) with path_to_image key",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="midpoint/results",
        help="Output directory for metrics and charts",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Enable XGBoost training for tabular part 2 (optional; may require OpenMP setup).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    labels_path = Path(args.labels_jsonl)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels JSONL not found: {labels_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = _load_demographics_and_labels(csv_path, labels_path)
    X_tab = merged[TABULAR_FEATURES].copy()
    y = merged["Severe"].copy()

    report = {
        "n_samples": int(len(merged)),
        "class_counts": y.value_counts().to_dict(),
        "class_distribution": y.value_counts(normalize=True).to_dict(),
        "tabular_part2": {},
        "multimodal_early": {},
    }

    tab_models, importance_rows = _train_tabular_part2(
        X_tab, y, include_xgboost=args.include_xgboost
    )
    report["tabular_part2"]["models"] = tab_models
    _save_importance_chart(importance_rows, output_dir)

    embeddings_arg = args.embeddings.strip()
    if embeddings_arg:
        embeddings_path = Path(embeddings_arg)
        if not embeddings_path.exists():
            report["multimodal_early"] = {
                "status": "wip",
                "note": f"Embeddings file not found: {embeddings_path}",
            }
        else:
            emb_df = _load_embeddings(embeddings_path)
            combined = merged.merge(emb_df, on=JOIN_KEY, how="inner")
            embedding_cols = [c for c in combined.columns if c.startswith("emb_")]
            if not embedding_cols:
                embedding_cols = [c for c in emb_df.columns if c != JOIN_KEY]
            if len(combined) < 10:
                report["multimodal_early"] = {
                    "status": "wip",
                    "note": "Too few merged rows after joining embeddings for a reliable split.",
                    "n_merged_samples": int(len(combined)),
                }
            else:
                report["multimodal_early"] = _train_multimodal(
                    combined, embedding_cols, {}
                )
                report["multimodal_early"]["status"] = "preliminary"
                report["multimodal_early"]["n_merged_samples"] = int(len(combined))
    else:
        report["multimodal_early"] = {
            "status": "wip",
            "note": "No --embeddings file provided. Skipping early multimodal run.",
        }

    report_path = output_dir / "multimodal_metrics.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    with (output_dir / "multimodal_metrics.txt").open("w") as f:
        f.write("=== Tabular Models - Part 2 ===\n")
        for model_name, metrics in report["tabular_part2"]["models"].items():
            f.write(f"\n{model_name}:\n")
            if isinstance(metrics, dict) and "accuracy" in metrics:
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  F1: {metrics['f1']:.4f}\n")
            else:
                f.write(f"  {metrics}\n")

        f.write("\n=== Early Multimodal ===\n")
        multi = report["multimodal_early"]
        if "combined_random_forest" in multi:
            f.write(
                f"combined_random_forest Accuracy: {multi['combined_random_forest']['accuracy']:.4f}\n"
            )
            f.write(f"combined_random_forest F1: {multi['combined_random_forest']['f1']:.4f}\n")
            f.write(f"Status: {multi.get('status', 'preliminary')}\n")
        else:
            f.write(f"Status: {multi.get('status', 'wip')}\n")
            f.write(f"Note: {multi.get('note', 'No details')}\n")

    print(f"Wrote report: {report_path}")
    print(f"Wrote text summary: {output_dir / 'multimodal_metrics.txt'}")
    print(f"Wrote feature importance chart: {output_dir / 'tabular_feature_importance.png'}")


if __name__ == "__main__":
    main()

