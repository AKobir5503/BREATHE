from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

SEVERE_LABELS = ["Pneumonia", "Edema", "Consolidation", "Pleural Effusion"]
TABULAR_FEATURES = ["age", "sex"]
JOIN_KEY = "path_to_image"

CV_SPLITS = 5
CV_RANDOM_STATE = 42


def load_demographics_and_labels_with_stats(
    csv_path: Path, labels_jsonl_path: Path
) -> tuple[pd.DataFrame, dict]:
    """Load merged demographics + labels and record row counts at each cleaning step."""
    demographics = pd.read_csv(csv_path)
    labels = pd.read_json(labels_jsonl_path, lines=True)

    missing_demo = [c for c in [JOIN_KEY, *TABULAR_FEATURES] if c not in demographics.columns]
    if missing_demo:
        raise ValueError(f"Demographics CSV missing columns: {missing_demo}")

    missing_labels = [c for c in [JOIN_KEY, *SEVERE_LABELS] if c not in labels.columns]
    if missing_labels:
        raise ValueError(f"Labels JSONL missing columns: {missing_labels}")

    stats: dict = {
        "original_rows_demographics": int(len(demographics)),
        "original_rows_labels": int(len(labels)),
    }

    demographics = demographics[[JOIN_KEY, *TABULAR_FEATURES]].copy()
    labels = labels[[JOIN_KEY, *SEVERE_LABELS]].copy()

    merged = demographics.merge(labels, on=JOIN_KEY, how="inner")
    stats["after_merge"] = int(len(merged))

    merged = merged.dropna(subset=[*TABULAR_FEATURES, *SEVERE_LABELS])
    stats["after_removing_missing"] = int(len(merged))

    for col in SEVERE_LABELS:
        merged = merged[merged[col] != -1]
    stats["after_removing_uncertain"] = int(len(merged))

    merged["sex"] = (merged["sex"] == "Male").astype(int)
    merged["Severe"] = (merged[SEVERE_LABELS].sum(axis=1) > 0).astype(int)
    merged["age"] = pd.to_numeric(merged["age"], errors="coerce")
    merged = merged.dropna(subset=["age"])

    stats["final_usable_rows"] = int(len(merged))
    y_tmp = merged["Severe"]
    stats["severe_count"] = int(y_tmp.sum())
    stats["non_severe_count"] = int(len(y_tmp) - stats["severe_count"])

    out = merged[[JOIN_KEY, "age", "sex", "Severe"]].copy()
    return out, stats


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


def _serialize_rf_params(params: dict) -> dict:
    out = dict(params)
    if "max_depth" in out and out["max_depth"] is None:
        out["max_depth"] = None
    return out


def _to_json_safe(obj: object) -> object:
    """Convert numpy scalars and other non-JSON types for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _cv_mean_std(
    estimator, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> tuple[float, float, float, float]:
    """Return mean/std for accuracy and F1 across folds."""
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1": "f1"},
        n_jobs=-1,
    )
    acc = scores["test_accuracy"]
    f1 = scores["test_f1"]
    return (
        float(np.mean(acc)),
        float(np.std(acc)),
        float(np.mean(f1)),
        float(np.std(f1)),
    )


def _grid_cv_tabular_models(
    X_tab: pd.DataFrame, y: pd.Series, include_xgboost: bool
) -> tuple[dict, list[dict], StratifiedKFold, LogisticRegression]:
    """
    Stratified 5-fold CV + grid search per model family.
    Reports mean ± std for accuracy and F1 using the best estimator from grid search.
    """
    skf = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )
    models_out: dict = {}
    importance_rows: list[dict] = []

    # --- Logistic Regression ---
    lr_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    }
    lr_gs = GridSearchCV(
        LogisticRegression(max_iter=5000, solver="lbfgs"),
        lr_grid,
        cv=skf,
        scoring="f1",
        refit=True,
        n_jobs=-1,
    )
    lr_gs.fit(X_tab, y)
    lr_best = lr_gs.best_estimator_
    acc_m, acc_s, f1_m, f1_s = _cv_mean_std(lr_best, X_tab, y, skf)
    models_out["logistic_regression"] = {
        "best_params": _to_json_safe(lr_gs.best_params_),
        "accuracy_mean": acc_m,
        "accuracy_std": acc_s,
        "f1_mean": f1_m,
        "f1_std": f1_s,
        "grid_best_f1_cv": float(lr_gs.best_score_),
    }

    # --- Random Forest ---
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, None],
    }
    rf_gs = GridSearchCV(
        RandomForestClassifier(random_state=CV_RANDOM_STATE),
        rf_grid,
        cv=skf,
        scoring="f1",
        refit=True,
        n_jobs=-1,
    )
    rf_gs.fit(X_tab, y)
    acc_m, acc_s, f1_m, f1_s = _cv_mean_std(rf_gs.best_estimator_, X_tab, y, skf)
    models_out["random_forest"] = {
        "best_params": _to_json_safe(_serialize_rf_params(dict(rf_gs.best_params_))),
        "accuracy_mean": acc_m,
        "accuracy_std": acc_s,
        "f1_mean": f1_m,
        "f1_std": f1_s,
        "grid_best_f1_cv": float(rf_gs.best_score_),
    }
    for f_name, importance in zip(X_tab.columns, rf_gs.best_estimator_.feature_importances_):
        importance_rows.append(
            {"model": "random_forest", "feature": f_name, "importance": float(importance)}
        )

    # --- Gradient Boosting ---
    gb_grid = {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 200],
        "max_depth": [2, 3],
    }
    gb_gs = GridSearchCV(
        GradientBoostingClassifier(random_state=CV_RANDOM_STATE),
        gb_grid,
        cv=skf,
        scoring="f1",
        refit=True,
        n_jobs=-1,
    )
    gb_gs.fit(X_tab, y)
    acc_m, acc_s, f1_m, f1_s = _cv_mean_std(gb_gs.best_estimator_, X_tab, y, skf)
    models_out["gradient_boosting"] = {
        "best_params": _to_json_safe(gb_gs.best_params_),
        "accuracy_mean": acc_m,
        "accuracy_std": acc_s,
        "f1_mean": f1_m,
        "f1_std": f1_s,
        "grid_best_f1_cv": float(gb_gs.best_score_),
    }
    for f_name, importance in zip(X_tab.columns, gb_gs.best_estimator_.feature_importances_):
        importance_rows.append(
            {
                "model": "gradient_boosting",
                "feature": f_name,
                "importance": float(importance),
            }
        )

    # --- XGBoost ---
    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            xgb_grid = {
                "learning_rate": [0.01, 0.1],
                "n_estimators": [100, 200],
                "max_depth": [2, 3],
            }
            xgb_gs = GridSearchCV(
                XGBClassifier(
                    random_state=CV_RANDOM_STATE,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=-1,
                ),
                xgb_grid,
                cv=skf,
                scoring="f1",
                refit=True,
                n_jobs=-1,
            )
            xgb_gs.fit(X_tab, y)
            acc_m, acc_s, f1_m, f1_s = _cv_mean_std(
                xgb_gs.best_estimator_, X_tab, y, skf
            )
            models_out["xgboost"] = {
                "best_params": _to_json_safe(xgb_gs.best_params_),
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "f1_mean": f1_m,
                "f1_std": f1_s,
                "grid_best_f1_cv": float(xgb_gs.best_score_),
            }
            for f_name, importance in zip(
                X_tab.columns, xgb_gs.best_estimator_.feature_importances_
            ):
                importance_rows.append(
                    {"model": "xgboost", "feature": f_name, "importance": float(importance)}
                )
        except Exception as exc:
            note = (
                str(exc).strip().splitlines()[0]
                if str(exc).strip()
                else exc.__class__.__name__
            )
            models_out["xgboost"] = {
                "status": "wip",
                "note": (
                    "XGBoost unavailable in this environment; install/repair xgboost and OpenMP "
                    f"(macOS: `brew install libomp`). Details: {note}"
                ),
            }
    else:
        models_out["xgboost"] = {
            "status": "wip",
            "note": "XGBoost is optional and currently disabled. Re-run with --include-xgboost.",
        }

    return models_out, importance_rows, skf, lr_best


def save_age_sex_visualizations(
    X_tab: pd.DataFrame,
    y: pd.Series,
    logistic_clf: LogisticRegression,
    output_dir: Path,
) -> None:
    """Scatter (age vs sex by severity) + logistic decision boundary in age–sex space."""
    rng = np.random.default_rng(CV_RANDOM_STATE)
    sex_jitter = X_tab["sex"].to_numpy(dtype=float) + rng.normal(
        0, 0.02, size=len(X_tab)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    severe = y == 1
    ax.scatter(
        X_tab.loc[severe, "age"],
        sex_jitter[severe.to_numpy()],
        alpha=0.35,
        s=12,
        c="#c0392b",
        label="Severe",
        edgecolors="none",
    )
    ax.scatter(
        X_tab.loc[~severe, "age"],
        sex_jitter[~severe.to_numpy()],
        alpha=0.25,
        s=10,
        c="#2980b9",
        label="Non-severe",
        edgecolors="none",
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Sex (0 = not Male, 1 = Male; jittered for visibility)")
    ax.set_title("Age vs sex colored by severity label")
    ax.set_yticks([0, 1])
    ax.set_ylim(-0.35, 1.35)
    ax.legend(markerscale=2)
    plt.tight_layout()
    fig.savefig(output_dir / "tabular_age_sex_scatter_by_severity.png", dpi=150)
    plt.close(fig)

    clf = logistic_clf
    age_min, age_max = float(X_tab["age"].min()), float(X_tab["age"].max())
    pad = 0.02 * (age_max - age_min + 1e-6)
    ag = np.linspace(age_min - pad, age_max + pad, 200)
    sg = np.linspace(-0.05, 1.05, 200)
    AA, SS = np.meshgrid(ag, sg)
    grid = np.column_stack([AA.ravel(), SS.ravel()])
    Z = clf.predict_proba(grid)[:, 1].reshape(AA.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(AA, SS, Z, levels=20, cmap="RdYlBu_r", alpha=0.85)
    ax.contour(
        AA,
        SS,
        Z,
        levels=[0.5],
        colors="k",
        linewidths=1.5,
        linestyles="--",
    )
    fig.colorbar(cs, ax=ax, label="P(Severe)")
    ax.scatter(
        X_tab.loc[severe, "age"],
        sex_jitter[severe.to_numpy()],
        alpha=0.25,
        s=10,
        c="white",
        edgecolors="#333",
        linewidths=0.3,
        label="Severe",
    )
    ax.scatter(
        X_tab.loc[~severe, "age"],
        sex_jitter[~severe.to_numpy()],
        alpha=0.15,
        s=8,
        c="#222",
        edgecolors="none",
        label="Non-severe",
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Sex (0–1; jittered overlays)")
    ax.set_title("Logistic regression decision regions (Age × Sex)")
    ax.set_ylim(-0.35, 1.35)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "tabular_logistic_decision_boundary_age_sex.png", dpi=150)
    plt.close(fig)


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
    ax.set_title("Tabular tree models: feature importance (Age vs Sex)")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_dir / "tabular_feature_importance.png")
    plt.close()


def _train_multimodal(
    merged: pd.DataFrame, embedding_cols: list[str], output: dict
) -> dict:
    X_tab = merged[TABULAR_FEATURES].copy()
    X_emb = merged[embedding_cols].copy()
    y = merged["Severe"].copy()
    X_multi = pd.concat([X_emb, X_tab], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, y, test_size=0.2, random_state=CV_RANDOM_STATE, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, random_state=CV_RANDOM_STATE, class_weight="balanced"
    )
    output["combined_random_forest"] = _train_model(clf, X_train, y_train, X_test, y_test)
    return output


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

    merged, pipeline_stats = load_demographics_and_labels_with_stats(csv_path, labels_path)
    X_tab = merged[TABULAR_FEATURES].copy()
    y = merged["Severe"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_tab, y, test_size=0.2, random_state=CV_RANDOM_STATE, stratify=y
    )
    pipeline_stats["train_rows"] = int(len(X_train))
    pipeline_stats["test_rows"] = int(len(X_test))

    tab_models, importance_rows, skf, lr_best = _grid_cv_tabular_models(
        X_tab, y, include_xgboost=args.include_xgboost
    )

    report = {
        "n_samples": int(len(merged)),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "class_distribution": {
            str(k): float(v) for k, v in y.value_counts(normalize=True).sort_index().items()
        },
        "dataset_pipeline_stats": pipeline_stats,
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": CV_SPLITS,
            "shuffle": True,
            "random_state": CV_RANDOM_STATE,
            "hyperparameter_tuning": "GridSearchCV optimizing F1 on each fold",
        },
        "tabular_part2": {"models": tab_models},
        "multimodal_early": {},
    }

    _save_importance_chart(importance_rows, output_dir)
    save_age_sex_visualizations(X_tab, y, lr_best, output_dir)

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
        json.dump(_to_json_safe(report), f, indent=2)

    with (output_dir / "multimodal_metrics.txt").open("w") as f:
        f.write("=== Dataset pipeline (exact counts) ===\n")
        ds = pipeline_stats
        f.write(f"Original rows (demographics CSV): {ds['original_rows_demographics']}\n")
        f.write(f"Original rows (labels JSONL): {ds['original_rows_labels']}\n")
        f.write(f"After merge on path_to_image: {ds['after_merge']}\n")
        f.write(f"After removing missing (features + label columns): {ds['after_removing_missing']}\n")
        f.write(f"After removing uncertain (-1) labels: {ds['after_removing_uncertain']}\n")
        f.write(f"Final usable rows (after age numeric + drop NA age): {ds['final_usable_rows']}\n")
        f.write(f"Severe (1): {ds['severe_count']}\n")
        f.write(f"Non-severe (0): {ds['non_severe_count']}\n")
        f.write(
            f"Train / test rows (80/20 stratified holdout for reference): "
            f"{ds['train_rows']} / {ds['test_rows']}\n"
        )
        f.write(
            "\nPrimary evaluation: StratifiedKFold(5) + GridSearchCV (F1); "
            "metrics below are mean ± std across folds on the best estimator.\n\n"
        )

        f.write("=== Tabular models — Part 2 (5-fold CV + grid search) ===\n")
        for model_name, metrics in report["tabular_part2"]["models"].items():
            f.write(f"\n{model_name}:\n")
            if isinstance(metrics, dict) and "f1_mean" in metrics:
                f.write(f"  Best params: {metrics.get('best_params', {})}\n")
                f.write(
                    f"  F1: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}\n"
                )
                f.write(
                    f"  Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}\n"
                )
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

        f.write("\n=== Outputs ===\n")
        f.write("- tabular_age_sex_scatter_by_severity.png\n")
        f.write("- tabular_logistic_decision_boundary_age_sex.png\n")
        f.write("- tabular_feature_importance.png / tabular_feature_importance.csv\n")

    print(f"Wrote report: {report_path}")
    print(f"Wrote text summary: {output_dir / 'multimodal_metrics.txt'}")
    print(f"Wrote feature importance: {output_dir / 'tabular_feature_importance.png'}")
    print(f"Wrote plots: {output_dir / 'tabular_age_sex_scatter_by_severity.png'}, "
          f"{output_dir / 'tabular_logistic_decision_boundary_age_sex.png'}")


if __name__ == "__main__":
    main()
