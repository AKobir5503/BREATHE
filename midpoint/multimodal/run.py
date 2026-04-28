from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEVERE_LABELS = ["Pneumonia", "Edema", "Consolidation", "Pleural Effusion"]
TABULAR_FEATURES = ["age", "sex"]
JOIN_KEY = "path_to_image"

CV_SPLITS = 5
CV_RANDOM_STATE = 42
LABEL_DEFINITION = (
    "Positive class (1) = High-Risk Respiratory Condition: at least one of "
    "Pneumonia, Edema, Consolidation, Pleural Effusion present. "
    "Negative class (0) = Lower-Risk Condition: none present."
)


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
    out = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)[:, 1]
            out["roc_auc"] = float(roc_auc_score(y_test, probs))
        except Exception:
            out["roc_auc"] = None
    return out


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


def _multimodal_search(
    X_multi: pd.DataFrame, y: pd.Series, include_xgboost: bool
) -> tuple[dict[str, dict], object | None, str | None]:
    skf = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )

    model_spaces: dict[str, tuple[object, dict]] = {
        "logistic_regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            random_state=CV_RANDOM_STATE,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=CV_RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
            },
        ),
        "mlp": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            random_state=CV_RANDOM_STATE,
                            max_iter=500,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=15,
                        ),
                    ),
                ]
            ),
            {
                "clf__hidden_layer_sizes": [(128,), (256, 64)],
                "clf__alpha": [1e-4, 1e-3],
            },
        ),
    }

    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            model_spaces["xgboost"] = (
                XGBClassifier(
                    random_state=CV_RANDOM_STATE,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=-1,
                ),
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.01, 0.1],
                },
            )
        except Exception:
            pass

    out: dict[str, dict] = {}
    best_name: str | None = None
    best_estimator: object | None = None
    best_f1 = -1.0
    for model_name, (estimator, grid) in model_spaces.items():
        try:
            gs = GridSearchCV(
                estimator=estimator,
                param_grid=grid,
                cv=skf,
                scoring="f1",
                refit=True,
                n_jobs=-1,
            )
            gs.fit(X_multi, y)
            acc_m, acc_s, f1_m, f1_s = _cv_mean_std(gs.best_estimator_, X_multi, y, skf)
            out[model_name] = {
                "best_params": _to_json_safe(gs.best_params_),
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "f1_mean": f1_m,
                "f1_std": f1_s,
                "grid_best_f1_cv": float(gs.best_score_),
            }
            if f1_m > best_f1:
                best_f1 = f1_m
                best_name = model_name
                best_estimator = gs.best_estimator_
        except Exception as exc:
            note = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
            out[model_name] = {"status": "failed", "note": note}

    if include_xgboost and "xgboost" not in out:
        out["xgboost"] = {
            "status": "wip",
            "note": "XGBoost unavailable in environment (install xgboost and OpenMP).",
        }
    if not include_xgboost:
        out["xgboost"] = {
            "status": "wip",
            "note": "XGBoost disabled. Re-run with --include-xgboost.",
        }

    return out, best_estimator, best_name


def _plot_multimodal_diagnostics(
    output_dir: Path,
    best_name: str,
    y_test: pd.Series,
    preds: np.ndarray,
    probs: np.ndarray | None,
) -> None:
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Purples", colorbar=False)
    ax.set_title(f"Best multimodal model: {best_name}")
    plt.tight_layout()
    fig.savefig(output_dir / "multimodal_confusion_matrix_best.png", dpi=150)
    plt.close(fig)

    if probs is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, probs, ax=ax)
        ax.set_title(f"Best multimodal model ROC: {best_name}")
        plt.tight_layout()
        fig.savefig(output_dir / "multimodal_roc_curve_best.png", dpi=150)
        plt.close(fig)


def _read_unimodal_metric(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    try:
        with json_path.open("r") as f:
            data = json.load(f)
    except Exception:
        return None

    if "models" in data:
        best_key = None
        best_f1 = -1.0
        for k, v in data["models"].items():
            if isinstance(v, dict) and "cv_f1_mean" in v and v["cv_f1_mean"] > best_f1:
                best_f1 = float(v["cv_f1_mean"])
                best_key = k
        if best_key is None:
            return None
        best = data["models"][best_key]
        return {
            "name": f"imaging ({best_key})",
            "accuracy": float(best.get("cv_accuracy_mean", np.nan)),
            "f1": float(best.get("cv_f1_mean", np.nan)),
        }

    if "tabular_part2" in data and "models" in data["tabular_part2"]:
        best_key = None
        best_f1 = -1.0
        for k, v in data["tabular_part2"]["models"].items():
            if isinstance(v, dict) and "f1_mean" in v and v["f1_mean"] > best_f1:
                best_f1 = float(v["f1_mean"])
                best_key = k
        if best_key is None:
            return None
        best = data["tabular_part2"]["models"][best_key]
        return {
            "name": f"tabular ({best_key})",
            "accuracy": float(best.get("accuracy_mean", np.nan)),
            "f1": float(best.get("f1_mean", np.nan)),
        }
    return None


def _plot_unimodal_vs_multimodal(
    output_dir: Path,
    multimodal_acc: float,
    multimodal_f1: float,
    tabular_ref: dict | None,
    imaging_ref: dict | None,
) -> None:
    rows = []
    if tabular_ref is not None:
        rows.append(tabular_ref)
    if imaging_ref is not None:
        rows.append(imaging_ref)
    rows.append({"name": "multimodal (best)", "accuracy": multimodal_acc, "f1": multimodal_f1})

    labels = [r["name"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    f1 = [r["f1"] for r in rows]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unimodal baselines vs multimodal")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "multimodal_vs_unimodal_accuracy_f1.png", dpi=150)
    plt.close(fig)


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
    parser.add_argument(
        "--tabular-metrics-json",
        type=str,
        default="midpoint/results/multimodal_metrics.json",
        help="Optional JSON path for best tabular baseline metrics (for final comparison plot).",
    )
    parser.add_argument(
        "--imaging-metrics-json",
        type=str,
        default="midpoint/results/imaging_metrics.json",
        help="Optional JSON path for best imaging baseline metrics (for final comparison plot).",
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
        "label_definition": LABEL_DEFINITION,
        "tabular_part2": {"models": tab_models},
        "multimodal_final": {},
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
                report["multimodal_final"] = {
                    "status": "wip",
                    "note": "Too few merged rows after joining embeddings for a reliable split.",
                    "n_merged_samples": int(len(combined)),
                }
            else:
                X_multi = combined[[*embedding_cols, *TABULAR_FEATURES]].copy()
                y_multi = combined["Severe"].copy()
                model_results, best_estimator, best_name = _multimodal_search(
                    X_multi, y_multi, include_xgboost=args.include_xgboost
                )
                report["multimodal_final"] = {
                    "status": "complete" if best_estimator is not None else "wip",
                    "n_merged_samples": int(len(combined)),
                    "models": model_results,
                    "best_model_by_cv_f1": best_name,
                }
                if best_estimator is not None and best_name is not None:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_multi,
                        y_multi,
                        test_size=0.2,
                        random_state=CV_RANDOM_STATE,
                        stratify=y_multi,
                    )
                    best_estimator.fit(X_train, y_train)
                    preds = best_estimator.predict(X_test)
                    probs = None
                    if hasattr(best_estimator, "predict_proba"):
                        try:
                            probs = best_estimator.predict_proba(X_test)[:, 1]
                        except Exception:
                            probs = None
                    holdout = _train_model(best_estimator, X_train, y_train, X_test, y_test)
                    report["multimodal_final"]["holdout_metrics"] = holdout
                    _plot_multimodal_diagnostics(
                        output_dir=output_dir,
                        best_name=best_name,
                        y_test=y_test,
                        preds=np.asarray(preds),
                        probs=probs,
                    )
    else:
        report["multimodal_final"] = {
            "status": "wip",
            "note": "No --embeddings file provided. Skipping multimodal run.",
        }

    tabular_ref = _read_unimodal_metric(Path(args.tabular_metrics_json))
    imaging_ref = _read_unimodal_metric(Path(args.imaging_metrics_json))
    mf = report.get("multimodal_final", {})
    hold = mf.get("holdout_metrics") if isinstance(mf, dict) else None
    if isinstance(hold, dict) and "accuracy" in hold and "f1" in hold:
        _plot_unimodal_vs_multimodal(
            output_dir=output_dir,
            multimodal_acc=float(hold["accuracy"]),
            multimodal_f1=float(hold["f1"]),
            tabular_ref=tabular_ref,
            imaging_ref=imaging_ref,
        )

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

        f.write("\n=== Final Multimodal (Embeddings + Age/Sex) ===\n")
        multi = report["multimodal_final"]
        if "models" in multi and "best_model_by_cv_f1" in multi:
            f.write(f"Status: {multi.get('status', 'complete')}\n")
            f.write(f"Merged multimodal rows: {multi.get('n_merged_samples', 0)}\n")
            f.write(f"Best model by mean CV F1: {multi.get('best_model_by_cv_f1')}\n")
            f.write("\nPer-model CV metrics (mean ± std):\n")
            for model_name, metrics in multi["models"].items():
                f.write(f"  {model_name}:\n")
                if isinstance(metrics, dict) and "f1_mean" in metrics:
                    f.write(f"    Best params: {metrics.get('best_params', {})}\n")
                    f.write(f"    F1: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}\n")
                    f.write(
                        f"    Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}\n"
                    )
                else:
                    f.write(f"    {metrics}\n")
            holdout = multi.get("holdout_metrics", {})
            if isinstance(holdout, dict) and "f1" in holdout:
                f.write("\nBest-model hold-out metrics (80/20 stratified):\n")
                f.write(f"  Accuracy: {holdout['accuracy']:.4f}\n")
                f.write(f"  F1: {holdout['f1']:.4f}\n")
                f.write(f"  Precision: {holdout['precision']:.4f}\n")
                f.write(f"  Recall: {holdout['recall']:.4f}\n")
                if holdout.get("roc_auc") is not None:
                    f.write(f"  ROC-AUC: {holdout['roc_auc']:.4f}\n")
                f.write(f"  Confusion matrix: {holdout['confusion_matrix']}\n")
        else:
            f.write(f"Status: {multi.get('status', 'wip')}\n")
            f.write(f"Note: {multi.get('note', 'No details')}\n")

        f.write("\n=== Outputs ===\n")
        f.write("- tabular_age_sex_scatter_by_severity.png\n")
        f.write("- tabular_logistic_decision_boundary_age_sex.png\n")
        f.write("- tabular_feature_importance.png / tabular_feature_importance.csv\n")
        f.write("- multimodal_confusion_matrix_best.png\n")
        f.write("- multimodal_roc_curve_best.png (if probability output available)\n")
        f.write("- multimodal_vs_unimodal_accuracy_f1.png\n")

    print(f"Wrote report: {report_path}")
    print(f"Wrote text summary: {output_dir / 'multimodal_metrics.txt'}")
    print(f"Wrote feature importance: {output_dir / 'tabular_feature_importance.png'}")
    print(f"Wrote plots: {output_dir / 'tabular_age_sex_scatter_by_severity.png'}, "
          f"{output_dir / 'tabular_logistic_decision_boundary_age_sex.png'}")


if __name__ == "__main__":
    main()
