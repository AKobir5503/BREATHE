"""Synthetic tests for imaging CV + hold-out logic (no images / torch required for this path)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_RUN_PATH = ROOT / "midpoint" / "imaging" / "run.py"


@pytest.fixture(scope="module")
def imaging_run():
    spec = importlib.util.spec_from_file_location("imaging_run", _RUN_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_evaluate_image_only_synthetic(imaging_run):
    np.random.seed(42)
    n, d = 120, 16
    emb_cols = [f"emb_{i}" for i in range(d)]
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(X, columns=emb_cols)
    df[imaging_run.JOIN_KEY] = [f"path_{i}.jpg" for i in range(n)]
    df["Severe"] = y.astype(int)

    summary, y_test, preds, probs, best_name = imaging_run._evaluate_image_only(
        df, include_xgboost=False, n_cv_splits=3
    )

    assert summary["n_samples"] == n
    assert summary["best_model_for_plots"] == best_name
    assert len(y_test) == summary["test_size_final_split"]
    assert len(preds) == len(y_test)
    assert "cv_f1_mean" in summary["models"]["logistic_regression"]
    assert "final_holdout_metrics" in summary
    if probs is not None:
        assert probs.shape == y_test.shape
