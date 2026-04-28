#!/usr/bin/env python3
"""Merge multiple imaging_metrics.json runs into one CSV + optional bar chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge imaging_metrics.json files for backbone comparison.")
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="Output directories each containing imaging_metrics.json",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="midpoint/results/imaging/imaging_architecture_comparison.csv",
        help="Path to write merged comparison CSV",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="",
        help="Optional path to save CV F1 bar chart (e.g. imaging_cv_f1_by_extractor.png)",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for d in args.dirs:
        p = Path(d) / "imaging_metrics.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing metrics file: {p}")
        with p.open() as f:
            data = json.load(f)
        extractor = data.get("extractor", Path(d).name)
        best = data.get("best_model_for_plots", "")
        models = data.get("models", {})
        bm = models.get(best, {}) if best else {}
        row = {
            "source_dir": str(Path(d).resolve()),
            "extractor": extractor,
            "best_classifier": best,
            "n_samples": data.get("n_samples"),
            "cv_splits": data.get("cv_splits"),
            "cv_f1_mean": bm.get("cv_f1_mean"),
            "cv_f1_std": bm.get("cv_f1_std"),
            "cv_accuracy_mean": bm.get("cv_accuracy_mean"),
            "cv_accuracy_std": bm.get("cv_accuracy_std"),
            "holdout_f1": (data.get("final_holdout_metrics") or {}).get("f1"),
            "holdout_accuracy": (data.get("final_holdout_metrics") or {}).get("accuracy"),
        }
        rows.append(row)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    if args.output_plot.strip():
        plot_path = Path(args.output_plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(df))
        means = df["cv_f1_mean"].fillna(0).to_numpy()
        stds = df["cv_f1_std"].fillna(0).to_numpy()
        labels = [f"{r['extractor']}\n{r['best_classifier']}" for _, r in df.iterrows()]
        ax.bar(x, means, yerr=stds, capsize=4, color="teal", alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Mean CV F1 (best classifier per backbone)")
        ax.set_ylim(0, 1.05)
        ax.set_title("Image-only: backbone comparison")
        plt.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
