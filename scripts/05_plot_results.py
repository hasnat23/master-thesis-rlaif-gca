#!/usr/bin/env python3
"""
Script 05: Generate plots from evaluation metrics.

Reads saved metrics JSON and creates:
1. Bar chart comparing ROUGE scores (Summary A vs Summary B)
2. Full metrics comparison bar chart (ROUGE + BERTScore)

Usage:
    python scripts/05_plot_results.py --metrics outputs/metrics/baseline_metrics_*.json
    python scripts/05_plot_results.py --metrics outputs/metrics/baseline_metrics_*.json --output-dir outputs/plots
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import PROJECT_ROOT

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_rouge_comparison(metrics: dict, output_path: str):
    """Bar chart comparing ROUGE scores between Summary A and Summary B."""
    rouge_keys = ["rouge1", "rouge2", "rougeL"]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

    a_vals = [metrics["summary_a_metrics"].get(k, 0) for k in rouge_keys]
    b_vals = [metrics["summary_b_metrics"].get(k, 0) for k in rouge_keys]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_a = ax.bar([i - width/2 for i in x], a_vals, width, label="Summary A (low temp)", color="#4C72B0")
    bars_b = ax.bar([i + width/2 for i in x], b_vals, width, label="Summary B (high temp)", color="#DD8452")

    ax.set_ylabel("F1 Score")
    ax.set_title(f"Baseline ROUGE Scores (n={metrics.get('n_samples', '?')})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(max(a_vals), max(b_vals)) * 1.2 if max(a_vals + b_vals) > 0 else 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Value labels on bars
    for bar in bars_a + bars_b:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_metrics(metrics: dict, output_path: str):
    """Full comparison bar chart including BERTScore."""
    all_keys = sorted(set(
        list(metrics["summary_a_metrics"].keys()) +
        list(metrics["summary_b_metrics"].keys())
    ))

    labels = [k.replace("_", " ").title() for k in all_keys]
    a_vals = [metrics["summary_a_metrics"].get(k, 0) for k in all_keys]
    b_vals = [metrics["summary_b_metrics"].get(k, 0) for k in all_keys]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width/2 for i in x], a_vals, width, label="Summary A (low temp)", color="#4C72B0")
    ax.bar([i + width/2 for i in x], b_vals, width, label="Summary B (high temp)", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title(f"Baseline Metrics Comparison (n={metrics.get('n_samples', '?')})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics JSON file")
    parser.add_argument("--output-dir", type=str, default="outputs/plots")
    args = parser.parse_args()

    metrics_path = args.metrics
    if not Path(metrics_path).is_absolute():
        metrics_path = str(PROJECT_ROOT / metrics_path)

    if not Path(metrics_path).exists():
        print(f"Error: {metrics_path} not found")
        sys.exit(1)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    output_dir = Path(PROJECT_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_rouge_comparison(metrics, str(output_dir / "rouge_comparison.png"))

    if any(k.startswith("bertscore") for k in metrics.get("summary_a_metrics", {})):
        plot_all_metrics(metrics, str(output_dir / "all_metrics_comparison.png"))

    print("Plotting complete.")


if __name__ == "__main__":
    main()
