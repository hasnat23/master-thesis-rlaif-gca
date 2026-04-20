#!/usr/bin/env python3
"""
Script 04: Evaluate baseline candidate summaries.

Loads cached candidate summaries, computes ROUGE and BERTScore
against reference summaries, and saves metrics to outputs/metrics/.

Usage:
    python scripts/04_evaluate_baseline.py --candidates data/candidates/candidates_200.jsonl
    python scripts/04_evaluate_baseline.py --candidates data/candidates/candidates_200.jsonl --no-bert
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_run_id, setup_logger, save_run_metadata
from src.data.schema import load_jsonl
from src.eval.metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline summaries")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates JSONL")
    parser.add_argument("--no-bert", action="store_true", help="Skip BERTScore (faster)")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics")
    args = parser.parse_args()

    run_id = get_run_id()
    logger = setup_logger("evaluate_baseline", run_id=run_id)

    candidates_path = str(PROJECT_ROOT / args.candidates)
    if not Path(candidates_path).exists():
        logger.error(f"File not found: {candidates_path}")
        sys.exit(1)

    pairs = load_jsonl(candidates_path)
    logger.info(f"Loaded {len(pairs)} candidate pairs from {candidates_path}")

    # Evaluate summary_a (low temperature) against references
    predictions_a = [p["summary_a"] for p in pairs]
    predictions_b = [p["summary_b"] for p in pairs]
    references = [p["reference_summary"] for p in pairs]

    logger.info("Computing metrics for Summary A (low_temp)...")
    metrics_a = compute_all_metrics(predictions_a, references, compute_bert=not args.no_bert)

    logger.info("Computing metrics for Summary B (high_temp)...")
    metrics_b = compute_all_metrics(predictions_b, references, compute_bert=not args.no_bert)

    results = {
        "n_samples": len(pairs),
        "summary_a_metrics": metrics_a,
        "summary_b_metrics": metrics_b,
    }

    # Print table
    logger.info("\n=== Baseline Evaluation Results ===")
    logger.info(f"{'Metric':<25} {'Summary A':>12} {'Summary B':>12}")
    logger.info("-" * 50)
    for key in sorted(set(list(metrics_a.keys()) + list(metrics_b.keys()))):
        val_a = metrics_a.get(key, 0.0)
        val_b = metrics_b.get(key, 0.0)
        logger.info(f"{key:<25} {val_a:>12.4f} {val_b:>12.4f}")

    # Save results
    output_dir = Path(PROJECT_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"baseline_metrics_{run_id}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nMetrics saved to {output_path}")

    save_run_metadata(
        run_id=run_id,
        script_name="04_evaluate_baseline",
        config={"candidates": args.candidates, "no_bert": args.no_bert},
        artifacts={"metrics": str(output_path)},
        metrics=results,
    )


if __name__ == "__main__":
    main()
