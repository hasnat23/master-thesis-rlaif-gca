#!/usr/bin/env python3
"""
Bootstrap confidence intervals and Wilcoxon signed-rank tests for DPO evaluation.

Reads generations.jsonl (per-sample text) and eval_results.json (per-sample AlignScore),
recomputes per-sample ROUGE-1/2/L, then runs:
  - 95% bootstrap CI (10 000 resamples) for each metric × condition
  - Wilcoxon signed-rank test (vs baseline) for each metric × DPO condition

Usage:
    python src/eval/bootstrap_ci.py \
        --generations outputs/eval/generations.jsonl \
        --eval-results outputs/eval/eval_results.json \
        --output-dir outputs/eval \
        [--n-bootstrap 10000] [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer as rouge_lib
from scipy.stats import wilcoxon


METRICS = ["rouge1", "rouge2", "rougeL", "alignscore"]
CONDITIONS = ["baseline", "dpo_holistic", "dpo_gca"]


# ---------------------------------------------------------------------------
# Per-sample ROUGE
# ---------------------------------------------------------------------------

def _per_sample_rouge(preds: list[str], refs: list[str]) -> dict[str, list[float]]:
    scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {"rouge1": r1, "rouge2": r2, "rougeL": rl}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) using percentile bootstrap."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations",   required=True,
                        help="Path to outputs/eval/generations.jsonl")
    parser.add_argument("--eval-results",  required=True,
                        help="Path to outputs/eval/eval_results.json")
    parser.add_argument("--output-dir",    default="outputs/eval")
    parser.add_argument("--n-bootstrap",   type=int, default=10_000)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load generations --------------------------------------------------
    rows = []
    with open(args.generations) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    print(f"Loaded {n} samples from {args.generations}")

    references = [r["reference"] for r in rows]
    summaries = {
        cond: [r[f"summary_{cond}"] for r in rows]
        for cond in CONDITIONS
    }

    # -- Compute per-sample ROUGE for each condition -----------------------
    per_sample: dict[str, dict[str, list[float]]] = {}
    for cond in CONDITIONS:
        rouge = _per_sample_rouge(summaries[cond], references)
        per_sample[cond] = rouge

    # -- Load per-sample AlignScore from eval_results.json -----------------
    with open(args.eval_results) as f:
        eval_results = json.load(f)

    for cond in CONDITIONS:
        as_scores = eval_results["conditions"][cond].get("alignscore_per_sample")
        if as_scores and len(as_scores) == n:
            per_sample[cond]["alignscore"] = as_scores
        else:
            print(f"  WARNING: no per-sample alignscore for {cond}, skipping")
            per_sample[cond]["alignscore"] = None

    # -- Bootstrap CIs -----------------------------------------------------
    ci_results: dict[str, dict[str, dict]] = {}
    for cond in CONDITIONS:
        ci_results[cond] = {}
        for metric in METRICS:
            vals = per_sample[cond].get(metric)
            if vals is None:
                continue
            mean, lo, hi = _bootstrap_ci(vals, n_bootstrap=args.n_bootstrap,
                                         seed=args.seed)
            ci_results[cond][metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

    # -- Wilcoxon signed-rank tests ----------------------------------------
    wilcoxon_results: dict[str, dict[str, dict]] = {}
    for cond in ["dpo_holistic", "dpo_gca"]:
        wilcoxon_results[cond] = {}
        for metric in METRICS:
            baseline_vals = per_sample["baseline"].get(metric)
            cond_vals = per_sample[cond].get(metric)
            if baseline_vals is None or cond_vals is None:
                continue
            try:
                stat, p = wilcoxon(cond_vals, baseline_vals, alternative="two-sided")
                wilcoxon_results[cond][metric] = {"statistic": stat, "p_value": p}
            except Exception as e:
                wilcoxon_results[cond][metric] = {"error": str(e)}

    # -- Print results table -----------------------------------------------
    print("\n" + "=" * 80)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS  (n={}, B={:,})".format(
        n, args.n_bootstrap))
    print("=" * 80)
    header = f"{'Condition':<16} {'Metric':<12} {'Mean':>8}  {'95% CI':>18}  {'Δ vs baseline':>14}"
    print(header)
    print("-" * 80)
    for cond in CONDITIONS:
        for metric in METRICS:
            r = ci_results[cond].get(metric)
            if r is None:
                continue
            base_r = ci_results["baseline"].get(metric)
            delta = f"{r['mean'] - base_r['mean']:+.4f}" if (base_r and cond != "baseline") else "  —    "
            print(f"{cond:<16} {metric:<12} {r['mean']:>8.4f}  "
                  f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  {delta:>14}")
        print()

    print("\n" + "=" * 80)
    print("WILCOXON SIGNED-RANK TEST vs BASELINE  (two-sided)")
    print("=" * 80)
    print(f"{'Condition':<16} {'Metric':<12} {'p-value':>10}  {'Significant (p<0.05)':>22}")
    print("-" * 80)
    for cond in ["dpo_holistic", "dpo_gca"]:
        for metric in METRICS:
            r = wilcoxon_results[cond].get(metric)
            if r is None:
                continue
            if "error" in r:
                print(f"{cond:<16} {metric:<12} {'ERROR':>10}  {r['error']}")
                continue
            sig = "YES *" if r["p_value"] < 0.05 else "no"
            print(f"{cond:<16} {metric:<12} {r['p_value']:>10.4f}  {sig:>22}")
        print()

    # -- Save JSON output --------------------------------------------------
    output = {
        "n_samples": n,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_ci": ci_results,
        "wilcoxon": wilcoxon_results,
    }
    out_path = output_dir / "bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
