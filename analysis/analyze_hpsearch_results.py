#!/usr/bin/env python3
"""
Comprehensive hyperparameter search results analyzer.
Compares 9 configurations and identifies the best GCA training parameters.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import statistics


def load_summary(output_dir: str) -> Dict:
    """Load RM training summary from directory."""
    summary_path = Path(output_dir) / "rm_training_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def extract_config_from_dir(output_dir: str) -> Tuple[str, str]:
    """Extract learning rate and epochs from directory name."""
    # Format: outputs/reward_models_hpsearch_lr1e-05_ep5
    parts = output_dir.split("lr")
    if len(parts) < 2:
        return None, None
    
    lr_part = parts[1].split("_")[0]
    ep_part = parts[1].split("ep")[-1]
    
    # Normalize LR format
    if lr_part == "1e-05":
        lr = "1e-5"
    elif lr_part == "2e-05":
        lr = "2e-5"
    elif lr_part == "5e-05":
        lr = "5e-5"
    else:
        lr = lr_part
    
    return lr, ep_part


def main():
    print("\n" + "=" * 90)
    print("GCA HYPERPARAMETER SEARCH RESULTS ANALYSIS".center(90))
    print("=" * 90)
    
    # Find all hpsearch directories
    hpsearch_dirs = []
    if Path("outputs").exists():
        for entry in Path("outputs").iterdir():
            if entry.is_dir() and "hpsearch" in entry.name:
                hpsearch_dirs.append(str(entry))
    
    if not hpsearch_dirs:
        print("\nERROR: No hyperparameter search results found!")
        print("Run: sbatch slurm/submit_gca_hpsearch.sh")
        return 1
    
    # Load results for each configuration
    results = []
    for output_dir in sorted(hpsearch_dirs):
        summary = load_summary(output_dir)
        if not summary:
            print(f"Skipping {output_dir} - no summary found")
            continue
        
        lr, epochs = extract_config_from_dir(output_dir)
        if not lr or not epochs:
            continue
        
        gca_acc = summary.get("gca", {}).get("mean_accuracy", 0.0)
        holistic_acc = summary.get("holistic", {}).get("mean_accuracy", 0.0)
        gca_folds = summary.get("gca", {}).get("fold_accuracies", [])
        holistic_folds = summary.get("holistic", {}).get("fold_accuracies", [])
        
        results.append({
            "lr": lr,
            "epochs": epochs,
            "output_dir": output_dir,
            "gca_accuracy": gca_acc,
            "holistic_accuracy": holistic_acc,
            "gap": holistic_acc - gca_acc,
            "gca_std": statistics.stdev(gca_folds) if len(gca_folds) > 1 else 0.0,
            "gca_folds": gca_folds,
            "holistic_folds": holistic_folds,
        })
    
    if not results:
        print("\nERROR: No valid results found!")
        return 1
    
    print(f"\nFound {len(results)} configurations:")
    print("-" * 90)
    
    # Sort by GCA accuracy descending
    results_sorted = sorted(results, key=lambda x: -x["gca_accuracy"])
    
    # Print table
    print(f"{'Rank':<4} {'LR':<8} {'Epochs':<6} {'GCA Acc':<10} {'Hol Acc':<10} {'Gap':<10} {'Std Dev':<10}")
    print("-" * 90)
    
    for rank, result in enumerate(results_sorted, 1):
        print(f"{rank:<4} {result['lr']:<8} {result['epochs']:<6} "
              f"{result['gca_accuracy']:.4f}      {result['holistic_accuracy']:.4f}      "
              f"{result['gap']:+.4f}      {result['gca_std']:.4f}")
    
    # Identify best configuration
    best = results_sorted[0]
    
    print("\n" + "=" * 90)
    print("BEST CONFIGURATION".center(90))
    print("=" * 90)
    print(f"\nHyperparameters:")
    print(f"  Learning Rate: {best['lr']}")
    print(f"  Epochs: {best['epochs']}")
    print(f"\nPerformance:")
    print(f"  GCA Accuracy: {best['gca_accuracy']:.4f}")
    print(f"  Holistic Accuracy: {best['holistic_accuracy']:.4f}")
    print(f"  Gap: {best['gap']:+.4f}")
    print(f"  Std Dev: {best['gca_std']:.4f}")
    print(f"\nFold-wise GCA Accuracies:")
    for i, fold_acc in enumerate(best['gca_folds'], 1):
        print(f"  Fold {i}: {fold_acc:.4f}")
    
    print(f"\nModel saved at: {best['output_dir']}")
    
    # Comparison with main optimization baseline
    baseline_gca = 0.5600  # From alpha=0.5 run
    improvement = best['gca_accuracy'] - baseline_gca
    
    print("\n" + "=" * 90)
    print("IMPROVEMENT ANALYSIS".center(90))
    print("=" * 90)
    print(f"\nBaseline (alpha=0.5, lr=2e-5, epochs=5): {baseline_gca:.4f}")
    print(f"Best config (alpha=0.0, lr={best['lr']}, epochs={best['epochs']}): {best['gca_accuracy']:.4f}")
    print(f"Absolute improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if best['gca_accuracy'] > best['holistic_accuracy']:
        print(f"\n✅ VICTORY: GCA outperforms Holistic by {(best['gca_accuracy'] - best['holistic_accuracy'])*100:.2f}%")
    else:
        print(f"\n⚠ Gap to Holistic: {abs(best['gap'])*100:.2f}%")
        print(f"   Still need: {(best['holistic_accuracy'] - best['gca_accuracy'])*100:.2f}% improvement")
    
    # Save results summary
    summary_output = {
        "experiment": "gca_hyperparameter_search",
        "base_preferences": "alpha=0.0 (optimized)",
        "num_configurations": len(results),
        "best_config": {
            "lr": best['lr'],
            "epochs": best['epochs'],
            "gca_accuracy": best['gca_accuracy'],
            "holistic_accuracy": best['holistic_accuracy'],
            "gap": best['gap'],
        },
        "all_results": results_sorted,
    }
    
    output_file = "outputs/hpsearch_results_summary.json"
    with open(output_file, "w") as f:
        json.dump(summary_output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 90 + "\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
