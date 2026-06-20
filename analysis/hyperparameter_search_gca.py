#!/usr/bin/env python3
"""
Hyperparameter grid search for GCA RM training.
Tests different learning rates and epochs to find optimal GCA configuration.
"""

import os
import json
import subprocess
from pathlib import Path
from itertools import product
from typing import Dict, List

def run_training(
    holistic_prefs: str,
    gca_prefs: str,
    output_dir: str,
    lr: float,
    epochs: int,
    batch_size: int = 8,
) -> Dict:
    """Run a single RM training and return metrics."""
    
    cmd = [
        "python3", "src/reward_model/run_training.py",
        "--holistic", holistic_prefs,
        "--gca", gca_prefs,
        "--output-dir", output_dir,
        "--kfold", "5",
        "--backbone", "FacebookAI/roberta-base",
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--max-length", "512",
    ]
    
    print(f"  Running: lr={lr:.0e}, epochs={epochs}, batch_size={batch_size}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[:200]}")
        return None
    
    # Load and return metrics
    summary_path = Path(output_dir) / "rm_training_summary.json"
    if not summary_path.exists():
        print(f"    ERROR: No summary file at {summary_path}")
        return None
    
    with open(summary_path) as f:
        metrics = json.load(f)
    
    gca_acc = metrics.get("gca", {}).get("mean_accuracy", 0.0)
    holistic_acc = metrics.get("holistic", {}).get("mean_accuracy", 0.0)
    print(f"    ✓ GCA={gca_acc:.4f}, Holistic={holistic_acc:.4f}")
    
    return {
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "gca_accuracy": gca_acc,
        "holistic_accuracy": holistic_acc,
        "output_dir": output_dir,
    }

def main():
    # Preferences (from optimized alpha=0.0 run)
    holistic_prefs = "data/preferences_1000_opt_alpha_0.0/holistic_reward_preferences_1000_opt_alpha_0.0.jsonl"
    gca_prefs = "data/preferences_1000_opt_alpha_0.0/gca_reward_preferences_1000_opt_alpha_0.0.jsonl"
    
    # Check if files exist
    if not Path(holistic_prefs).exists() or not Path(gca_prefs).exists():
        print("ERROR: Optimized preference files not found!")
        print("Run validate_gca_optimization.sh first")
        return
    
    # Hyperparameter grid
    learning_rates = [1e-5, 2e-5, 5e-5]
    epochs_list = [3, 5, 7]
    batch_size = 8  # Keep constant for stability
    
    print("=" * 70)
    print("GCA RM Hyperparameter Search")
    print("=" * 70)
    print(f"Learning rates: {learning_rates}")
    print(f"Epochs: {epochs_list}")
    print(f"Batch size: {batch_size}")
    print(f"Total experiments: {len(learning_rates) * len(epochs_list)}")
    print("")
    
    results = []
    
    for i, (lr, epochs) in enumerate(product(learning_rates, epochs_list), 1):
        output_dir = f"outputs/reward_models_hpsearch_lr{lr:.0e}_ep{epochs}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Experiment {i}/{len(learning_rates)*len(epochs_list)}:")
        metrics = run_training(holistic_prefs, gca_prefs, output_dir, lr, epochs, batch_size)
        
        if metrics:
            results.append(metrics)
    
    if not results:
        print("ERROR: No successful experiments!")
        return
    
    # Analyze results
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 70)
    
    # Sort by GCA accuracy descending
    results_sorted = sorted(results, key=lambda x: -x["gca_accuracy"])
    
    print("\nRanked by GCA Accuracy:")
    print("-" * 70)
    print(f"{'Rank':<4} {'LR':<8} {'Epochs':<6} {'GCA Acc':<10} {'Hol Acc':<10} {'Gap':<10}")
    print("-" * 70)
    
    for rank, result in enumerate(results_sorted, 1):
        lr = result["lr"]
        epochs = result["epochs"]
        gca_acc = result["gca_accuracy"]
        hol_acc = result["holistic_accuracy"]
        gap = hol_acc - gca_acc
        
        marker = "⭐" if rank == 1 else "  "
        print(f"{marker} {rank:<2} {lr:.0e}   {epochs:<6} {gca_acc:.4f}      {hol_acc:.4f}      {gap:+.4f}")
    
    # Identify best configuration
    best = results_sorted[0]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION:")
    print(f"  Learning Rate: {best['lr']:.0e}")
    print(f"  Epochs: {best['epochs']}")
    print(f"  GCA Accuracy: {best['gca_accuracy']:.4f}")
    print(f"  Holistic Accuracy: {best['holistic_accuracy']:.4f}")
    print(f"  Gap: {best['holistic_accuracy'] - best['gca_accuracy']:+.4f}")
    print(f"  Model saved at: {best['output_dir']}")
    print("=" * 70)
    
    # Save results
    results_file = "outputs/hyperparameter_search_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "experiment": "gca_rm_hyperparameter_search",
            "preferences": "alpha=0.0 (optimized)",
            "results": results_sorted,
            "best_config": best,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
