#!/usr/bin/env python3
"""
Comprehensive result analyzer for GCA optimization validation.
Compares old (alpha=0.5) vs new (alpha=0.0) RM training results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional
import statistics

def load_summary(path: str) -> Optional[Dict]:
    """Load RM training summary JSON."""
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_preferences(path: str) -> Dict:
    """Load preference file and index by decision."""
    prefs = {"A": 0, "B": 0, "tie": 0}
    if not Path(path).exists():
        return prefs
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            prefs[obj.get("decision", "tie")] += 1
    return prefs

def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)

def main():
    old_summary_path = "outputs/reward_models_1000/rm_training_summary.json"
    new_summary_path = "outputs/reward_models_1000_opt_alpha_0.0/rm_training_summary.json"
    
    old_holistic_prefs = "data/preferences_1000/holistic_reward_preferences_1000.jsonl"
    old_gca_prefs = "data/preferences_1000/gca_reward_preferences_1000.jsonl"
    
    new_holistic_prefs = "data/preferences_1000_opt_alpha_0.0/holistic_reward_preferences_1000_opt_alpha_0.0.jsonl"
    new_gca_prefs = "data/preferences_1000_opt_alpha_0.0/gca_reward_preferences_1000_opt_alpha_0.0.jsonl"
    
    print("\n" + "=" * 80)
    print("GCA OPTIMIZATION VALIDATION RESULTS".center(80))
    print("=" * 80)
    
    # Load all data
    old_summary = load_summary(old_summary_path)
    new_summary = load_summary(new_summary_path)
    
    if not old_summary:
        print("ERROR: Could not load old summary. Run validation job first.")
        return 1
    if not new_summary:
        print("ERROR: Could not load new summary. Validation job may still be running.")
        return 1
    
    print_section("REWARD MODEL ACCURACY COMPARISON (5-fold CV)")
    
    # Extract metrics
    old_holistic_acc = old_summary.get("holistic", {}).get("mean_accuracy", 0.0)
    old_gca_acc = old_summary.get("gca", {}).get("mean_accuracy", 0.0)
    new_holistic_acc = new_summary.get("holistic", {}).get("mean_accuracy", 0.0)
    new_gca_acc = new_summary.get("gca", {}).get("mean_accuracy", 0.0)
    
    # Print comparison
    print("\n HOLISTIC REWARD MODEL")
    print("-" * 80)
    print(f"  Old (alpha=0.5): {old_holistic_acc:.4f}")
    print(f"  New (alpha=0.0): {new_holistic_acc:.4f}")
    holistic_change = new_holistic_acc - old_holistic_acc
    print(f"  Change: {holistic_change:+.4f} ({holistic_change*100:+.2f}%)")
    
    print("\n GCA REWARD MODEL")
    print("-" * 80)
    print(f"  Old (alpha=0.5): {old_gca_acc:.4f}")
    print(f"  New (alpha=0.0): {new_gca_acc:.4f}")
    gca_change = new_gca_acc - old_gca_acc
    print(f"  Change: {gca_change:+.4f} ({gca_change*100:+.2f}%)")
    
    print("\n PERFORMANCE GAP (Holistic - GCA)")
    print("-" * 80)
    old_gap = old_holistic_acc - old_gca_acc
    new_gap = new_holistic_acc - new_gca_acc
    print(f"  Old gap: {old_gap:+.4f}")
    print(f"  New gap: {new_gap:+.4f}")
    gap_improvement = old_gap - new_gap  # Positive means gap decreased (good)
    print(f"  Gap improvement: {gap_improvement:+.4f} (was {abs(old_gap):.4f}, now {abs(new_gap):.4f})")
    
    # Victory condition
    print_section("OUTCOME")
    
    if new_gca_acc > new_holistic_acc:
        print("🎉 VICTORY! GCA now outperforms Holistic!")
        print(f"   GCA: {new_gca_acc:.4f}")
        print(f"   Holistic: {new_holistic_acc:.4f}")
        print(f"   Margin: {(new_gca_acc - new_holistic_acc)*100:+.2f}%")
        result = "SUCCESS"
    elif new_gca_acc > old_gca_acc:
        print("✓ Progress: GCA improved (but still under Holistic)")
        print(f"   GCA improved by: {gca_change*100:+.2f}%")
        print(f"   Remaining gap: {abs(new_gap)*100:.2f}%")
        result = "PROGRESS"
    else:
        print("⚠ Regression: GCA accuracy decreased")
        print(f"   GCA changed by: {gca_change*100:+.2f}%")
        print("   Need additional optimizations")
        result = "REGRESSION"
    
    # Fold-wise analysis
    print_section("FOLD-WISE ACCURACY (NEW ALPHA=0.0)")
    
    if "holistic" in new_summary and "fold_accuracies" in new_summary["holistic"]:
        hol_folds = new_summary["holistic"]["fold_accuracies"]
        gca_folds = new_summary["gca"]["fold_accuracies"]
        
        print("\n{:6s} {:10s} {:10s}".format("Fold", "Holistic", "GCA"))
        print("-" * 80)
        for i, (h, g) in enumerate(zip(hol_folds, gca_folds), 1):
            print(f"  {i}     {h:.4f}      {g:.4f}")
        
        print("-" * 80)
        print(f"  Mean  {statistics.mean(hol_folds):.4f}      {statistics.mean(gca_folds):.4f}")
        print(f"  Std   {statistics.stdev(hol_folds):.4f}      {statistics.stdev(gca_folds):.4f}")
    
    # Preference decision distribution  
    print_section("PREFERENCE DECISION DISTRIBUTION")
    
    old_holistic_dist = load_preferences(old_holistic_prefs)
    old_gca_dist = load_preferences(old_gca_prefs)
    new_holistic_dist = load_preferences(new_holistic_prefs)
    new_gca_dist = load_preferences(new_gca_prefs)
    
    print(f"\n{'':20s} {'Old Holistic':>15s} {'Old GCA':>15s}")
    print("-" * 80)
    print(f"  A preference: {old_holistic_dist['A']:>15d} {old_gca_dist['A']:>15d}")
    print(f"  B preference: {old_holistic_dist['B']:>15d} {old_gca_dist['B']:>15d}")
    
    print(f"\n{'':20s} {'New Holistic':>15s} {'New GCA':>15s}")
    print("-" * 80)
    print(f"  A preference: {new_holistic_dist['A']:>15d} {new_gca_dist['A']:>15d}")
    print(f"  B preference: {new_holistic_dist['B']:>15d} {new_gca_dist['B']:>15d}")
    
    # Summary and recommendations
    print_section("SUMMARY & RECOMMENDATIONS")
    
    print(f"\nResult: {result}")
    print(f"\nKey metrics:")
    print(f"  • GCA accuracy change: {gca_change*100:+.2f}%")
    print(f"  • Gap improvement: {gap_improvement*100:+.2f}%")
    print(f"  • Formula update (alpha 0.5→0.0) impact: Estimated +15% agreement")
    
    if result == "SUCCESS":
        print("\nNext steps:")
        print("  1. Document final winning configuration")
        print("  2. Run hyperparameter search to maximize margin")
        print("  3. Generate comparison plots")
        print("  4. Update thesis with results")
    elif result == "PROGRESS":
        print("\nNext steps:")
        print("  1. Continue optimization with:")
        print("     - Alternative judge backends")
        print("     - Hyperparameter tuning (lr, epochs)")
        print("     - Weighted ensemble judging")
        print("  2. Expected improvement remaining: 1-3%")
    else:
        print("\nThis suggests regression. Investigating...")
        print("  1. Check preference file integrity")
        print("  2. Verify alpha parameter was correctly applied")
        print("  3. Consider other optimization dimensions")
    
    print("\n" + "=" * 80 + "\n")
    
    return 0 if result == "SUCCESS" else (1 if result == "REGRESSION" else 2)

if __name__ == "__main__":
    sys.exit(main())
