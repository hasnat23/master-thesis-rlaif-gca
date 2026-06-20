#!/usr/bin/env python3
"""
Test alternative GCA aggregation formulas to improve performance.
Hypothesis: the current alpha=0.5 formula is too harsh; alternatives may work better.
"""

import json
from pathlib import Path
import statistics

def load_preferences(jsonl_path: str):
    """Load preferences."""
    prefs = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            prefs[obj["sample_id"]] = obj
    return prefs

def aggregate_mean(scores):
    """Simple mean (no penalty)."""
    return sum(scores) / len(scores) if scores else 0.0

def aggregate_harmonic(scores):
    """Harmonic mean (moderate penalty for low scores)."""
    if not scores:
        return 0.0
    n = len(scores)
    return n / sum(1/s for s in scores if s > 0) if all(s > 0 for s in scores) else 0.0

def aggregate_geometric(scores):
    """Geometric mean (penalizes outliers moderately)."""
    if not scores:
        return 0.0
    prod = 1.0
    for s in scores:
        prod *= s
    return prod ** (1.0 / len(scores))

def aggregate_min(scores):
    """Minimum score (most aggressive penalty)."""
    return min(scores) if scores else 0.0

def aggregate_current(scores, alpha=0.5):
    """Current formula: mean * (min/mean)^alpha."""
    if not scores:
        return 0.0
    mean_score = sum(scores) / len(scores)
    if mean_score == 0.0:
        return 0.0
    min_score = min(scores)
    penalty = (min_score / mean_score) ** alpha
    return mean_score * penalty

def aggregate_alpha_variants(scores):
    """Test different alpha values."""
    if not scores:
        return {}
    mean_score = sum(scores) / len(scores)
    if mean_score == 0.0:
        return {}
    min_score = min(scores)
    results = {}
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        penalty = (min_score / mean_score) ** alpha
        results[f"alpha_{alpha}"] = mean_score * penalty
    return results

def evaluate_formula(holistic_path, gca_path, formula_func, formula_name):
    """Evaluate a formula against holistic and measure agreement."""
    holistic = load_preferences(holistic_path)
    with open(gca_path) as f:
        gca_raw = [json.loads(l) for l in f]
    
    agreements = 0
    correct = 0
    total = 0
    
    for rec in gca_raw:
        sample_id = rec["sample_id"]
        if sample_id not in holistic:
            continue
        
        # Compute new GCA score using formula
        scores_a = [s["score"] for s in rec["sentence_details_a"]]
        scores_b = [s["score"] for s in rec["sentence_details_b"]]
        
        new_score_a = formula_func(scores_a)
        new_score_b = formula_func(scores_b)
        
        # Determine new decision
        if new_score_a > new_score_b:
            new_decision = "A"
        elif new_score_b > new_score_a:
            new_decision = "B"
        else:
            new_decision = "no_preference"
        
        holistic_decision = holistic[sample_id]["decision"]
        
        # Check agreement with holistic
        if new_decision == holistic_decision:
            agreements += 1
        
        total += 1
    
    return agreements / total if total > 0 else 0.0

def main():
    hol_path = "artifacts/mogon_run_1000/holistic_reward_preferences_1000.jsonl"
    gca_path = "artifacts/mogon_run_1000/gca_reward_preferences_1000.jsonl"
    
    print("=" * 70)
    print("GCA Aggregation Formula Optimization")
    print("=" * 70)
    print("\nTesting alternative aggregation formulas against holistic ground truth.\n")
    
    formulas = {
        "Current (alpha=0.5)": lambda scores: aggregate_current(scores, 0.5),
        "Mean (no penalty)": aggregate_mean,
        "Harmonic mean": aggregate_harmonic,
        "Geometric mean": aggregate_geometric,
        "Minimum score": aggregate_min,
        "Current (alpha=0.2)": lambda scores: aggregate_current(scores, 0.2),
        "Current (alpha=0.3)": lambda scores: aggregate_current(scores, 0.3),
        "Current (alpha=0.7)": lambda scores: aggregate_current(scores, 0.7),
        "Current (alpha=1.0)": lambda scores: aggregate_current(scores, 1.0),
    }
    
    results = []
    for name, func in formulas.items():
        agreement = evaluate_formula(hol_path, gca_path, func, name)
        results.append((name, agreement))
        print(f"{name:30s}: {agreement:.4f} agreement with holistic")
    
    print("\n" + "=" * 70)
    print("RANKING (best to worst):")
    print("=" * 70)
    for i, (name, agreement) in enumerate(sorted(results, key=lambda x: -x[1]), 1):
        print(f"{i}. {name:30s}: {agreement:.4f} ({int(agreement*1000)} / 1000 pairs agree)")
    
    best_name, best_agreement = sorted(results, key=lambda x: -x[1])[0]
    print(f"\n✓ BEST FORMULA: {best_name} ({best_agreement:.4f})")
    print(f"  Improvement vs current: {(best_agreement - results[0][1])*100:+.2f}%")

if __name__ == "__main__":
    main()
