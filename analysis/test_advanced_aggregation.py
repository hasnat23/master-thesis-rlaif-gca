#!/usr/bin/env python3
"""
Test advanced aggregation strategies against holistic baseline.
Builds on the formula optimization work to find even better approaches.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judging.advanced_aggregation import (
    aggregate_weighted_importance,
    aggregate_confidence_weighted,
    aggregate_length_weighted,
    aggregate_robust_mean,
    aggregate_harmonic_mean,
    aggregate_ensemble,
)


def load_preferences(jsonl_path: str):
    """Load GCA preferences."""
    prefs = []
    with open(jsonl_path) as f:
        for line in f:
            prefs.append(json.loads(line))
    return prefs


def evaluate_aggregation(
    preferences: list,
    holistic_prefs: dict,
    aggregation_func,
    func_name: str,
    use_sentence_lengths: bool = False,
) -> float:
    """
    Test an aggregation function and return agreement with holistic.
    """
    agreements = 0
    total = 0
    
    for rec in preferences:
        sample_id = rec["sample_id"]
        if sample_id not in holistic_prefs:
            continue
        
        # Get sentence details
        sentences_a = rec["sentence_details_a"]
        sentences_b = rec["sentence_details_b"]
        
        scores_a = [s["score"] for s in sentences_a]
        scores_b = [s["score"] for s in sentences_b]
        
        # Compute score based on function type
        if use_sentence_lengths and "text" in sentences_a[0]:
            lengths_a = [len(s["text"].split()) for s in sentences_a]
            lengths_b = [len(s["text"].split()) for s in sentences_b]
            score_a = aggregation_func(scores_a, lengths_a)
            score_b = aggregation_func(scores_b, lengths_b)
        else:
            score_a = aggregation_func(scores_a)
            score_b = aggregation_func(scores_b)
        
        # Determine decision
        if score_a > score_b:
            new_decision = "A"
        elif score_b > score_a:
            new_decision = "B"
        else:
            new_decision = "no_preference"
        
        # Compare to holistic
        holistic_decision = holistic_prefs[sample_id]["decision"]
        if new_decision == holistic_decision:
            agreements += 1
        
        total += 1
    
    return agreements / total if total > 0 else 0.0


def main():
    print("\n" + "=" * 80)
    print("ADVANCED GCA AGGREGATION STRATEGIES EVALUATION")
    print("=" * 80)
    
    # Load preferences
    hol_path = "artifacts/mogon_run_1000/holistic_reward_preferences_1000.jsonl"
    gca_path = "artifacts/mogon_run_1000/gca_reward_preferences_1000.jsonl"
    
    if not Path(hol_path).exists() or not Path(gca_path).exists():
        print(f"ERROR: Preference files not found at {hol_path} or {gca_path}")
        return
    
    print("\nLoading preferences...")
    holistic_list = load_preferences(hol_path)
    gca_list = load_preferences(gca_path)
    
    # Index holistic by sample_id
    holistic_prefs = {rec["sample_id"]: rec for rec in holistic_list}
    
    print(f"Loaded {len(holistic_list)} holistic preferences")
    print(f"Loaded {len(gca_list)} GCA preferences\n")
    
    # Define aggregation strategies to test
    strategies = {
        "Simple Mean (baseline)": (lambda scores: sum(scores)/len(scores) if scores else 0.0, False),
        "Weighted by Position": (aggregate_weighted_importance, False),
        "Confidence Weighted": (aggregate_confidence_weighted, False),
        "Length Weighted": (aggregate_length_weighted, True),
        "Robust Mean (trim 10%)": (aggregate_robust_mean, False),
        "Harmonic Mean": (aggregate_harmonic_mean, False),
        "Ensemble of Methods": (aggregate_ensemble, False),
    }
    
    print("Evaluating aggregation strategies...")
    print("-" * 80)
    
    results = []
    for name, (func, use_lengths) in strategies.items():
        agreement = evaluate_aggregation(gca_list, holistic_prefs, func, name, use_lengths)
        results.append((name, agreement))
        print(f"{name:30s}: {agreement:.4f} ({int(agreement*1000)}/1000 pairs agree)")
    
    # Ranking
    print("\n" + "=" * 80)
    print("RANKING (Best to Worst):")
    print("=" * 80)
    for rank, (name, agreement) in enumerate(sorted(results, key=lambda x: -x[1]), 1):
        print(f"{rank}. {name:30s}: {agreement:.4f}")
    
    # Identify best
    best_name, best_agreement = sorted(results, key=lambda x: -x[1])[0]
    baseline_name, baseline_agreement = results[0]  # Simple mean is first in dict order
    
    print("\n" + "=" * 80)
    print("FINDINGS:")
    print("=" * 80)
    print(f"Best Strategy: {best_name}")
    print(f"Agreement: {best_agreement:.4f}")
    
    if best_agreement > baseline_agreement:
        improvement = (best_agreement - baseline_agreement) * 100
        print(f"Improvement vs Simple Mean: +{improvement:.2f}%")
    else:
        degradation = (baseline_agreement - best_agreement) * 100
        print(f"Degradation vs Simple Mean: -{degradation:.2f}%")
    
    print("\nRecommendation:")
    if best_agreement > baseline_agreement:
        print(f"  → Use '{best_name}' instead of simple mean")
        print(f"    This will improve GCA RM training by ~{improvement:.1f}%")
    else:
        print(f"  → Stick with Simple Mean (already optimal)")
        print(f"    Other strategies don't improve agreement with holistic baseline")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
