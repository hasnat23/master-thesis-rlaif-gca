#!/usr/bin/env python3
"""
Test alternative judge backends for GCA preference generation.

Current: AlignScore (NLI-based factuality metric)
Alternatives to test:
  1. BERTScore: Semantic similarity between source and summary
  2. AlignScore alternative modes: NLI vs Bin
  3. ROGUE variants: Position-weighted combinations
  4. Ensemble: Combine multiple metrics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import Counter


def load_preferences(path: str, judge_type: str = "alignscore") -> Dict:
    """Load and parse preference file."""
    prefs = {}
    if not Path(path).exists():
        return prefs
    
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            sample_id = obj.get("sample_id", "")
            
            # Extract judge scores based on type
            if judge_type == "alignscore":
                score_a = obj.get("gca_score_a", obj.get("score_a", 0.0))
                score_b = obj.get("gca_score_b", obj.get("score_b", 0.0))
                decision = obj.get("decision", "no_preference")
            else:
                # Generic format
                score_a = obj.get("score_a", 0.0)
                score_b = obj.get("score_b", 0.0)
                decision = obj.get("decision", "no_preference")
            
            prefs[sample_id] = {
                "score_a": score_a,
                "score_b": score_b,
                "decision": decision,
            }
    
    return prefs


def compute_agreement(prefs_a: Dict, prefs_b: Dict) -> float:
    """Compute agreement between two preference sets."""
    agreements = 0
    total = 0
    
    for sample_id in prefs_a:
        if sample_id not in prefs_b:
            continue
        
        if prefs_a[sample_id]["decision"] == prefs_b[sample_id]["decision"]:
            agreements += 1
        total += 1
    
    return agreements / total if total > 0 else 0.0


def analyze_judge_backend():
    """Analyze agreement between different judge backends."""
    
    print("\n" + "=" * 80)
    print("JUDGE BACKEND COMPATIBILITY ANALYSIS".center(80))
    print("=" * 80)
    
    # Load current AlignScore preferences
    hol_path = "data/preferences_1000_opt_alpha_0.0/holistic_reward_preferences_1000_opt_alpha_0.0.jsonl"
    gca_path = "data/preferences_1000_opt_alpha_0.0/gca_reward_preferences_1000_opt_alpha_0.0.jsonl"
    
    if not Path(hol_path).exists():
        print("\nERROR: Optimized preferences not found")
        print(f"  Expected: {hol_path}")
        return 1
    
    print(f"\nLoading current AlignScore preferences...")
    holistic_prefs = load_preferences(hol_path, "alignscore")
    gca_prefs = load_preferences(gca_path, "alignscore")
    
    print(f"  Holistic: {len(holistic_prefs)} pairs")
    print(f"  GCA: {len(gca_prefs)} pairs")
    
    # Analyze decision distribution
    print("\n" + "-" * 80)
    print("CURRENT ALIGNMENT (AlignScore + alpha=0.0)".center(80))
    print("-" * 80)
    
    hol_decisions = Counter([p["decision"] for p in holistic_prefs.values()])
    gca_decisions = Counter([p["decision"] for p in gca_prefs.values()])
    
    print(f"\nDecision distribution:")
    print(f"  Holistic: A={hol_decisions['A']}, B={hol_decisions['B']}, tie={hol_decisions.get('no_preference', 0)}")
    print(f"  GCA:      A={gca_decisions['A']}, B={gca_decisions['B']}, tie={gca_decisions.get('no_preference', 0)}")
    
    agreement = compute_agreement(holistic_prefs, gca_prefs)
    print(f"\nGCA vs Holistic agreement: {agreement:.4f} ({int(agreement*len(gca_prefs))}/{len(gca_prefs)})")
    
    # Score analysis
    hol_scores_diff = [abs(p["score_a"] - p["score_b"]) for p in holistic_prefs.values()]
    gca_scores_diff = [abs(p["score_a"] - p["score_b"]) for p in gca_prefs.values()]
    
    print(f"\nScore discrimination (abs difference):")
    print(f"  Holistic: mean={sum(hol_scores_diff)/len(hol_scores_diff):.4f}, "
          f"median={sorted(hol_scores_diff)[len(hol_scores_diff)//2]:.4f}")
    print(f"  GCA:      mean={sum(gca_scores_diff)/len(gca_scores_diff):.4f}, "
          f"median={sorted(gca_scores_diff)[len(gca_scores_diff)//2]:.4f}")
    
    print("\n" + "=" * 80)
    print("FUTURE OPTIMIZATIONS".center(80))
    print("=" * 80)
    
    print("""
1. BERTSCORE JUDGING
   ├─ Metric: Semantic similarity (using RoBERTa embeddings)
   ├─ Intuition: Captures content overlap vs factuality
   ├─ Expected impact: +2-3% for diversity-aware preferences
   └─ Implementation: Add BERTScore backend to judge
   
2. ALTERNATIVE ALIGNSCORE MODES
   ├─ NLI vs Binary mode: Different confidence thresholds
   ├─ Intuition: May reduce noise vs improve discrimination
   ├─ Expected impact: +1-2% for specific modes
   └─ Implementation: Test --alignscore-eval-mode variants
   
3. WEIGHTED ENSEMBLE JUDGING
   ├─ Combine: 0.6*AlignScore + 0.4*BERTScore
   ├─ Intuition: Balances factuality + semantic similarity
   ├─ Expected impact: +3-5% via complementary signals
   └─ Implementation: Create judge_ensemble.py
   
4. MARGIN OPTIMIZATION
   ├─ Current: 0 (all pairs included)
   ├─ Test: 0.02, 0.05, 0.10 (filter low-confidence pairs)
   ├─ Expected impact: +1-2% (better training signal)
   └─ Implementation: Vary --margin in build_preferences.sh
   
5. RM ARCHITECTURE IMPROVEMENTS
   ├─ Current: FacebookAI/roberta-base
   ├─ Test: roberta-large, deberta-base, electra-base
   ├─ Expected impact: +2-4% (better model capacity)
   └─ Implementation: Test --backbone variants
""")
    
    print("=" * 80)
    print("RECOMMENDATION".center(80))
    print("=" * 80)
    print("""
Priority order for next optimizations:

1. FIRST: Check main job results (submit_gca_opt_v2.sh)
   → If GCA > Holistic: Run hyperparameter search only
   → If GCA ≤ Holistic: Proceed with alternative backends

2. WEIGHTED ENSEMBLE judging (+3-5% potential)
   → Highest expected impact
   → Combines multiple complementary signals
   → Lower implementation complexity

3. Alternative ALIGNSCORE modes (+1-2%)
   → Quick to test (parameter sweep)
   → May reduce noise

4. BERTSCORE backend (+2-3%)
   → More computational cost
   → Good for diversity-aware training

5. RM Architecture (+2-4%)
   → Largest models (roberta-large)
   → Test last if other optimizations plateau
""")
    
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(analyze_judge_backend())
