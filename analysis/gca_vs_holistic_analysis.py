#!/usr/bin/env python3
"""
Analyze disagreement patterns between holistic and GCA preferences.
This is a diagnostic tool to understand why GCA underperforms holistic.
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def load_preferences(jsonl_path: str) -> dict:
    """Load preferences and index by sample_id."""
    prefs = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            prefs[obj["sample_id"]] = obj
    return prefs

def main():
    # Load 1000-sample preferences
    holistic_path = "data/preferences_1000/holistic_reward_preferences_1000.jsonl"
    gca_path = "data/preferences_1000/gca_reward_preferences_1000.jsonl"
    
    print("Loading preferences...")
    holistic = load_preferences(holistic_path)
    gca = load_preferences(gca_path)
    
    print(f"Holistic: {len(holistic)} pairs")
    print(f"GCA: {len(gca)} pairs")
    
    # Analyze disagreements
    agreements = 0
    disagreements = 0
    disagreement_types = Counter()
    score_diffs_holistic = []
    score_diffs_gca = []
    
    for sample_id in holistic:
        if sample_id not in gca:
            continue
            
        h_rec = holistic[sample_id]
        g_rec = gca[sample_id]
        
        h_decision = h_rec["decision"]
        g_decision = g_rec["decision"]
        
        if h_decision == g_decision:
            agreements += 1
        else:
            disagreements += 1
            disagreement_types[f"{h_decision}→{g_decision}"] += 1
            
        score_diffs_holistic.append(abs(h_rec["score_a"] - h_rec["score_b"]))
        score_diffs_gca.append(abs(g_rec["gca_score_a"] - g_rec["gca_score_b"]))
    
    total = agreements + disagreements
    print(f"\nAgreement: {agreements}/{total} ({100*agreements/total:.1f}%)")
    print(f"Disagreement: {disagreements}/{total} ({100*disagreements/total:.1f}%)")
    print(f"\nDisagreement breakdown:")
    for key, count in sorted(disagreement_types.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")
    
    # Score analysis
    print(f"\nScore differences:")
    print(f"  Holistic: mean={sum(score_diffs_holistic)/len(score_diffs_holistic):.4f}, "
          f"median={sorted(score_diffs_holistic)[len(score_diffs_holistic)//2]:.4f}")
    print(f"  GCA: mean={sum(score_diffs_gca)/len(score_diffs_gca):.4f}, "
          f"median={sorted(score_diffs_gca)[len(score_diffs_gca)//2]:.4f}")
    
    # Decision distribution
    h_decisions = Counter([h["decision"] for h in holistic.values()])
    g_decisions = Counter([g["decision"] for g in gca.values()])
    
    print(f"\nDecision distribution:")
    print(f"  Holistic: A={h_decisions['A']}, B={h_decisions['B']}, tie={h_decisions.get('no_preference', 0)}")
    print(f"  GCA: A={g_decisions['A']}, B={g_decisions['B']}, tie={g_decisions.get('no_preference', 0)}")

if __name__ == "__main__":
    main()
