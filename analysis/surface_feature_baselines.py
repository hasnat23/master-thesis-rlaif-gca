#!/usr/bin/env python3
"""Can trivial surface features recover the preference labels?

Chapter 4 flags a confound: the two candidates in each pair are generated at
different decoding temperatures, so features correlated with temperature
(notably summary length) might carry part of the label information. A reward
model could then score well without representing factuality at all.

That concern is only as serious as the amount of label information those
features actually carry, which is measurable directly. This script fits the
cheapest possible "reward models" on the stored preference pairs and reports
how far they get:

  length      - prefer whichever candidate is longer (or shorter)
  sent_count  - prefer whichever candidate has more (or fewer) sentences
  position    - always prefer candidate A (the low-temperature one)

Each is scored against the same decision the AlignScore judge made. If these
baselines sit near 50%, the labels are not recoverable from surface form and
the trained reward models must be using something else. If they approach the
53-57% the real reward models reach, the confound is doing real work and the
headline comparison is correspondingly weaker.

Both the holistic and GCA preference sets are evaluated, since Chapter 4 also
notes the temperature asymmetry differs between them.

Usage:
    python analysis/surface_feature_baselines.py
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PREFERENCE_FILES = {
    "holistic": "data/preferences/holistic_reward_preferences_200.jsonl",
    "gca": "data/preferences/gca_reward_preferences_200.jsonl",
}

# Matches the regex splitter used in src/judging so sentence counts here are
# comparable with the ones the GCA pipeline actually saw.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_count(text: str) -> int:
    return len([s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()])


def load_pairs(path: Path) -> list[dict]:
    """Load pairs that carry an actual A/B decision (skipping no_preference)."""
    pairs = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("decision") not in ("A", "B"):
                continue
            pairs.append({
                "decision": rec["decision"],
                "len_a": len(rec["summary_a"]),
                "len_b": len(rec["summary_b"]),
                "sent_a": sentence_count(rec["summary_a"]),
                "sent_b": sentence_count(rec["summary_b"]),
                "score_diff": rec.get("score_diff", 0.0),
            })
    return pairs


def feature_baseline(pairs: list[dict], key_a: str, key_b: str) -> dict:
    """Accuracy of preferring the larger, and of preferring the smaller, value.

    Ties are counted as half-correct, which is what a classifier forced to
    guess would average.
    """
    larger_correct = 0.0
    ties = 0
    for p in pairs:
        va, vb = p[key_a], p[key_b]
        if va == vb:
            ties += 1
            larger_correct += 0.5
            continue
        predicted = "A" if va > vb else "B"
        if predicted == p["decision"]:
            larger_correct += 1.0

    n = len(pairs)
    acc_larger = larger_correct / n
    return {
        "prefer_larger_acc": acc_larger,
        "prefer_smaller_acc": 1.0 - acc_larger,
        "best_acc": max(acc_larger, 1.0 - acc_larger),
        "ties": ties,
    }


def position_baseline(pairs: list[dict]) -> dict:
    """Accuracy of always choosing candidate A (the T=0.7 candidate)."""
    a_wins = sum(1 for p in pairs if p["decision"] == "A")
    acc = a_wins / len(pairs)
    return {"always_a_acc": acc, "a_wins": a_wins, "n": len(pairs)}


def mean_by_outcome(pairs: list[dict], key_a: str, key_b: str) -> tuple[float, float]:
    """Mean feature value for chosen and for rejected candidates."""
    chosen, rejected = [], []
    for p in pairs:
        if p["decision"] == "A":
            chosen.append(p[key_a])
            rejected.append(p[key_b])
        else:
            chosen.append(p[key_b])
            rejected.append(p[key_a])
    return statistics.mean(chosen), statistics.mean(rejected)


def main() -> None:
    results = {}

    for condition, rel_path in PREFERENCE_FILES.items():
        path = REPO / rel_path
        if not path.exists():
            print(f"missing: {rel_path}")
            continue

        pairs = load_pairs(path)
        n = len(pairs)

        length = feature_baseline(pairs, "len_a", "len_b")
        sents = feature_baseline(pairs, "sent_a", "sent_b")
        position = position_baseline(pairs)

        len_chosen, len_rejected = mean_by_outcome(pairs, "len_a", "len_b")
        sent_chosen, sent_rejected = mean_by_outcome(pairs, "sent_a", "sent_b")

        results[condition] = {
            "n_pairs": n,
            "length_baseline": length,
            "sentence_count_baseline": sents,
            "position_baseline": position,
            "mean_length_chosen": len_chosen,
            "mean_length_rejected": len_rejected,
            "mean_sentences_chosen": sent_chosen,
            "mean_sentences_rejected": sent_rejected,
        }

        print(f"\n=== {condition.upper()}  ({n} decided pairs) ===")
        print(f"  characters   chosen {len_chosen:7.1f}   rejected {len_rejected:7.1f}"
              f"   diff {len_chosen - len_rejected:+.1f}")
        print(f"  sentences    chosen {sent_chosen:7.2f}   rejected {sent_rejected:7.2f}"
              f"   diff {sent_chosen - sent_rejected:+.2f}")
        print(f"  -- baseline accuracies against the judge's decision --")
        print(f"  prefer longer          {length['prefer_larger_acc']:.3f}")
        print(f"  prefer shorter         {length['prefer_smaller_acc']:.3f}")
        print(f"  prefer more sentences  {sents['prefer_larger_acc']:.3f}")
        print(f"  prefer fewer sentences {sents['prefer_smaller_acc']:.3f}")
        print(f"  always choose A (T=0.7){position['always_a_acc']:.3f}")
        print(f"  best surface baseline  {max(length['best_acc'], sents['best_acc'], position['always_a_acc']):.3f}")

    out = REPO / "reports" / "campaigns" / "surface_feature_baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out.relative_to(REPO)}")

    print("\nInterpretation: 'always choose A' is the majority-class baseline and")
    print("reflects the temperature asymmetry already reported in Chapter 4. The")
    print("length and sentence-count rows are the ones that matter for the")
    print("confound question, since a reward model reading the summary text could")
    print("in principle exploit them.")


if __name__ == "__main__":
    main()
