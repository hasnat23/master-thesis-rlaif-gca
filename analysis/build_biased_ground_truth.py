#!/usr/bin/env python3
"""Build a deliberately biased (stark-contrast) ground truth, as Lingxiao
specified: pool every candidate summary, sort by its AlignScore
sentence-level score aggregated via GCA, and split into a clearly-high
subset (A) and a clearly-low subset (B) -- not paired within an article.

This is a different, easier test than analysis/evaluate_ground_truth.py's
same-article ground truth. There, the two candidates being compared come
from the same article and the same strong model at two nearby temperatures,
so their scores are naturally close (e.g. 0.55 vs 0.51) -- a subtle,
hard discrimination. Here, subset A and subset B are the true top and
bottom of the whole score distribution, pooled across all articles, so the
gap between what's being compared is large by construction. A reward model
that has learned even a coarse factuality signal should score close to
100% on this; if it doesn't, that says something is wrong with the RM or
the measurement itself, not with GCA specifically. This is a sanity check
on the measurement pipeline, not a replacement for the harder, same-article
precision test.

Output matches the schema evaluate_ground_truth.py already reads
(sample_id, article, chosen, rejected, decision), so no changes are needed
to the evaluation script -- point it at this file instead.

Usage:
    python analysis/build_biased_ground_truth.py \
        --preferences data/preferences_groundtruth/gca_reward_preferences_groundtruth.jsonl \
        --top-pct 0.25 --bottom-pct 0.25 \
        --out data/preferences_groundtruth/biased_ground_truth.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preferences", default="data/preferences_groundtruth/gca_reward_preferences_groundtruth.jsonl")
    ap.add_argument("--top-pct", type=float, default=0.25,
                     help="Fraction of the pooled summaries (by score) forming subset A (high).")
    ap.add_argument("--bottom-pct", type=float, default=0.25,
                     help="Fraction of the pooled summaries (by score) forming subset B (low).")
    ap.add_argument("--out", default="data/preferences_groundtruth/biased_ground_truth.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-pairs", action="store_true",
                     help="Evaluate every high-vs-low combination instead of one "
                          "random 1-1 pairing. Removes pairing-order noise at the "
                          "cost of len(A)*len(B) pairs instead of min(len(A),len(B)).")
    args = ap.parse_args()

    pool = []  # (sample_id, article, summary_text, score)
    with open(REPO / args.preferences) as fh:
        for line in fh:
            r = json.loads(line)
            pool.append((r["sample_id"], r["article"], r["summary_a"], r["gca_score_a"]))
            pool.append((r["sample_id"], r["article"], r["summary_b"], r["gca_score_b"]))

    pool.sort(key=lambda x: x[3])
    n = len(pool)
    n_bottom = int(n * args.bottom_pct)
    n_top = int(n * args.top_pct)

    subset_b = pool[:n_bottom]          # lowest scores
    subset_a = pool[n - n_top:]         # highest scores

    print(f"Pooled {n} individual summaries.")
    print(f"Subset A (high): {len(subset_a)}, score range "
          f"[{subset_a[0][3]:.4f}, {subset_a[-1][3]:.4f}]")
    print(f"Subset B (low):  {len(subset_b)}, score range "
          f"[{subset_b[0][3]:.4f}, {subset_b[-1][3]:.4f}]")
    gap = subset_a[0][3] - subset_b[-1][3]
    print(f"Minimum gap between the two subsets: {gap:+.4f}")

    records = []
    if args.all_pairs:
        # Every high-vs-low combination: removes the noise of which particular
        # A happened to be matched with which particular B in a single random
        # draw, at the cost of len(A)*len(B) pairs instead of min(len(A),len(B)).
        for sid_a, art_a, summ_a, score_a in subset_a:
            for sid_b, art_b, summ_b, score_b in subset_b:
                records.append({
                    "sample_id": f"{sid_a}__vs__{sid_b}",
                    "article": art_a,
                    "article_low": art_b,
                    "chosen": summ_a,
                    "rejected": summ_b,
                    "chosen_score": score_a,
                    "rejected_score": score_b,
                    "decision": "A",
                })
        print(f"All-pairs mode: {len(subset_a)} x {len(subset_b)} = {len(records)} pairs")
    else:
        rng = random.Random(args.seed)
        a_shuf = subset_a[:]
        b_shuf = subset_b[:]
        rng.shuffle(a_shuf)
        rng.shuffle(b_shuf)
        n_pairs = min(len(a_shuf), len(b_shuf))
        for i in range(n_pairs):
            sid_a, art_a, summ_a, score_a = a_shuf[i]
            sid_b, art_b, summ_b, score_b = b_shuf[i]
            # Each summary is scored against its OWN source article, matching how
            # the reward model is used everywhere else; the pair need not share
            # an article since this is a deliberately global, stark comparison.
            records.append({
                "sample_id": f"{sid_a}__vs__{sid_b}",
                "article": art_a,
                "article_low": art_b,
                "chosen": summ_a,
                "rejected": summ_b,
                "chosen_score": score_a,
                "rejected_score": score_b,
                "decision": "A",
            })

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} biased pairs to {out_path}")


if __name__ == "__main__":
    main()
