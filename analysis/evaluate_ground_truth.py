#!/usr/bin/env python3
"""Evaluate holistic and GCA reward models against the independent ground truth.

For each article in the ground-truth set, AlignScore's sentence-level score
aggregated via GCA (alpha=0.0, the locked formula) has already ranked the two
candidate summaries into "should score higher" (chosen/A) and "should score
lower" (rejected/B) -- see slurm/build_ground_truth.sh. That ranking is fixed
and shared: both reward models are tested against the SAME ground truth,
rather than each being re-tested against its own training-time labels.

For each trained checkpoint (holistic or GCA, one per seed), this scores
every ground-truth pair and computes the "success rate": the fraction of
pairs where the reward model scores the higher-AlignScore summary above the
lower-AlignScore one. This is Lingxiao's requested metric (email, 11 Aug
2026): "the success rate of the GCA reward model in giving a higher score to
the summary that should receive the higher score."

Usage:
    python analysis/evaluate_ground_truth.py \
        --ground-truth data/preferences_groundtruth/gca_reward_preferences_groundtruth.jsonl \
        --checkpoints-root outputs/ground_truth_eval \
        --seeds 1-20 \
        --max-article-chars 2000 --max-length 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.reward_model.train import BradleyTerryRewardModel


def parse_seed_range(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def load_ground_truth(path: Path, max_article_chars: int) -> list[dict]:
    """Each pair's "low" side is scored against its own source article.

    The same-article ground truth (evaluate_ground_truth's default input)
    only ever has one "article" field, since both summaries come from the
    same source. The biased/global ground truth
    (build_biased_ground_truth.py) pools summaries across all articles, so
    the high and low summaries can come from different source articles --
    it supplies "article_low" explicitly for that case. Falling back to
    "article" keeps this reader compatible with both files.
    """
    pairs = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("decision") not in ("A", "B"):
                continue
            pairs.append({
                "sample_id": rec["sample_id"],
                "article_high": rec["article"][:max_article_chars],
                "article_low": rec.get("article_low", rec["article"])[:max_article_chars],
                "high": rec["chosen"],   # higher GCA-aggregated AlignScore
                "low": rec["rejected"],  # lower GCA-aggregated AlignScore
            })
    return pairs


@torch.no_grad()
def score_texts(model, tokenizer, texts: list[str], device: str,
                 max_length: int, batch_size: int = 16) -> list[float]:
    scores = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, max_length=max_length, truncation=True,
                         padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        r = model(**enc)
        scores.extend(r.detach().cpu().tolist())
    return scores


def load_checkpoint_model(ckpt_dir: Path, device: str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    model = BradleyTerryRewardModel(backbone=str(ckpt_dir))
    head_path = ckpt_dir / "reward_head.pt"
    model.reward_head.load_state_dict(torch.load(head_path, map_location=device))
    model.to(device)
    return model, tokenizer


def evaluate_checkpoint(ckpt_dir: Path, pairs: list[dict], device: str,
                         max_length: int) -> dict:
    model, tokenizer = load_checkpoint_model(ckpt_dir, device)

    high_texts = [f"{p['article_high']} [SEP] {p['high']}" for p in pairs]
    low_texts = [f"{p['article_low']} [SEP] {p['low']}" for p in pairs]

    r_high = score_texts(model, tokenizer, high_texts, device, max_length)
    r_low = score_texts(model, tokenizer, low_texts, device, max_length)

    successes = [1 if rh > rl else 0 for rh, rl in zip(r_high, r_low)]
    success_rate = sum(successes) / len(successes)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "n_pairs": len(pairs),
        "successes": sum(successes),
        "success_rate": success_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="data/preferences_groundtruth/gca_reward_preferences_groundtruth.jsonl")
    ap.add_argument("--checkpoints-root", default="outputs/ground_truth_eval")
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument("--max-article-chars", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", default="reports/campaigns/ground_truth_eval.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    gt_path = REPO / args.ground_truth
    pairs = load_ground_truth(gt_path, args.max_article_chars)
    print(f"Loaded {len(pairs)} ground-truth pairs from {gt_path}")

    seeds = parse_seed_range(args.seeds)
    root = REPO / args.checkpoints_root

    results = {"holistic": {}, "gca": {}}
    for seed in seeds:
        for condition in ("holistic", "gca"):
            ckpt_dir = root / f"seed_{seed}" / condition / "best"
            if not ckpt_dir.exists():
                print(f"  seed {seed} [{condition}]: checkpoint not found at {ckpt_dir}, skipping")
                continue
            r = evaluate_checkpoint(ckpt_dir, pairs, device, args.max_length)
            results[condition][seed] = r
            print(f"  seed {seed:2d} [{condition:>8s}]: success_rate={r['success_rate']:.4f} "
                  f"({r['successes']}/{r['n_pairs']})")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "ground_truth_path": str(args.ground_truth),
            "n_pairs": len(pairs),
            "results": results,
        }, fh, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
