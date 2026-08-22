#!/usr/bin/env python3
"""Check whether a reward model's score correlates with summary length/sentence
count rather than genuine quality -- a "shortcut learning" check.

Hypothesis (closing update, 25 Aug 2026): GCA's precision advantage reverses
as training-set size grows because the GCA reward model learns to exploit a
length- or sentence-count-based shortcut instead of genuine factuality
signal, and leans on that shortcut harder with more training data. If true,
GCA's score-vs-length correlation should be stronger than Holistic's, and
should grow with the training-set size the checkpoint was trained on.

Reuses the same held-out ground-truth summaries and checkpoint-loading code
as evaluate_ground_truth.py, but instead of pairwise win-rate, scores every
individual summary and correlates the raw score against its character
length and sentence count.

Usage:
    python analysis/check_length_shortcut.py \
        --checkpoints-root outputs/ground_truth_eval_3000 --seeds 1-5 \
        --out reports/campaigns/length_shortcut_3000.json
"""

from __future__ import annotations

import argparse
import json
import statistics
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


def load_summaries(path: Path, max_article_chars: int) -> list[dict]:
    """Flattens the same-article ground truth into one row per summary."""
    items = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            article = rec["article"][:max_article_chars]
            items.append({
                "text": rec["summary_a"],
                "article": article,
                "n_sentences": rec["n_sentences_a"],
                "n_chars": len(rec["summary_a"]),
            })
            items.append({
                "text": rec["summary_b"],
                "article": article,
                "n_sentences": rec["n_sentences_b"],
                "n_chars": len(rec["summary_b"]),
            })
    return items


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


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="data/preferences_groundtruth/gca_reward_preferences_groundtruth.jsonl")
    ap.add_argument("--checkpoints-root", required=True)
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--condition", choices=["holistic", "gca"], required=True)
    ap.add_argument("--max-article-chars", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    gt_path = REPO / args.ground_truth
    items = load_summaries(gt_path, args.max_article_chars)
    texts = [f"{it['article']} [SEP] {it['text']}" for it in items]
    n_chars = [it["n_chars"] for it in items]
    n_sentences = [it["n_sentences"] for it in items]
    print(f"Loaded {len(items)} individual summaries from {gt_path}")

    seeds = parse_seed_range(args.seeds)
    root = REPO / args.checkpoints_root

    per_seed = {}
    for seed in seeds:
        ckpt_dir = root / f"seed_{seed}" / args.condition / "best"
        if not ckpt_dir.exists():
            print(f"  seed {seed}: checkpoint not found at {ckpt_dir}, skipping")
            continue
        model, tokenizer = load_checkpoint_model(ckpt_dir, device)
        scores = score_texts(model, tokenizer, texts, device, args.max_length)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        r_len = pearson(scores, [float(c) for c in n_chars])
        r_sent = pearson(scores, [float(s) for s in n_sentences])
        per_seed[seed] = {"r_length": r_len, "r_n_sentences": r_sent}
        print(f"  seed {seed}: r(score, char_len)={r_len:+.3f}  r(score, n_sentences)={r_sent:+.3f}")

    mean_r_len = statistics.mean(v["r_length"] for v in per_seed.values())
    mean_r_sent = statistics.mean(v["r_n_sentences"] for v in per_seed.values())
    print(f"\nMean across {len(per_seed)} seeds: r(score, char_len)={mean_r_len:+.3f}  "
          f"r(score, n_sentences)={mean_r_sent:+.3f}")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "checkpoints_root": args.checkpoints_root,
            "condition": args.condition,
            "n_items": len(items),
            "per_seed": per_seed,
            "mean_r_length": mean_r_len,
            "mean_r_n_sentences": mean_r_sent,
        }, fh, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
