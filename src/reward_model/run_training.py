#!/usr/bin/env python3
"""
Train Bradley-Terry reward models on holistic and GCA preference pairs.

Usage — train both conditions in one call:
    python src/reward_model/run_training.py \
        --holistic data/preferences_rm500/holistic_reward_preferences_rm500.jsonl \
        --gca      data/preferences_rm500/gca_reward_preferences_rm500.jsonl \
        --output-dir outputs/reward_models \
        --backbone microsoft/deberta-v3-base \
        --epochs 5 --batch-size 8 --lr 2e-5

Or train a single condition:
    python src/reward_model/run_training.py \
        --holistic data/preferences_rm500/holistic_reward_preferences_rm500.jsonl \
        --output-dir outputs/reward_models

Cross-validation (k-fold, no held-out split needed):
    python src/reward_model/run_training.py \
        --holistic ... --gca ... --output-dir ... --kfold 5
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.reward_model.train import train_reward_model, PreferencePairDataset, _evaluate
from src.utils.logging import setup_logger, get_run_id


def _kfold_cv(
    jsonl_path: str,
    condition: str,
    output_dir: Path,
    backbone: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    max_article_chars: int,
    k: int,
    seed: int,
):
    """Run k-fold cross-validation and return per-fold accuracies."""
    import math, copy
    from torch.utils.data import DataLoader, Subset
    from src.reward_model.train import (
        PreferencePairDataset, collate_fn, BradleyTerryRewardModel,
        bradley_terry_loss, _evaluate,
    )
    import torch, torch.nn as nn
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    logger = logging.getLogger("kfold")
    logger.info(f"K-fold CV (k={k}) for condition={condition}")

    ds = PreferencePairDataset(jsonl_path, max_article_chars=max_article_chars)
    n  = len(ds)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    fold_size = n // k

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    _coll     = collate_fn(tokenizer, max_length=max_length)
    device    = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    fold_accs = []
    for fold in range(k):
        val_idx   = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]

        train_loader = DataLoader(
            Subset(ds, train_idx), batch_size=batch_size, shuffle=True,
            collate_fn=_coll, num_workers=2,
        )
        val_loader = DataLoader(
            Subset(ds, val_idx), batch_size=batch_size, shuffle=False,
            collate_fn=_coll, num_workers=2,
        )

        model     = BradleyTerryRewardModel(backbone=backbone).to(device)
        total_steps  = len(train_loader) * epochs
        warmup_steps = math.ceil(total_steps * 0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                ce = {k2: v.to(device) for k2, v in batch["chosen"].items()}
                re = {k2: v.to(device) for k2, v in batch["rejected"].items()}
                loss = bradley_terry_loss(model(**ce), model(**re))
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()

        acc = _evaluate(model, val_loader, device)
        fold_accs.append(acc)
        logger.info(f"  Fold {fold+1}/{k}: val_acc={acc:.3f}")

    mean_acc = sum(fold_accs) / len(fold_accs)
    logger.info(f"K-fold mean acc ({condition}): {mean_acc:.3f}  folds={fold_accs}")
    return fold_accs, mean_acc


def main():
    parser = argparse.ArgumentParser(description="Train Bradley-Terry reward models")
    parser.add_argument("--holistic", type=str, default=None,
                        help="Holistic preference JSONL path")
    parser.add_argument("--gca", type=str, default=None,
                        help="GCA preference JSONL path")
    parser.add_argument("--holistic-val", type=str, default=None)
    parser.add_argument("--gca-val", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/reward_models")
    parser.add_argument("--backbone", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-article-chars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kfold", type=int, default=0,
                        help="If >0, run k-fold CV instead of single train/val split")
    args = parser.parse_args()

    if not args.holistic and not args.gca:
        parser.error("Provide at least --holistic or --gca")

    run_id = get_run_id()
    logger = setup_logger("rm_training", run_id=run_id)
    output_dir = Path(args.output_dir)

    summary: dict = {"run_id": run_id, "backbone": args.backbone, "conditions": {}}

    for condition, train_path, val_path in [
        ("holistic", args.holistic, args.holistic_val),
        ("gca",      args.gca,      args.gca_val),
    ]:
        if not train_path:
            continue
        logger.info(f"\n{'='*60}\nCondition: {condition}\n{'='*60}")
        cond_out = output_dir / condition

        if args.kfold > 0:
            fold_accs, mean_acc = _kfold_cv(
                jsonl_path=train_path,
                condition=condition,
                output_dir=cond_out,
                backbone=args.backbone,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
                max_article_chars=args.max_article_chars,
                k=args.kfold,
                seed=args.seed,
            )
            summary["conditions"][condition] = {
                "kfold": args.kfold,
                "fold_accs": fold_accs,
                "mean_val_acc": mean_acc,
            }
        else:
            best_ckpt = train_reward_model(
                train_path=train_path,
                val_path=val_path,
                output_dir=str(cond_out),
                backbone=args.backbone,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
                max_article_chars=args.max_article_chars,
                seed=args.seed,
            )
            summary["conditions"][condition] = {"best_checkpoint": best_ckpt}

    # Save summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "rm_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
