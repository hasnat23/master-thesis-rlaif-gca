"""
Bradley-Terry Reward Model training for faithfulness preference learning.

Trains a scalar reward model R(article, summary) using pairwise preference data
via the Bradley-Terry log-likelihood objective:

    L = -sum_i [ log σ(R(w_i) - R(l_i)) ]

where w_i is the preferred (chosen) summary and l_i is the rejected one.

This is equivalent to the IRL framing: given preference demonstrations, learn
the latent reward function that explains them.

Backbone: microsoft/deberta-v3-base (384-dim, fast, no proxy needed if cached)
Input:    [CLS] article [SEP] summary [SEP]  →  scalar reward head
"""

import json
import logging
import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


# ── Dataset ──────────────────────────────────────────────────────────────────

class PreferencePairDataset(Dataset):
    """
    Loads a JSONL preference file and returns (chosen, rejected) text pairs.

    Each JSONL record is expected to have:
        article, summary_a, summary_b, decision  ("A" | "B")
    """

    def __init__(self, jsonl_path: str, max_article_chars: int = 2000):
        self.pairs: list[dict] = []
        self.max_article_chars = max_article_chars

        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                decision = rec.get("decision") or rec.get("winner")
                if decision not in ("A", "B"):
                    continue  # skip ties / no_preference
                article = rec.get("article", "")[:max_article_chars]
                summary_a = rec.get("summary_a", "")
                summary_b = rec.get("summary_b", "")
                if decision == "A":
                    chosen, rejected = summary_a, summary_b
                else:
                    chosen, rejected = summary_b, summary_a
                self.pairs.append({
                    "article": article,
                    "chosen": chosen,
                    "rejected": rejected,
                })

        logger.info(f"Loaded {len(self.pairs)} usable preference pairs from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return self.pairs[idx]


def collate_fn(tokenizer, max_length: int = 512):
    """Returns a collate function that tokenizes chosen/rejected pairs."""
    def _collate(batch: list[dict]) -> dict:
        chosen_texts  = [f"{b['article']} [SEP] {b['chosen']}"  for b in batch]
        rejected_texts = [f"{b['article']} [SEP] {b['rejected']}" for b in batch]

        chosen_enc = tokenizer(
            chosen_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        rejected_enc = tokenizer(
            rejected_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return {"chosen": chosen_enc, "rejected": rejected_enc}
    return _collate


# ── Model ─────────────────────────────────────────────────────────────────────

class BradleyTerryRewardModel(nn.Module):
    """
    Encoder + scalar reward head.

    R(article, summary) = W · pool(encoder([article; summary])) + b
    """

    def __init__(self, backbone: str = "microsoft/deberta-v3-base", dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        hidden = self.encoder.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.reward_head(pooled).squeeze(-1)  # (batch,)


# ── Training ──────────────────────────────────────────────────────────────────

def bradley_terry_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry pairwise ranking loss:
        L = -log σ(r_w - r_l)
    """
    return -torch.nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()


def train_reward_model(
    train_path: str,
    val_path: Optional[str],
    output_dir: str,
    backbone: str = "microsoft/deberta-v3-base",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 512,
    max_article_chars: int = 2000,
    seed: int = 42,
    device: str = "auto",
) -> str:
    """
    Train a Bradley-Terry RM on preference pairs and save checkpoints.

    Args:
        train_path:       Path to training preference JSONL (holistic or GCA).
        val_path:         Optional validation JSONL for held-out accuracy tracking.
        output_dir:       Directory to save best checkpoint and metrics.
        backbone:         HF model ID for the encoder backbone.
        epochs:           Training epochs.
        batch_size:       Per-device batch size.
        lr:               Learning rate.
        warmup_ratio:     Fraction of total steps used for linear warmup.
        max_length:       Max token length for [article; summary] concatenation.
        max_article_chars: Truncate articles to this many chars before tokenising.
        seed:             Random seed.
        device:           "auto", "cuda", or "cpu".
        
    Returns:
        Path to the saved best model checkpoint directory.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Data
    train_ds = PreferencePairDataset(train_path, max_article_chars=max_article_chars)
    val_ds   = PreferencePairDataset(val_path, max_article_chars=max_article_chars) if val_path else None

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    _collate  = collate_fn(tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=2) if val_ds else None

    # Model
    model = BradleyTerryRewardModel(backbone=backbone).to(device)

    total_steps   = len(train_loader) * epochs
    warmup_steps  = math.ceil(total_steps * warmup_ratio)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log: list[dict] = []
    best_val_acc = -1.0
    best_ckpt_dir = str(output_dir / "best")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            chosen_enc   = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected_enc = {k: v.to(device) for k, v in batch["rejected"].items()}

            r_chosen   = model(**chosen_enc)
            r_rejected = model(**rejected_enc)
            loss       = bradley_terry_loss(r_chosen, r_rejected)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 10 == 0:
                logger.info(f"  epoch {epoch} step {step+1}/{len(train_loader)} "
                            f"loss={loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # ── Validate ──
        val_acc = _evaluate(model, val_loader, device) if val_loader else None

        row = {"epoch": epoch, "train_loss": avg_train_loss, "val_acc": val_acc}
        metrics_log.append(row)
        logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}"
                    + (f"  val_acc={val_acc:.3f}" if val_acc is not None else ""))

        # Save best checkpoint by val accuracy (or every epoch if no val set)
        if val_acc is None or val_acc > best_val_acc:
            best_val_acc = val_acc or 0.0
            _save_checkpoint(model, tokenizer, best_ckpt_dir)
            logger.info(f"  → Saved best checkpoint to {best_ckpt_dir}")

    # Save metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

    return best_ckpt_dir


def _evaluate(model: BradleyTerryRewardModel, loader: DataLoader, device: str) -> float:
    """Pairwise accuracy: fraction of pairs where R(chosen) > R(rejected)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            chosen_enc   = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected_enc = {k: v.to(device) for k, v in batch["rejected"].items()}
            r_chosen   = model(**chosen_enc)
            r_rejected = model(**rejected_enc)
            correct += (r_chosen > r_rejected).sum().item()
            total   += r_chosen.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def _save_checkpoint(model: BradleyTerryRewardModel, tokenizer, path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(model.reward_head.state_dict(), str(Path(path) / "reward_head.pt"))
