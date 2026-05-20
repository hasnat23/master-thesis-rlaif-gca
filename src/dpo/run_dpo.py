#!/usr/bin/env python3
"""
Fine-tune Mistral-7B-Instruct-v0.3 with DPO on holistic or GCA preference pairs.

Usage:
    python src/dpo/run_dpo.py \
        --condition holistic \
        --preferences data/preferences/holistic_reward_preferences_200.jsonl \
        --model-path models/Mistral-7B-Instruct-v0.3 \
        --output-dir outputs/dpo/holistic \
        --epochs 1 --batch-size 2 --grad-accum 8 --lr 5e-7

The script uses LoRA (via PEFT) so the full 7B model fits on a single A100-40GB.
Adapter weights are saved to <output_dir>/adapter/; a training metrics JSON is
written to <output_dir>/dpo_metrics.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logging import setup_logger, get_run_id


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_preferences(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_hf_dataset(records: list[dict], tokenizer):
    """
    Convert preference records to the format expected by TRL DPOTrainer:
      {"prompt": str, "chosen": str, "rejected": str}

    We use the chat-template to format article+instruction as the prompt and
    each summary as the completion.
    """
    from datasets import Dataset

    rows = []
    skipped = 0
    for r in records:
        if not r.get("chosen") or not r.get("rejected"):
            skipped += 1
            continue
        article = r["article"].strip()
        prompt_text = (
            f"Summarize the following news article in 2-3 sentences, "
            f"focusing on the most important factual information.\n\n"
            f"Article:\n{article}\n\nSummary:"
        )
        chosen_text  = " " + r["chosen"].strip()
        rejected_text = " " + r["rejected"].strip()

        # Apply chat template so the model sees [INST]…[/INST] formatting
        prompt_messages = [{"role": "user", "content": prompt_text}]
        prompt_formatted = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        rows.append({
            "prompt":   prompt_formatted,
            "chosen":   chosen_text,
            "rejected": rejected_text,
        })

    if skipped:
        print(f"  WARNING: skipped {skipped} records with null chosen/rejected fields")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning on preference pairs")
    parser.add_argument("--condition",    type=str, required=True,
                        choices=["holistic", "gca"],
                        help="Which preference condition to train on")
    parser.add_argument("--preferences",  type=str, required=True,
                        help="Path to preference JSONL file")
    parser.add_argument("--model-path",   type=str, default="models/Mistral-7B-Instruct-v0.3",
                        help="Local path to base model weights")
    parser.add_argument("--output-dir",   type=str, default="outputs/dpo",
                        help="Directory to write adapter and metrics")
    parser.add_argument("--epochs",       type=int,   default=1)
    parser.add_argument("--batch-size",   type=int,   default=2,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum",   type=int,   default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr",           type=float, default=5e-7)
    parser.add_argument("--beta",         type=float, default=0.1,
                        help="KL penalty coefficient for DPO")
    parser.add_argument("--max-length",   type=int,   default=1024,
                        help="Max total tokens (prompt + completion)")
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--lora-r",       type=int,   default=16)
    parser.add_argument("--lora-alpha",   type=int,   default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    run_id = get_run_id()
    logger = setup_logger("dpo_training", run_id=run_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Condition: {args.condition}")
    logger.info(f"Preferences: {args.preferences}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output: {output_dir}")

    # ------------------------------------------------------------------
    # Imports (deferred so CLI --help works without GPU)
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import DPOTrainer, DPOConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                    f"{torch.cuda.get_device_properties(0).total_memory // (1024**2)} MiB")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info("Loading preference pairs …")
    records = _load_preferences(args.preferences)
    logger.info(f"  {len(records)} pairs loaded")
    dataset = _build_hf_dataset(records, tokenizer)
    logger.info(f"  Dataset size: {len(dataset)}")

    # ------------------------------------------------------------------
    # Model — load in bf16, no device_map (single GPU)
    # ------------------------------------------------------------------
    logger.info("Loading base model …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # Reference model: same base, frozen. DPOTrainer handles this internally
    # when peft_config is passed (it creates its own frozen copy via adapter disable).
    # We pass the base model once; TRL creates ref by disabling adapters.

    # ------------------------------------------------------------------
    # LoRA config
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # ------------------------------------------------------------------
    # DPO training config
    # ------------------------------------------------------------------
    training_args = DPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    logger.info("Initialising DPOTrainer …")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info(f"Training for {args.epochs} epoch(s) …")
    train_result = trainer.train()

    # ------------------------------------------------------------------
    # Save adapter
    # ------------------------------------------------------------------
    logger.info(f"Saving LoRA adapter to {adapter_dir} …")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = {
        "run_id":        run_id,
        "condition":     args.condition,
        "model_path":    args.model_path,
        "preferences":   args.preferences,
        "n_pairs":       len(records),
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "grad_accum":    args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr":            args.lr,
        "beta":          args.beta,
        "lora_r":        args.lora_r,
        "lora_alpha":    args.lora_alpha,
        "train_loss":    train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", None),
    }
    metrics_path = output_dir / "dpo_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("=== DPO fine-tuning complete ===")
    logger.info(f"  Train loss : {train_result.training_loss:.4f}")
    logger.info(f"  Adapter    : {adapter_dir}")


if __name__ == "__main__":
    main()
