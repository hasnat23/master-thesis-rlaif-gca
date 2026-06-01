#!/usr/bin/env python3
"""
Evaluate DPO-fine-tuned Mistral-7B adapters vs baseline on factual summarization.

Generates summaries from three models:
  - baseline:  Mistral-7B-Instruct-v0.3 (no fine-tuning)
  - dpo_holistic:  base + LoRA adapter trained on holistic preferences
  - dpo_gca:       base + LoRA adapter trained on GCA preferences

Scores each summary against the CNN/DM reference with:
  - ROUGE-1/2/L (lexical overlap)
  - BERTScore F1 (semantic similarity)
  - AlignScore (factual consistency)

Usage:
    python src/eval/run_eval.py \
        --candidates data/candidates/candidates_200.jsonl \
        --model-path models/Mistral-7B-Instruct-v0.3 \
        --holistic-adapter outputs/dpo/holistic/adapter \
        --gca-adapter outputs/dpo/gca/adapter \
        --alignscore-ckpt models/alignscore/AlignScore-base.ckpt \
        --output-dir outputs/eval \
        --n-test 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logging import setup_logger, get_run_id

SUMMARIZATION_PROMPT = (
    "Summarize the following news article in 2-3 sentences, "
    "focusing on the most important factual information.\n\n"
    "Article:\n{article}\n\nSummary:"
)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _load_base_model(model_path: str, adapter_path: str | None = None):
    """Load base model (bfloat16, device_map=auto) with optional LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, padding_side="left", use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        model = model.merge_and_unload()  # merge LoRA weights for clean inference

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def _generate_summaries(
    model,
    tokenizer,
    articles: list[str],
    max_new_tokens: int = 120,
    batch_size: int = 4,
) -> list[str]:
    """Batch-generate summaries for a list of articles."""
    summaries = []
    device = next(model.parameters()).device

    for start in range(0, len(articles), batch_size):
        batch_articles = articles[start : start + batch_size]
        prompts = []
        for article in batch_articles:
            prompt_text = SUMMARIZATION_PROMPT.format(article=article[:2000])
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(formatted)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic evaluation
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode only the newly generated tokens (strip prompt)
        input_len = inputs["input_ids"].shape[1]
        for out_ids in outputs:
            new_ids = out_ids[input_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            summaries.append(text)

    return summaries


# ---------------------------------------------------------------------------
# AlignScore helper
# ---------------------------------------------------------------------------

def _alignscore_batch(
    contexts: list[str],
    claims: list[str],
    ckpt_path: str,
    backbone: str = "FacebookAI/roberta-base",
    eval_mode: str = "nli_sp",
    batch_size: int = 16,
    device: str = "cuda",
) -> list[float]:
    """Compute AlignScore for a list of (context, claim) pairs."""
    # transformers ≥ 5.0 removed AdamW; patch it back so alignscore can import.
    import transformers as _tf
    if not hasattr(_tf, "AdamW"):
        import torch.optim as _optim
        _tf.AdamW = _optim.AdamW

    from alignscore import AlignScore

    scorer = AlignScore(
        model=backbone,
        batch_size=batch_size,
        device=device,
        ckpt_path=ckpt_path,
        evaluation_mode=eval_mode,
    )
    scores = scorer.score(contexts=contexts, claims=claims)
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO adapters vs baseline")
    parser.add_argument("--candidates",        type=str, required=True,
                        help="Path to candidates JSONL (has article + reference_summary)")
    parser.add_argument("--model-path",        type=str, default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--holistic-adapter",  type=str, default="outputs/dpo/holistic/adapter")
    parser.add_argument("--gca-adapter",       type=str, default="outputs/dpo/gca/adapter")
    parser.add_argument("--alignscore-ckpt",   type=str, default="models/alignscore/AlignScore-base.ckpt")
    parser.add_argument("--alignscore-backbone", type=str, default="models/roberta-base")
    parser.add_argument("--output-dir",        type=str, default="outputs/eval")
    parser.add_argument("--n-test",            type=int, default=200,
                        help="Number of articles to evaluate (sampled from candidates)")
    parser.add_argument("--max-new-tokens",    type=int, default=120)
    parser.add_argument("--gen-batch-size",    type=int, default=4)
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()

    run_id = get_run_id()
    logger = setup_logger("dpo_eval", run_id=run_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Candidates: {args.candidates}")
    logger.info(f"Test size: {args.n_test}")
    logger.info(f"Output: {output_dir}")

    # ------------------------------------------------------------------
    # Load test articles
    # ------------------------------------------------------------------
    import random
    random.seed(args.seed)

    records = []
    with open(args.candidates) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.n_test < len(records):
        records = random.sample(records, args.n_test)
    logger.info(f"Evaluating on {len(records)} articles")

    articles    = [r["article"] for r in records]
    references  = [r["reference_summary"] for r in records]
    sample_ids  = [r["sample_id"] for r in records]

    # ------------------------------------------------------------------
    # Generate summaries for each condition
    # ------------------------------------------------------------------
    conditions = {
        "baseline":      None,
        "dpo_holistic":  args.holistic_adapter,
        "dpo_gca":       args.gca_adapter,
    }

    generated: dict[str, list[str]] = {}
    for condition, adapter_path in conditions.items():
        logger.info(f"Loading model for condition: {condition} …")
        model, tokenizer = _load_base_model(args.model_path, adapter_path)
        logger.info(f"Generating summaries …")
        summaries = _generate_summaries(
            model, tokenizer, articles,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.gen_batch_size,
        )
        generated[condition] = summaries
        logger.info(f"  {condition}: {len(summaries)} summaries generated")

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save raw generations
    # ------------------------------------------------------------------
    generations_path = output_dir / "generations.jsonl"
    with open(generations_path, "w") as f:
        for i, (sid, article, ref) in enumerate(zip(sample_ids, articles, references)):
            row = {
                "sample_id": sid,
                "article":   article,
                "reference": ref,
            }
            for cond in conditions:
                row[f"summary_{cond}"] = generated[cond][i]
            f.write(json.dumps(row) + "\n")
    logger.info(f"Raw generations saved to {generations_path}")

    # ------------------------------------------------------------------
    # ROUGE + BERTScore
    # ------------------------------------------------------------------
    from rouge_score import rouge_scorer as rouge_lib
    import bert_score as bs_lib

    def _rouge(preds, refs):
        scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for p, r in zip(preds, refs):
            s = scorer.score(r, p)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        return {
            "rouge1": sum(r1) / len(r1),
            "rouge2": sum(r2) / len(r2),
            "rougeL": sum(rl) / len(rl),
            "rouge1_per_sample": r1,
            "rouge2_per_sample": r2,
            "rougeL_per_sample": rl,
        }

    def _bertscore(preds, refs):
        # Use the local roberta-base checkpoint (already on cluster at models/roberta-base/).
        # num_layers=9 is the bert_score default for roberta-base and must be set
        # explicitly when passing a local path (not looked up from the internal table).
        local_rb = str(Path(args.model_path).parent / "roberta-base")
        P, R, F1 = bs_lib.score(
            preds, refs,
            model_type=local_rb,
            num_layers=9,
            verbose=False,
        )
        return {
            "bertscore_p": P.mean().item(),
            "bertscore_r": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
            "bertscore_f1_per_sample": F1.tolist(),
        }

    metrics: dict[str, dict] = {}
    for cond, preds in generated.items():
        logger.info(f"Computing ROUGE + BERTScore for {cond} …")
        m = {}
        m.update(_rouge(preds, references))
        m.update(_bertscore(preds, references))
        metrics[cond] = m
        logger.info(f"  ROUGE-1={m['rouge1']:.4f}  ROUGE-L={m['rougeL']:.4f}  "
                    f"BERTScore-F1={m['bertscore_f1']:.4f}")

    # ------------------------------------------------------------------
    # AlignScore (factual consistency against source article, not reference)
    # ------------------------------------------------------------------
    alignscore_ckpt = args.alignscore_ckpt
    if not Path(alignscore_ckpt).exists():
        logger.warning(f"AlignScore checkpoint not found at {alignscore_ckpt} — skipping")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for cond, preds in generated.items():
            logger.info(f"Computing AlignScore for {cond} …")
            scores = _alignscore_batch(
                contexts=articles,
                claims=preds,
                ckpt_path=alignscore_ckpt,
                backbone=args.alignscore_backbone,
                device=device,
            )
            avg = sum(scores) / len(scores)
            metrics[cond]["alignscore"] = avg
            metrics[cond]["alignscore_per_sample"] = scores
            logger.info(f"  AlignScore = {avg:.4f}")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    # Summary metrics (scalar only) for clean printing; full results include per-sample arrays
    PER_SAMPLE_KEYS = {"rouge1_per_sample", "rouge2_per_sample", "rougeL_per_sample",
                       "bertscore_f1_per_sample", "alignscore_per_sample"}
    summary_metrics = {}
    for cond, m in metrics.items():
        summary_metrics[cond] = {k: v for k, v in m.items() if k not in PER_SAMPLE_KEYS}

    results = {
        "run_id":       run_id,
        "n_test":       len(records),
        "model_path":   args.model_path,
        "conditions":   metrics,   # full: includes per-sample arrays for bootstrap CI
    }
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    logger.info("\n=== Evaluation Results ===")
    header = f"{'Condition':<18}  {'ROUGE-1':>8}  {'ROUGE-2':>8}  {'ROUGE-L':>8}  {'BERTScore-F1':>12}  {'AlignScore':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for cond, m in summary_metrics.items():
        align = f"{m['alignscore']:.4f}" if "alignscore" in m else "    —   "
        logger.info(
            f"{cond:<18}  {m['rouge1']:>8.4f}  {m['rouge2']:>8.4f}  "
            f"{m['rougeL']:>8.4f}  {m['bertscore_f1']:>12.4f}  {align:>10}"
        )

    logger.info("=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
