"""
Candidate summary generation.

Generates candidate summary pairs for each article in the subset.
Uses an instruction-tuned model (7B class, per proposal) with two
different temperature settings to produce diverse candidates.

The exact model is configured via configs/generation.yaml.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.schema import CandidatePair, SubsetSample, load_jsonl, save_jsonl


SUMMARIZATION_PROMPT = (
    "Summarize the following news article in a concise paragraph.\n\n"
    "Article:\n{article}\n\n"
    "Summary:"
)


def load_model_and_tokenizer(config: dict):
    """
    Load the generation model and tokenizer.

    Applies:
    - 4-bit quantization if configured (for 7B models on single GPU)
    - Left padding for decoder-only models
    """
    model_name = config["model_name"]
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Fix tokenizer padding (known issue from progress updates)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config.get("padding_side", "left")

    # Quantization config
    quantization_config = None
    if config.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()

    return model, tokenizer


def generate_single(
    model,
    tokenizer,
    article: str,
    gen_config: dict,
    max_input_length: int = 1024,
    max_new_tokens: int = 256,
) -> str:
    """Generate a single candidate summary for one article."""
    prompt = SUMMARIZATION_PROMPT.format(article=article)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.9),
            do_sample=gen_config.get("do_sample", True),
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (skip the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return summary


def generate_candidate_pairs(
    model,
    tokenizer,
    samples: list[dict],
    config: dict,
) -> list[CandidatePair]:
    """
    Generate candidate pairs for a list of subset samples.

    Each article gets two summaries generated with different temperature settings.
    """
    gen_configs = config.get("generation_configs", [
        {"name": "low_temp", "temperature": 0.7, "top_p": 0.9, "do_sample": True},
        {"name": "high_temp", "temperature": 1.0, "top_p": 0.95, "do_sample": True},
    ])

    if len(gen_configs) < 2:
        raise ValueError("Need at least 2 generation configs for candidate pairs")

    max_input_length = config.get("max_input_length", 1024)
    max_new_tokens = config.get("max_new_tokens", 256)

    pairs = []
    for i, sample in enumerate(samples):
        article = sample["article"]
        ref = sample.get("reference_summary", "")
        sample_id = sample.get("sample_id", f"sample_{i:05d}")

        print(f"  [{i+1}/{len(samples)}] Generating candidates for {sample_id}...")

        summary_a = generate_single(
            model, tokenizer, article, gen_configs[0],
            max_input_length, max_new_tokens,
        )
        summary_b = generate_single(
            model, tokenizer, article, gen_configs[1],
            max_input_length, max_new_tokens,
        )

        pair = CandidatePair(
            sample_id=sample_id,
            article=article,
            reference_summary=ref,
            summary_a=summary_a,
            summary_b=summary_b,
            generation_params_a=gen_configs[0],
            generation_params_b=gen_configs[1],
        )
        pairs.append(pair)

    return pairs


def generate_and_cache(config: dict) -> str:
    """Full pipeline: load model, generate pairs, cache to disk. Returns output path."""
    model, tokenizer = load_model_and_tokenizer(config)

    subset_path = config.get("subset_path", "data/subset/subset_200.jsonl")
    samples = load_jsonl(subset_path)
    print(f"Loaded {len(samples)} samples from {subset_path}")

    pairs = generate_candidate_pairs(model, tokenizer, samples, config)

    output_dir = config.get("output_dir", "data/candidates")
    output_filename = config.get("output_filename", "candidates_200.jsonl")
    output_path = str(Path(output_dir) / output_filename)

    count = save_jsonl(pairs, output_path)
    print(f"Cached {count} candidate pairs to {output_path}")

    return output_path
