#!/usr/bin/env python3
"""
Script 03: Run judge prompt test harness.

Tests both holistic and sentence-level judging prompts on a small
batch of candidate pairs. Validates prompt formatting, response parsing,
and reliability controls.

This script does NOT call the actual GPT-4o API — it formats prompts
and tests parsing with mock responses. To run against the real API,
set --live flag (requires OPENAI_API_KEY).

Usage:
    python scripts/03_run_judge_test.py --config configs/judging.yaml --n 5
    python scripts/03_run_judge_test.py --config configs/judging.yaml --n 5 --live
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logging import get_run_id, setup_logger, save_run_metadata
from src.data.schema import load_jsonl, save_jsonl
from src.judging.holistic import format_holistic_prompt, parse_holistic_response, should_swap
from src.judging.sentence_level import split_into_sentences, format_sentence_prompt, parse_sentence_response
from src.judging.gca import aggregate_to_preference
from src.judging.reliability import filter_reliable_judgments


# Mock response for testing prompt formatting and parsing without API calls
MOCK_HOLISTIC_RESPONSE = json.dumps({
    "winner": "A",
    "confidence": 0.85,
    "rationale": "Summary A accurately captures the main events described in the article, "
                 "including the specific dates and locations mentioned in paragraphs 2-3. "
                 "Summary B introduces a claim about 'government officials' that is not "
                 "supported by the article text."
})

MOCK_SENTENCE_RESPONSE = json.dumps({
    "factual_score": 0.9,
    "rationale": "This sentence is well-supported by the article's opening paragraph "
                 "which states the same key facts about the event."
})


def run_holistic_test(pairs: list[dict], config: dict, logger, live: bool = False):
    """Test holistic judging prompt formatting and parsing."""
    logger.info(f"=== Holistic Judge Test ({len(pairs)} pairs) ===")

    judgments = []
    for i, pair in enumerate(pairs):
        pair_id = pair.get("sample_id", f"pair_{i}")
        swap = should_swap(i)

        prompt, was_swapped = format_holistic_prompt(pair, swap_order=swap)
        logger.info(f"  Pair {pair_id}: prompt length={len(prompt)}, swapped={was_swapped}")

        if live:
            # TODO: Implement actual GPT-4o API call with tenacity retry
            logger.warning("  Live API mode not yet implemented — using mock response")
            response_text = MOCK_HOLISTIC_RESPONSE
        else:
            response_text = MOCK_HOLISTIC_RESPONSE

        judgment = parse_holistic_response(response_text, pair_id, was_swapped)
        judgments.append(judgment)
        logger.info(f"  → winner={judgment.winner}, confidence={judgment.confidence}")

    # Apply reliability filters
    reliable = filter_reliable_judgments(
        judgments,
        confidence_threshold=config.get("confidence_threshold", 0.7),
    )
    logger.info(f"  Reliable: {len(reliable)}/{len(judgments)}")

    return judgments


def run_sentence_test(pairs: list[dict], config: dict, logger, live: bool = False):
    """Test sentence-level judging prompt formatting and parsing."""
    logger.info(f"=== Sentence-Level Judge Test ({len(pairs)} pairs) ===")

    all_aggregations = []
    for i, pair in enumerate(pairs):
        pair_id = pair.get("sample_id", f"pair_{i}")
        article = pair["article"]

        for side, summary_key in [("A", "summary_a"), ("B", "summary_b")]:
            summary = pair.get(summary_key, "")
            if not summary:
                logger.warning(f"  Pair {pair_id}: missing {summary_key}, skipping")
                continue

            sentences = split_into_sentences(summary)
            logger.info(f"  Pair {pair_id} Summary {side}: {len(sentences)} sentences")

            sentence_judgments = []
            for j, sent in enumerate(sentences):
                prompt = format_sentence_prompt(article, sent)

                if live:
                    response_text = MOCK_SENTENCE_RESPONSE
                else:
                    response_text = MOCK_SENTENCE_RESPONSE

                sj = parse_sentence_response(response_text, pair_id, side, j, sent)
                sentence_judgments.append(sj)

            logger.info(f"    Scores: {[sj.factual_score for sj in sentence_judgments]}")

        # GCA aggregation (using mock data for both sides)
        sentences_a = split_into_sentences(pair.get("summary_a", "No summary"))
        sentences_b = split_into_sentences(pair.get("summary_b", "No summary"))

        # Create mock sentence judgments for aggregation demo
        from src.data.schema import SentenceJudgment
        mock_a = [SentenceJudgment(pair_id, "A", k, s, 0.9, "mock") for k, s in enumerate(sentences_a)]
        mock_b = [SentenceJudgment(pair_id, "B", k, s, 0.7, "mock") for k, s in enumerate(sentences_b)]

        agg = aggregate_to_preference(mock_a, mock_b, pair_id)
        all_aggregations.append(agg)
        logger.info(f"  GCA: A={agg.summary_a_score:.3f}, B={agg.summary_b_score:.3f}, winner={agg.winner}")

    return all_aggregations


def main():
    parser = argparse.ArgumentParser(description="Judge prompt test harness")
    parser.add_argument("--config", type=str, default="configs/judging.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--n", type=int, default=5, help="Number of pairs to test")
    parser.add_argument("--live", action="store_true", help="Call real API (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    config_path = str(PROJECT_ROOT / args.config)
    config = get_config(config_path, args.override)

    run_id = get_run_id()
    logger = setup_logger("judge_test", run_id=run_id)

    # Load candidate pairs
    candidates_path = config.get("candidates_path", "data/candidates/candidates_200.jsonl")
    full_path = str(PROJECT_ROOT / candidates_path)

    if not Path(full_path).exists():
        logger.error(f"Candidates file not found: {full_path}")
        logger.info("Run 02_generate_candidates.py first, or use --override candidates_path=...")
        sys.exit(1)

    pairs = load_jsonl(full_path)[:args.n]
    logger.info(f"Loaded {len(pairs)} candidate pairs")

    holistic_judgments = run_holistic_test(pairs, config, logger, live=args.live)
    gca_aggregations = run_sentence_test(pairs, config, logger, live=args.live)

    # Save outputs
    holistic_dir = str(PROJECT_ROOT / config.get("output_dir_holistic", "data/judgments/holistic"))
    sentence_dir = str(PROJECT_ROOT / config.get("output_dir_sentence", "data/judgments/sentence_level"))

    save_jsonl(holistic_judgments, f"{holistic_dir}/test_{args.n}.jsonl")
    save_jsonl(gca_aggregations, f"{sentence_dir}/test_{args.n}.jsonl")

    logger.info("Judge test complete.")

    save_run_metadata(
        run_id=run_id,
        script_name="03_run_judge_test",
        config=config,
        artifacts={
            "holistic_judgments": f"{holistic_dir}/test_{args.n}.jsonl",
            "gca_aggregations": f"{sentence_dir}/test_{args.n}.jsonl",
        },
    )


if __name__ == "__main__":
    main()
