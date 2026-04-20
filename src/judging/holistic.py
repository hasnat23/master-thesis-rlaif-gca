"""
Holistic A/B preference judging.

Formats a pairwise comparison prompt for the AI judge (GPT-4o).
The judge sees two candidate summaries and the source article,
then picks a winner with a confidence score and rationale.

Implements A/B order randomization as a reliability control.
"""

import random

from src.data.schema import CandidatePair, Judgment


HOLISTIC_JUDGE_PROMPT = """\
You are an expert judge evaluating the factual accuracy of news article summaries.

You will be given a news article and two candidate summaries (Summary A and Summary B).

Your task:
1. Determine which summary is more factually consistent with the article.
2. Provide a confidence score from 0.0 to 1.0.
3. Cite specific evidence from the article to justify your choice.

Article:
{article}

Summary A:
{summary_a}

Summary B:
{summary_b}

Respond in this exact JSON format:
{{
  "winner": "A" or "B" or "tie",
  "confidence": <float between 0.0 and 1.0>,
  "rationale": "<your reasoning, citing specific article evidence>"
}}
"""


def format_holistic_prompt(
    pair: dict,
    swap_order: bool = False,
) -> tuple[str, bool]:
    """
    Format the holistic A/B judge prompt.

    Args:
        pair: A CandidatePair dict with summary_a and summary_b.
        swap_order: If True, present B as A and A as B.

    Returns:
        (formatted_prompt, was_swapped)
    """
    article = pair["article"]
    sa = pair["summary_a"]
    sb = pair["summary_b"]

    if swap_order:
        sa, sb = sb, sa

    prompt = HOLISTIC_JUDGE_PROMPT.format(
        article=article,
        summary_a=sa,
        summary_b=sb,
    )
    return prompt, swap_order


def parse_holistic_response(
    response_text: str,
    pair_id: str,
    was_swapped: bool = False,
) -> Judgment:
    """
    Parse the judge's JSON response into a Judgment dataclass.

    If the A/B order was swapped, un-swap the winner label.
    """
    import json

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return Judgment(
            pair_id=pair_id,
            winner="tie",
            confidence=0.0,
            rationale=f"PARSE_ERROR: {response_text[:500]}",
            order_swapped=was_swapped,
        )

    winner = data.get("winner", "tie").upper()
    confidence = float(data.get("confidence", 0.0))
    rationale = data.get("rationale", "")

    # Un-swap if needed
    if was_swapped:
        if winner == "A":
            winner = "B"
        elif winner == "B":
            winner = "A"

    return Judgment(
        pair_id=pair_id,
        winner=winner,
        confidence=confidence,
        rationale=rationale,
        order_swapped=was_swapped,
    )


def should_swap(pair_index: int, seed: int = 42) -> bool:
    """Deterministic A/B order randomization."""
    rng = random.Random(seed + pair_index)
    return rng.random() < 0.5
