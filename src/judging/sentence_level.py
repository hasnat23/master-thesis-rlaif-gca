"""
Sentence-level factual judgment.

Splits a candidate summary into sentences, then prompts the AI judge
to evaluate the factual accuracy of each sentence against the source article.

This is the input to the GCA aggregation step.
"""

import re

from src.data.schema import SentenceJudgment


SENTENCE_JUDGE_PROMPT = """\
You are an expert judge evaluating the factual accuracy of individual sentences
in a news article summary.

You will be given a news article and a single sentence from a summary.

Your task:
1. Rate how factually consistent this sentence is with the article (0.0 to 1.0).
   - 1.0 = fully supported by the article
   - 0.5 = partially supported or vague
   - 0.0 = contradicts the article or is fabricated
2. Cite the specific evidence from the article that supports or contradicts the sentence.

Article:
{article}

Sentence to evaluate:
{sentence}

Respond in this exact JSON format:
{{
  "factual_score": <float between 0.0 and 1.0>,
  "rationale": "<your reasoning, citing specific article evidence>"
}}
"""


def split_into_sentences(text: str) -> list[str]:
    """
    Split summary text into sentences.

    Uses a simple regex-based approach. Handles common abbreviations.
    """
    # Split on sentence-ending punctuation followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings
    return [s.strip() for s in sentences if s.strip()]


def format_sentence_prompt(article: str, sentence: str) -> str:
    """Format the per-sentence judge prompt."""
    return SENTENCE_JUDGE_PROMPT.format(article=article, sentence=sentence)


def parse_sentence_response(
    response_text: str,
    pair_id: str,
    summary_side: str,
    sentence_index: int,
    sentence_text: str,
) -> SentenceJudgment:
    """Parse the judge's JSON response into a SentenceJudgment dataclass."""
    import json

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return SentenceJudgment(
            pair_id=pair_id,
            summary_side=summary_side,
            sentence_index=sentence_index,
            sentence_text=sentence_text,
            factual_score=0.0,
            rationale=f"PARSE_ERROR: {response_text[:500]}",
        )

    return SentenceJudgment(
        pair_id=pair_id,
        summary_side=summary_side,
        sentence_index=sentence_index,
        sentence_text=sentence_text,
        factual_score=float(data.get("factual_score", 0.0)),
        rationale=data.get("rationale", ""),
    )
