"""
Granular Credit Assignment (GCA) aggregation.

Aggregates per-sentence factual scores into a summary-level preference score.
This is the core mechanism that distinguishes the sentence-level RLAIF approach
from the holistic approach.
"""

from src.data.schema import SentenceJudgment, GCAAggregation


def aggregate_sentence_scores(scores: list[float], alpha: float = 0.5) -> float:
    """
    Aggregate per-sentence factual scores into a single summary score.

    Formula:
        gca_score = mean(scores) * (min(scores) / mean(scores))^alpha

    This penalizes summaries that contain even one poorly-supported sentence,
    while still rewarding overall quality. The alpha parameter controls how
    harshly the worst sentence is penalized:
    - alpha=0: equivalent to simple mean (no penalty)
    - alpha=1: strong penalty for worst sentence

    Args:
        scores: List of per-sentence factual scores (0.0 to 1.0).
        alpha: Penalty exponent for the worst sentence.

    Returns:
        Aggregated score between 0.0 and 1.0.
    """
    if not scores:
        return 0.0

    mean_score = sum(scores) / len(scores)

    if mean_score == 0.0:
        return 0.0

    min_score = min(scores)
    penalty = (min_score / mean_score) ** alpha
    return mean_score * penalty


def aggregate_to_preference(
    judgments_a: list[SentenceJudgment],
    judgments_b: list[SentenceJudgment],
    pair_id: str,
    alpha: float = 0.5,
    tie_margin: float = 0.05,
) -> GCAAggregation:
    """
    Aggregate sentence-level judgments into a summary-level preference.

    Args:
        judgments_a: Sentence judgments for summary A.
        judgments_b: Sentence judgments for summary B.
        pair_id: Identifier for this candidate pair.
        alpha: GCA penalty exponent.
        tie_margin: If score difference is below this, declare a tie.

    Returns:
        GCAAggregation with scores and winner.
    """
    scores_a = [j.factual_score for j in judgments_a]
    scores_b = [j.factual_score for j in judgments_b]

    agg_a = aggregate_sentence_scores(scores_a, alpha=alpha)
    agg_b = aggregate_sentence_scores(scores_b, alpha=alpha)

    diff = agg_a - agg_b
    if abs(diff) < tie_margin:
        winner = "tie"
    elif diff > 0:
        winner = "A"
    else:
        winner = "B"

    return GCAAggregation(
        pair_id=pair_id,
        summary_a_score=agg_a,
        summary_b_score=agg_b,
        winner=winner,
        sentence_judgments_a=[
            {"index": j.sentence_index, "score": j.factual_score, "text": j.sentence_text}
            for j in judgments_a
        ],
        sentence_judgments_b=[
            {"index": j.sentence_index, "score": j.factual_score, "text": j.sentence_text}
            for j in judgments_b
        ],
    )
