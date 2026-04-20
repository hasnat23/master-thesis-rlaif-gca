"""
Reliability controls for AI judging.

Implements the reliability measures specified in the thesis proposal:
1. Confidence gating — filter judgments below a confidence threshold
2. A/B order randomization — run each pair twice with swapped order, discard disagreements
3. Evidence-grounded rationale check — verify the judge cites article evidence
"""

from src.data.schema import Judgment


def passes_confidence_gate(judgment: Judgment, threshold: float = 0.7) -> bool:
    """Check if the judgment meets the minimum confidence threshold."""
    return judgment.confidence >= threshold


def check_swap_consistency(
    judgment_original: Judgment,
    judgment_swapped: Judgment,
) -> bool:
    """
    Check if the judge gives the same winner regardless of A/B order.

    The swapped judgment should already have its winner un-swapped
    (done in parse_holistic_response). So both should agree.
    """
    return judgment_original.winner == judgment_swapped.winner


def has_evidence_rationale(judgment: Judgment, min_rationale_length: int = 20) -> bool:
    """
    Basic check that the rationale is substantive.

    A proper rationale should cite specific evidence from the article.
    This is a minimal length check — a more sophisticated version
    could check for article quote overlap.
    """
    if not judgment.rationale:
        return False
    if judgment.rationale.startswith("PARSE_ERROR"):
        return False
    return len(judgment.rationale) >= min_rationale_length


def filter_reliable_judgments(
    judgments: list[Judgment],
    confidence_threshold: float = 0.7,
    min_rationale_length: int = 20,
) -> list[Judgment]:
    """
    Apply all reliability filters to a list of judgments.

    Returns only judgments that pass confidence gating and have
    substantive rationales.
    """
    reliable = []
    for j in judgments:
        if not passes_confidence_gate(j, confidence_threshold):
            continue
        if not has_evidence_rationale(j, min_rationale_length):
            continue
        reliable.append(j)
    return reliable
