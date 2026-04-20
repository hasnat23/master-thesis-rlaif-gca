"""
Evaluation metrics for summarization.

Computes ROUGE-1/2/L and BERTScore F1 for generated summaries
against reference summaries.
"""

from typing import Optional


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Args:
        predictions: List of generated summaries.
        references: List of reference summaries.

    Returns:
        Dict with rouge1, rouge2, rougeL keys, each containing
        precision, recall, fmeasure averages.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    # Average
    avg_scores = {}
    for key, vals in scores.items():
        avg_scores[key] = sum(vals) / len(vals) if vals else 0.0

    return avg_scores


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict:
    """
    Compute BERTScore F1.

    Args:
        predictions: List of generated summaries.
        references: List of reference summaries.
        model_type: Model to use for BERTScore computation.

    Returns:
        Dict with precision, recall, f1 averages.
    """
    from bert_score import score as bert_score

    P, R, F1 = bert_score(
        predictions,
        references,
        model_type=model_type,
        verbose=False,
    )

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    compute_bert: bool = True,
) -> dict:
    """Compute all metrics and return a single dict."""
    metrics = compute_rouge(predictions, references)

    if compute_bert:
        bert_metrics = compute_bertscore(predictions, references)
        metrics.update(bert_metrics)

    return metrics
