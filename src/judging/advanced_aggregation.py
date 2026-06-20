"""
Advanced aggregation strategies for GCA.

After finding that simple mean outperforms penalized mean, we test hybrid approaches:
1. Weighted aggregation: favor more important sentences
2. Confidence-weighted: combine with score confidence/variance
3. Sentence importance: weight by length or position
4. Ensemble: combine multiple aggregation methods
"""

from typing import List


def aggregate_weighted_importance(scores: List[float]) -> float:
    """
    Weighted mean based on position: later sentences get higher weight.
    Intuition: conclusions in later sentences often carry more weight.
    """
    if not scores:
        return 0.0
    
    n = len(scores)
    weights = [(i + 1) / n for i in range(n)]  # Linear increasing weight
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / sum(weights)


def aggregate_confidence_weighted(scores: List[float]) -> float:
    """
    Weight scores by confidence (inverse variance).
    Consistent scores (low variance) get higher weight than noisy ones.
    """
    if not scores:
        return 0.0
    
    if len(scores) == 1:
        return scores[0]
    
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    
    # If variance is very low, all scores are consistent - high confidence
    if variance < 0.01:
        return mean_score
    
    # Weight each score inversely by its distance from mean
    # Closer to mean = more confident = higher weight
    weights = [1.0 / (1.0 + abs(s - mean_score)) for s in scores]
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / sum(weights)


def aggregate_length_weighted(scores: List[float], sentence_lengths: List[int]) -> float:
    """
    Weight scores by sentence length.
    Longer sentences (more content) get higher weight.
    Requires sentence_lengths to be provided.
    """
    if not scores or not sentence_lengths:
        return sum(scores) / len(scores) if scores else 0.0
    
    if len(scores) != len(sentence_lengths):
        return sum(scores) / len(scores)
    
    total_length = sum(sentence_lengths)
    if total_length == 0:
        return sum(scores) / len(scores)
    
    weights = [length / total_length for length in sentence_lengths]
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return weighted_sum / sum(weights)


def aggregate_robust_mean(scores: List[float]) -> float:
    """
    Robust mean: ignore outliers (top and bottom 10%).
    More robust to scoring noise than simple mean.
    """
    if not scores or len(scores) < 3:
        return sum(scores) / len(scores) if scores else 0.0
    
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    
    # Remove bottom 10% and top 10%
    exclude_count = max(1, n // 10)
    trimmed = sorted_scores[exclude_count:n-exclude_count]
    
    if not trimmed:
        return sum(scores) / len(scores)
    
    return sum(trimmed) / len(trimmed)


def aggregate_harmonic_mean(scores: List[float]) -> float:
    """
    Harmonic mean: penalizes low scores more than arithmetic mean.
    More conservative than simple mean.
    """
    if not scores:
        return 0.0
    
    if any(s <= 0 for s in scores):
        # If any score is 0, harmonic mean is 0
        return 0.0 if any(s == 0 for s in scores) else sum(scores) / len(scores)
    
    n = len(scores)
    return n / sum(1.0 / s for s in scores)


def aggregate_ensemble(scores: List[float]) -> float:
    """
    Ensemble: average of multiple aggregation methods.
    More robust than single method.
    """
    methods = [
        sum(scores) / len(scores),  # Simple mean
        aggregate_robust_mean(scores),  # Robust mean
        aggregate_harmonic_mean(scores),  # Harmonic mean
    ]
    
    # Filter out invalid values
    valid_methods = [m for m in methods if m is not None and 0 <= m <= 1]
    
    if not valid_methods:
        return sum(scores) / len(scores) if scores else 0.0
    
    return sum(valid_methods) / len(valid_methods)


if __name__ == "__main__":
    # Test on sample data
    test_scores = [0.8, 0.75, 0.9, 0.7, 0.85]
    test_lengths = [50, 40, 60, 35, 55]
    
    print("Test scores:", test_scores)
    print("Test lengths:", test_lengths)
    print()
    
    print("Simple mean:", f"{sum(test_scores)/len(test_scores):.4f}")
    print("Weighted by position:", f"{aggregate_weighted_importance(test_scores):.4f}")
    print("Confidence weighted:", f"{aggregate_confidence_weighted(test_scores):.4f}")
    print("Length weighted:", f"{aggregate_length_weighted(test_scores, test_lengths):.4f}")
    print("Robust mean (trim 10%):", f"{aggregate_robust_mean(test_scores):.4f}")
    print("Harmonic mean:", f"{aggregate_harmonic_mean(test_scores):.4f}")
    print("Ensemble:", f"{aggregate_ensemble(test_scores):.4f}")
