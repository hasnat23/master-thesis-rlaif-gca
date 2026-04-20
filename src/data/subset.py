"""
CNN/DailyMail subset selection.

Deterministically selects N articles from the CNN/DailyMail dataset,
filters by length constraints, and saves as JSONL for reproducible experiments.
"""

import hashlib
import random
from pathlib import Path

from datasets import load_dataset, load_from_disk

from src.data.schema import SubsetSample, save_jsonl


def make_sample_id(article: str, index: int) -> str:
    """Create a deterministic sample ID from article content."""
    h = hashlib.sha256(article.encode("utf-8")).hexdigest()[:12]
    return f"cnn_{index:05d}_{h}"


def select_subset(
    n_samples: int = 200,
    seed: int = 42,
    split: str = "test",
    max_article_chars: int = 8000,
    max_summary_chars: int = 2000,
    local_dataset_path: str | None = None,
) -> list[SubsetSample]:
    """
    Load CNN/DailyMail and select a deterministic subset.

    Args:
        n_samples: Number of samples to select.
        seed: Random seed for reproducibility.
        split: Dataset split to sample from.
        max_article_chars: Skip articles longer than this.
        max_summary_chars: Skip summaries longer than this.
        local_dataset_path: Path to a saved HF dataset on disk (load_from_disk).

    Returns:
        List of SubsetSample dataclass instances.
    """
    if local_dataset_path:
        print(f"Loading dataset from local path: {local_dataset_path}")
        dataset = load_from_disk(local_dataset_path)
    else:
        print(f"Loading CNN/DailyMail split='{split}'...")
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)

    # Filter by length
    eligible = []
    for i, row in enumerate(dataset):
        article = row["article"]
        summary = row["highlights"]
        if len(article) <= max_article_chars and len(summary) <= max_summary_chars:
            eligible.append((i, article, summary))

    print(f"  Total in split: {len(dataset)}")
    print(f"  After length filter: {len(eligible)}")

    if len(eligible) < n_samples:
        print(f"  WARNING: Only {len(eligible)} eligible samples, requested {n_samples}")
        n_samples = len(eligible)

    # Deterministic selection
    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(eligible)), n_samples)
    selected_indices.sort()

    samples = []
    for rank, idx in enumerate(selected_indices):
        orig_idx, article, summary = eligible[idx]
        sample = SubsetSample(
            sample_id=make_sample_id(article, orig_idx),
            article=article,
            reference_summary=summary,
            split=split,
        )
        samples.append(sample)

    print(f"  Selected: {len(samples)} samples")
    return samples


def prepare_and_save(config: dict) -> str:
    """Run subset selection and save to disk. Returns output path."""
    samples = select_subset(
        n_samples=config.get("n_samples", 200),
        seed=config.get("seed", 42),
        split=config.get("split", "test"),
        max_article_chars=config.get("max_article_chars", 8000),
        max_summary_chars=config.get("max_summary_chars", 2000),
        local_dataset_path=config.get("local_dataset_path"),
    )

    output_dir = config.get("output_dir", "data/subset")
    output_filename = config.get("output_filename", "subset_200.jsonl")
    output_path = str(Path(output_dir) / output_filename)

    count = save_jsonl(samples, output_path)
    print(f"  Saved {count} samples to {output_path}")
    return output_path
