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


def select_disjoint_subset(
    n_samples: int,
    exclude_seed: int = 200,
    exclude_n: int = 10000,
    new_seed: int = 999,
    split: str = "test",
    max_article_chars: int = 8000,
    max_summary_chars: int = 2000,
    local_dataset_path: str | None = None,
) -> list[SubsetSample]:
    """
    Select a subset guaranteed disjoint from an earlier seeded selection.

    Every existing experiment subset (n=1,000 / 5,000 / 10,000) was drawn with
    seed=200 from the same eligible pool, and is therefore a subset of the
    n=10,000 draw. Rather than trusting a stored file to say which articles
    that draw used, this reconstructs the exact same `rng.sample` call
    `select_subset` would have made, removes those indices from the eligible
    pool, and draws `n_samples` from what remains using `new_seed`. Used to
    build a ground-truth set that no reward model in this project could have
    been trained or cross-validated on.
    """
    if local_dataset_path:
        print(f"Loading dataset from local path: {local_dataset_path}")
        dataset = load_from_disk(local_dataset_path)
    else:
        print(f"Loading CNN/DailyMail split='{split}'...")
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)

    eligible = []
    for i, row in enumerate(dataset):
        article = row["article"]
        summary = row["highlights"]
        if len(article) <= max_article_chars and len(summary) <= max_summary_chars:
            eligible.append((i, article, summary))
    print(f"  Total in split: {len(dataset)}")
    print(f"  After length filter: {len(eligible)}")

    rng_used = random.Random(exclude_seed)
    used_n = min(exclude_n, len(eligible))
    used_indices = set(rng_used.sample(range(len(eligible)), used_n))

    remaining = [idx for idx in range(len(eligible)) if idx not in used_indices]
    print(f"  Already used by seed={exclude_seed}/n={exclude_n}: {len(used_indices)}")
    print(f"  Disjoint remainder available: {len(remaining)}")

    if len(remaining) < n_samples:
        raise ValueError(
            f"Only {len(remaining)} disjoint samples available, requested {n_samples}"
        )

    rng_new = random.Random(new_seed)
    selected = sorted(rng_new.sample(remaining, n_samples))

    samples = []
    for idx in selected:
        orig_idx, article, summary = eligible[idx]
        samples.append(SubsetSample(
            sample_id=make_sample_id(article, orig_idx),
            article=article,
            reference_summary=summary,
            split=split,
        ))
    print(f"  Selected: {len(samples)} disjoint samples")
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
