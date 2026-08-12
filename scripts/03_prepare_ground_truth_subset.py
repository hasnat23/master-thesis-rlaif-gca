#!/usr/bin/env python3
"""
Script 03: Prepare the ground-truth evaluation subset.

Selects articles guaranteed disjoint from every training subset used so far
(all of which are seed=200 draws from the test split, nested inside the
n=10,000 sample per thesis/chapters/04_methodology.tex Section 4.6). This is
a CPU-only operation — run locally or on the MOGON login node.

Usage:
    python scripts/03_prepare_ground_truth_subset.py --config configs/subset_groundtruth.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logging import get_run_id, setup_logger, save_run_metadata
from src.data.subset import select_disjoint_subset
from src.data.schema import save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Prepare disjoint ground-truth subset")
    parser.add_argument("--config", type=str, default="configs/subset_groundtruth.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config_path = str(PROJECT_ROOT / args.config)
    config = get_config(config_path, args.override)

    run_id = get_run_id()
    logger = setup_logger("prepare_ground_truth_subset", run_id=run_id)
    logger.info(f"Config: {config}")

    samples = select_disjoint_subset(
        n_samples=config.get("n_samples", 500),
        exclude_seed=config.get("exclude_seed", 200),
        exclude_n=config.get("exclude_n", 10000),
        new_seed=config.get("new_seed", 999),
        split=config.get("split", "test"),
        max_article_chars=config.get("max_article_chars", 8000),
        max_summary_chars=config.get("max_summary_chars", 2000),
        local_dataset_path=config.get("local_dataset_path"),
    )

    output_dir = config.get("output_dir", "data/subset")
    output_filename = config.get("output_filename", "subset_groundtruth.jsonl")
    output_path = str(Path(output_dir) / output_filename)
    count = save_jsonl(samples, output_path)
    logger.info(f"Saved {count} samples to {output_path}")

    # Sanity check: zero overlap with the n=10,000 pool's sample IDs, checked
    # directly rather than assumed from the selection logic alone.
    used_ids_path = config.get("check_against_subset")
    if used_ids_path and Path(PROJECT_ROOT / used_ids_path).exists():
        import json
        used_ids = set()
        with open(PROJECT_ROOT / used_ids_path) as f:
            for line in f:
                used_ids.add(json.loads(line)["sample_id"])
        new_ids = {s.sample_id for s in samples}
        overlap = new_ids & used_ids
        if overlap:
            raise RuntimeError(f"FOUND {len(overlap)} OVERLAPPING sample IDs — not disjoint!")
        logger.info(f"Verified disjoint against {used_ids_path}: 0 overlap with {len(used_ids)} IDs.")

    save_run_metadata(
        run_id=run_id,
        script_name="03_prepare_ground_truth_subset",
        config=config,
        artifacts={"subset": output_path},
    )


if __name__ == "__main__":
    main()
