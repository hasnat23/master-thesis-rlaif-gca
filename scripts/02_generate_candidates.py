#!/usr/bin/env python3
"""
Script 02: Generate candidate summary pairs.

Loads the subset, generates two candidate summaries per article using
different temperature settings, and caches the results as JSONL.

Requires GPU — run on MOGON via slurm/generate_candidates.sh.

Usage:
    python scripts/02_generate_candidates.py --config configs/generation.yaml
    python scripts/02_generate_candidates.py --config configs/generation.yaml --override batch_size=2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logging import get_run_id, setup_logger, save_run_metadata
from src.generation.candidates import generate_and_cache


def main():
    parser = argparse.ArgumentParser(description="Generate candidate summary pairs")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config_path = str(PROJECT_ROOT / args.config)
    config = get_config(config_path, args.override)

    run_id = get_run_id()
    logger = setup_logger("generate_candidates", run_id=run_id)

    logger.info(f"Config: {config}")
    logger.info("Starting candidate generation...")

    output_path = generate_and_cache(config)

    logger.info(f"Done. Output: {output_path}")

    save_run_metadata(
        run_id=run_id,
        script_name="02_generate_candidates",
        config=config,
        artifacts={"candidates": output_path},
    )


if __name__ == "__main__":
    main()
