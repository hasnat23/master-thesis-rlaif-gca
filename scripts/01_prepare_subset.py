#!/usr/bin/env python3
"""
Script 01: Prepare the CNN/DailyMail subset.

Deterministically selects N articles and saves as JSONL.
This is a CPU-only operation — run locally or on MOGON login node.

Usage:
    python scripts/01_prepare_subset.py --config configs/subset.yaml
    python scripts/01_prepare_subset.py --config configs/subset.yaml --override n_samples=50
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logging import get_run_id, setup_logger, save_run_metadata
from src.data.subset import prepare_and_save


def main():
    parser = argparse.ArgumentParser(description="Prepare CNN/DM subset")
    parser.add_argument("--config", type=str, default="configs/subset.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config_path = str(PROJECT_ROOT / args.config)
    config = get_config(config_path, args.override)

    run_id = get_run_id()
    logger = setup_logger("prepare_subset", run_id=run_id)

    logger.info(f"Config: {config}")
    logger.info("Starting subset preparation...")

    output_path = prepare_and_save(config)

    logger.info(f"Done. Output: {output_path}")

    save_run_metadata(
        run_id=run_id,
        script_name="01_prepare_subset",
        config=config,
        artifacts={"subset": output_path},
    )


if __name__ == "__main__":
    main()
