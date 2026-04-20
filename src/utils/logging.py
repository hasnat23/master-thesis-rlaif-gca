"""
Run logger for the thesis pipeline.

Produces structured JSON logs with timestamps, run IDs, and artifact paths.
"""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import PROJECT_ROOT


def get_run_id() -> str:
    """Generate a short unique run ID."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def setup_logger(name: str, log_dir: str | None = None, run_id: str | None = None) -> logging.Logger:
    """
    Set up a logger that writes to both console and a JSON log file.

    Args:
        name: Logger name (usually the script name).
        log_dir: Directory to write log files. Defaults to outputs/logs/.
        run_id: Unique run identifier. Auto-generated if not provided.
    """
    if run_id is None:
        run_id = get_run_id()

    if log_dir is None:
        log_dir = str(PROJECT_ROOT / "outputs" / "logs")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    fmt = logging.Formatter(f"[{run_id}] %(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_path / f"{name}_{run_id}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def save_run_metadata(
    run_id: str,
    script_name: str,
    config: dict,
    artifacts: dict[str, str] | None = None,
    metrics: dict | None = None,
    output_dir: str | None = None,
) -> str:
    """
    Save a JSON metadata file for this run.

    Returns the path to the saved file.
    """
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "outputs" / "logs")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "script": script_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "artifacts": artifacts or {},
        "metrics": metrics or {},
    }

    filepath = out_path / f"run_{run_id}.json"
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return str(filepath)
