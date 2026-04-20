"""
Configuration loader for the thesis pipeline.

Loads YAML config files and merges with CLI overrides.
"""

import yaml
import argparse
from pathlib import Path
from typing import Any


def load_yaml(config_path: str) -> dict:
    """Load a YAML config file and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_cli_overrides(config: dict, overrides: list[str]) -> dict:
    """
    Merge CLI overrides into config dict.

    Overrides are in the form key=value. Nested keys use dot notation:
        subset.n_samples=100
    """
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to cast to int, float, bool
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        d[keys[-1]] = value
    return config


def get_config(config_path: str, cli_overrides: list[str] | None = None) -> dict:
    """Load config and apply any CLI overrides."""
    config = load_yaml(config_path)
    if cli_overrides:
        config = merge_cli_overrides(config, cli_overrides)
    return config


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add standard config arguments to an argparse parser."""
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides: key=value")
    return parser


def resolve_paths(config: dict, project_root: Path) -> dict:
    """Resolve relative paths in config against project root."""
    for key in ("output_dir", "data_dir", "cache_dir", "log_dir"):
        if key in config:
            config[key] = str(project_root / config[key])
    return config


# Project root: two levels up from src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
