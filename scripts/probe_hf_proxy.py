#!/usr/bin/env python3
"""Quick check that required HF models are accessible via MOGON proxy."""
import os
import sys
from huggingface_hub import hf_hub_download

pairs = [
    ("yzha/AlignScore", "AlignScore-base.ckpt"),
    ("FacebookAI/roberta-base", "config.json"),
    ("CogComp/bart-faithful-summary-detector", "config.json"),
]

all_ok = True
for repo, fname in pairs:
    try:
        path = hf_hub_download(repo_id=repo, filename=fname)
        print(f"OK   {repo}  →  {path}")
    except Exception as exc:
        print(f"FAIL {repo}  →  {type(exc).__name__}: {str(exc)[:200]}")
        all_ok = False

sys.exit(0 if all_ok else 1)
