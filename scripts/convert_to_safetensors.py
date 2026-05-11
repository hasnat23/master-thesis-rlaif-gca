"""Convert pytorch_model.bin → model.safetensors for CogComp/bart-faithful-summary-detector.

Run on MOGON (which has torch 2.4.0):
    cd ~/thesis
    conda activate thesis_env
    python scripts/convert_to_safetensors.py
"""
import os
import sys

import torch
from safetensors.torch import save_file

MODEL_DIR = os.path.expanduser("~/thesis/models/bart-faithful-summary-detector")
src = os.path.join(MODEL_DIR, "pytorch_model.bin")
dst = os.path.join(MODEL_DIR, "model.safetensors")

if not os.path.exists(src):
    print(f"ERROR: {src} not found")
    sys.exit(1)

if os.path.exists(dst):
    print(f"Already exists: {dst}  ({os.path.getsize(dst)/1e6:.0f} MB) — skipping.")
    sys.exit(0)

print(f"Loading {src}  ({os.path.getsize(src)/1e6:.0f} MB) ...")
state_dict = torch.load(src, map_location="cpu", weights_only=True)
print(f"Loaded {len(state_dict)} tensors.")
# Clone all tensors to resolve shared memory (BART shares embed weights across encoder/decoder)
state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
print("Saving as safetensors (with cloned tensors to handle shared weights) ...")
save_file(state_dict, dst)
print(f"Saved {dst}  ({os.path.getsize(dst)/1e6:.0f} MB)")
print("Done. You can now run build_reward_preferences.py with --model-name models/bart-faithful-summary-detector")
