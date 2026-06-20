#!/bin/bash
#SBATCH --job-name=prefs_1000
#SBATCH --output=slurm/prefs_1000_%j.out
#SBATCH --error=slurm/prefs_1000_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# MOGON Preference Construction — 1000-sample, NO margin filter
#
# Prof. feedback (02-06-2026): remove the margin-based preference
# construction step (steps 5A/7B). Use ALL candidate pairs, not
# just those with score difference > 0.05.
#
# Setting --margin 0 means any score_A > score_B gives "A wins",
# recovering the 23% of pairs previously discarded as no_preference.
#
# Input:   data/candidates/candidates_1000.jsonl
# Output:  data/preferences_1000/holistic_preferences_1000.jsonl
#          data/preferences_1000/gca_preferences_1000.jsonl
# Est.:    ~15-20 min on A100
#
# Dependency: run after generate_candidates_1000.sh completes.
# Submit:  sbatch slurm/build_preferences_1000.sh
# ============================================================

echo "=== 1000-Sample Preference Construction (no margin) Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:    $(date)"
echo ""

export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
echo "HF_ENDPOINT: $HF_ENDPOINT"

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

export PYTHONUNBUFFERED=1

# Validate input
if [[ ! -f data/candidates/candidates_1000.jsonl ]]; then
    echo "ERROR: candidates_1000.jsonl not found. Run generate_candidates_1000.sh first."
    exit 1
fi
WC=$(wc -l < data/candidates/candidates_1000.jsonl)
echo "Input: $WC candidate pairs"

mkdir -p data/preferences_1000

echo "--- Building preferences (holistic + GCA, margin=0) ---"
# Note: AdamW patch for alignscore is applied in src/judging/reward_model_judge.py
python src/judging/build_reward_preferences.py \
    --candidates data/candidates/candidates_1000.jsonl \
    --output-dir data/preferences_1000 \
    --judge-backend alignscore \
    --model-name yzha/AlignScore \
    --alignscore-backbone FacebookAI/roberta-base \
    --alignscore-filename AlignScore-base.ckpt \
    --margin 0 \
    --max-samples 1000 \
    --mode both \
    --alpha 0.5

echo ""
echo "--- Step check: decision distribution ---"
python3 -c "
import json
from collections import Counter
for tag, path in [('holistic', 'data/preferences_1000/holistic_reward_preferences_1000.jsonl'),
                   ('gca',      'data/preferences_1000/gca_reward_preferences_1000.jsonl')]:
    try:
        rows = [json.loads(l) for l in open(path)]
        c = Counter(r['decision'] for r in rows)
        usable = c['A'] + c['B']
        print(f'{tag}: total={len(rows)} usable={usable} ({usable/len(rows):.1%}) no_pref={c[\"no_preference\"]}')
    except FileNotFoundError:
        print(f'{tag}: file not found')
"

echo "=== Preference Construction Complete ==="
echo "Date: $(date)"
