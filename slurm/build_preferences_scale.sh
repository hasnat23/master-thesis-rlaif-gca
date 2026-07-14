#!/bin/bash
#SBATCH --job-name=prefs_scale
#SBATCH --output=slurm/prefs_scale_%j.out
#SBATCH --error=slurm/prefs_scale_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# MOGON Preference Construction — scalable 5000/10000 experiment
#
# Usage:
#   sbatch --export=ALL,SAMPLE_SIZE=5000 slurm/build_preferences_scale.sh
#   sbatch --export=ALL,SAMPLE_SIZE=10000 slurm/build_preferences_scale.sh
#
# Requires the matching candidates_${SAMPLE_SIZE}.jsonl file.
# ============================================================

set -euo pipefail

: "${SAMPLE_SIZE:?Set SAMPLE_SIZE to 5000 or 10000}"

echo "=== Preference Construction Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Sample:  $SAMPLE_SIZE"
echo "Date:    $(date)"
echo ""

export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

export PYTHONUNBUFFERED=1

CANDIDATES_PATH="data/candidates/candidates_${SAMPLE_SIZE}.jsonl"
PREF_DIR="data/preferences_${SAMPLE_SIZE}"

if [[ ! -f "$CANDIDATES_PATH" ]]; then
    echo "ERROR: missing candidates file: $CANDIDATES_PATH"
    exit 1
fi

mkdir -p "$PREF_DIR"

echo "--- Building preferences for ${SAMPLE_SIZE} samples ---"
python src/judging/build_reward_preferences.py \
    --candidates "$CANDIDATES_PATH" \
    --output-dir "$PREF_DIR" \
    --output-suffix "$SAMPLE_SIZE" \
    --judge-backend alignscore \
    --model-name yzha/AlignScore \
    --alignscore-backbone FacebookAI/roberta-base \
    --alignscore-filename AlignScore-base.ckpt \
    --margin 0 \
    --max-samples "$SAMPLE_SIZE" \
    --mode both \
    --alpha 0.0

echo ""
echo "=== Preference Construction Complete ==="
echo "Date: $(date)"
