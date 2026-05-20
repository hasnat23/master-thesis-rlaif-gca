#!/bin/bash
#SBATCH --job-name=prefs_rm500
#SBATCH --output=slurm/prefs_rm500_%j.out
#SBATCH --error=slurm/prefs_rm500_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# MOGON Reward-Model Preference Construction — 500-sample RM set
#
# Purpose: Score all 500 RM-training candidate pairs using AlignScore.
#   Produces two DPO-ready preference files:
#     data/preferences_rm500/holistic_reward_preferences_rm500.jsonl
#     data/preferences_rm500/gca_reward_preferences_rm500.jsonl
#
# Input:   data/candidates/candidates_rm500.jsonl
# Model:   yzha/AlignScore (AlignScore-base, FacebookAI/roberta-base)
# Est.:    ~8 minutes on A100
#
# Dependency: run after generate_candidates_rm500.sh completes.
# Submit:  sbatch slurm/build_reward_preferences_rm500.sh
# ============================================================

echo "=== RM-500 Reward Preference Construction Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:    $(date)"
echo ""

export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
echo "HF_ENDPOINT: $HF_ENDPOINT"

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Working directory: $(pwd)"
echo "Python:            $(which python)"
echo ""

export PYTHONUNBUFFERED=1

echo "--- Building reward-model preferences (holistic + GCA, 500 samples) ---"
python src/judging/build_reward_preferences.py \
    --candidates data/candidates/candidates_rm500.jsonl \
    --output-dir data/preferences_rm500 \
    --judge-backend alignscore \
    --model-name yzha/AlignScore \
    --alignscore-backbone FacebookAI/roberta-base \
    --alignscore-filename AlignScore-base.ckpt \
    --margin 0.05 \
    --max-samples 495 \
    --mode both \
    --alpha 0.5

echo ""
echo "=== RM-500 Reward Preference Construction Complete ==="
echo "Date: $(date)"
