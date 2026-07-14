#!/bin/bash
#SBATCH --job-name=train_rm_scale
#SBATCH --output=slurm/train_rm_scale_%j.out
#SBATCH --error=slurm/train_rm_scale_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# MOGON Bradley-Terry RM Training — scalable 5000/10000 experiment
#
# Usage:
#   sbatch --export=ALL,SAMPLE_SIZE=5000 slurm/train_rm_scale.sh
#   sbatch --export=ALL,SAMPLE_SIZE=10000 slurm/train_rm_scale.sh
#
# Requires the matching preference files in data/preferences_${SAMPLE_SIZE}/.
# ============================================================

set -euo pipefail

: "${SAMPLE_SIZE:?Set SAMPLE_SIZE to 5000 or 10000}"
SEED="${SEED:-42}"

echo "=== RM Training Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Sample:  $SAMPLE_SIZE"
echo "Seed:    $SEED"
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

HOLISTIC="data/preferences_${SAMPLE_SIZE}/holistic_reward_preferences_${SAMPLE_SIZE}.jsonl"
GCA="data/preferences_${SAMPLE_SIZE}/gca_reward_preferences_${SAMPLE_SIZE}.jsonl"
OUTPUT_DIR="outputs/reward_models_${SAMPLE_SIZE}"

if [[ ! -f "$HOLISTIC" || ! -f "$GCA" ]]; then
    echo "ERROR: preference files not found for sample size ${SAMPLE_SIZE}"
    echo "Expected:"
    echo "  $HOLISTIC"
    echo "  $GCA"
    exit 1
fi

echo "--- Training holistic RM + GCA RM (5-fold CV, n=${SAMPLE_SIZE}) ---"
python src/reward_model/run_training.py \
    --holistic "$HOLISTIC" \
    --gca "$GCA" \
    --output-dir "$OUTPUT_DIR" \
    --backbone FacebookAI/roberta-base \
    --epochs 5 \
    --batch-size 8 \
    --lr 2e-5 \
    --max-length 512 \
    --max-article-chars 2000 \
    --kfold 5 \
    --seed "$SEED"

echo ""
echo "=== RM Training Complete ==="
echo "Date:   $(date)"
echo "Output: $OUTPUT_DIR/"
