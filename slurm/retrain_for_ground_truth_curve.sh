#!/bin/bash
#SBATCH --job-name=retrain_gt_curve
#SBATCH --output=slurm/logs/retrain_gt_curve_%A_%a.out
#SBATCH --error=slurm/logs/retrain_gt_curve_%A_%a.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=1-5
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# Retrain holistic + GCA reward models at a curve-point scale
# (n=2,000 or n=3,000), checkpoints saved, 5 seeds only.
#
# Locates the crossover point between the n=1,000 GCA-wins result
# and the n=5,000 no-difference result. preferences_{SCALE} was
# built by filtering the existing preferences_10000 files by
# sample_id (n=2000/3000 are exact subsets of the n=10000 draw at
# seed=200) -- no new candidate generation or AlignScore scoring.
#
# 5 seeds, not 20: this is a supplementary curve-point check, not
# a headline result -- keep it short (professor's guidance, final
# week before submission).
#
# Usage: sbatch --export=SCALE=2000 slurm/retrain_for_ground_truth_curve.sh
#        sbatch --export=SCALE=3000 slurm/retrain_for_ground_truth_curve.sh
# ============================================================

set -euo pipefail

: "${SCALE:?Must set SCALE=2000 or SCALE=3000 via --export}"

SEED="${SLURM_ARRAY_TASK_ID}"

echo "=== Ground-truth retrain (n=${SCALE}): seed=${SEED} ==="
echo "Job:  ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Node: ${SLURM_NODELIST}"
echo "Date: $(date)"

export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

HOLISTIC="data/preferences_${SCALE}/holistic_reward_preferences_${SCALE}.jsonl"
GCA="data/preferences_${SCALE}/gca_reward_preferences_${SCALE}.jsonl"
OUT="outputs/ground_truth_eval_${SCALE}/seed_${SEED}"

if [[ ! -f "$HOLISTIC" || ! -f "$GCA" ]]; then
    echo "ERROR: preference files not found."
    exit 1
fi

if [[ -f "${OUT}/rm_training_summary.json" ]]; then
    echo "Summary already exists for seed ${SEED}; nothing to do."
    exit 0
fi

python src/reward_model/run_training.py \
    --holistic "$HOLISTIC" \
    --gca      "$GCA" \
    --output-dir "$OUT" \
    --backbone FacebookAI/roberta-base \
    --epochs 5 \
    --batch-size 8 \
    --lr 2e-5 \
    --max-length 512 \
    --max-article-chars 2000 \
    --seed "${SEED}"

echo ""
echo "=== Done: seed=${SEED} ==="
echo "Date:   $(date)"
echo "Output: ${OUT}/"
