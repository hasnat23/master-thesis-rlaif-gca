#!/bin/bash
#SBATCH --job-name=retrain_gt5k
#SBATCH --output=slurm/logs/retrain_gt5k_%A_%a.out
#SBATCH --error=slurm/logs/retrain_gt5k_%A_%a.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:15:00
#SBATCH --array=1-20
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# Retrain holistic + GCA reward models at n=5,000, checkpoints saved
#
# Same as retrain_for_ground_truth.sh (n=1,000) but at n=5,000, to check
# whether the ground-truth precision advantage found at n=1,000 follows
# the same scale-dependent pattern already established for learnability
# (thesis Chapter 6: significant at n=1,000, absent at n=5,000/10,000).
#
# The held-out ground-truth set (data/subset/subset_groundtruth.jsonl,
# 500 articles) is disjoint from n=5,000 for the same reason it is disjoint
# from n=1,000: both are seed=200 subsets nested inside the n=10,000 draw,
# and select_disjoint_subset excluded against that full n=10,000 draw
# specifically. No ground-truth rebuild needed.
#
# Single full-data training run per condition (no --kfold), ~27 min
# expected based on 1/5 of the 5-fold n=5,000 seed-campaign runtime
# (2h14m / 5 folds).
#
# Submit: sbatch slurm/retrain_for_ground_truth_5000.sh
# ============================================================

set -euo pipefail

SEED="${SLURM_ARRAY_TASK_ID}"

echo "=== Ground-truth retrain (n=5000): seed=${SEED} ==="
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

HOLISTIC="data/preferences_5000/holistic_reward_preferences_5000.jsonl"
GCA="data/preferences_5000/gca_reward_preferences_5000.jsonl"
OUT="outputs/ground_truth_eval_5000/seed_${SEED}"

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
