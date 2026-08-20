#!/bin/bash
#SBATCH --job-name=eval_gt_curve
#SBATCH --output=slurm/logs/eval_gt_curve_%j.out
#SBATCH --error=slurm/logs/eval_gt_curve_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# Run all four ground-truth precision conditions against a
# curve-point scale's checkpoints (5 seeds, not 20 -- see
# retrain_for_ground_truth_curve.sh for why).
#
# Usage: sbatch --export=SCALE=2000 slurm/evaluate_ground_truth_curve.sh
#        sbatch --export=SCALE=3000 slurm/evaluate_ground_truth_curve.sh
# ============================================================

set -euo pipefail

: "${SCALE:?Must set SCALE=2000 or SCALE=3000 via --export}"
SEEDS="${SEEDS:-1-5}"

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

CKPT_ROOT="outputs/ground_truth_eval_${SCALE}"

echo "=== Ground-truth evaluation, scale=${SCALE} (seeds=${SEEDS}) ==="
echo "Checkpoints: ${CKPT_ROOT}"
echo "Date: $(date)"

python analysis/evaluate_ground_truth.py \
    --ground-truth data/preferences_groundtruth/gca_reward_preferences_groundtruth.jsonl \
    --checkpoints-root "$CKPT_ROOT" \
    --seeds "$SEEDS" \
    --out "reports/campaigns/ground_truth_eval_${SCALE}.json"

python analysis/evaluate_ground_truth.py \
    --ground-truth data/preferences_groundtruth/biased_ground_truth.jsonl \
    --checkpoints-root "$CKPT_ROOT" \
    --seeds "$SEEDS" \
    --out "reports/campaigns/biased_ground_truth_eval_${SCALE}.json"

python analysis/evaluate_ground_truth.py \
    --ground-truth data/preferences_groundtruth/biased_ground_truth_top10.jsonl \
    --checkpoints-root "$CKPT_ROOT" \
    --seeds "$SEEDS" \
    --out "reports/campaigns/biased_top10_eval_${SCALE}.json"

python analysis/evaluate_ground_truth.py \
    --ground-truth data/preferences_groundtruth/biased_ground_truth_top5_allpairs.jsonl \
    --checkpoints-root "$CKPT_ROOT" \
    --seeds "$SEEDS" \
    --out "reports/campaigns/biased_top5_allpairs_eval_${SCALE}.json"

echo "=== Done, scale=${SCALE} ==="
echo "Date: $(date)"
