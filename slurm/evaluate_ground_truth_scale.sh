#!/bin/bash
#SBATCH --job-name=eval_gt_scale
#SBATCH --output=slurm/logs/eval_gt_scale_%j.out
#SBATCH --error=slurm/logs/eval_gt_scale_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# Run all four ground-truth precision conditions (same-article,
# top/bottom 25%, top/bottom 10%, top/bottom 5% all-pairs)
# against a single training-scale's checkpoints.
#
# Usage: sbatch --export=SCALE=5000 slurm/evaluate_ground_truth_scale.sh
#        sbatch --export=SCALE=10000 slurm/evaluate_ground_truth_scale.sh
# ============================================================

set -euo pipefail

: "${SCALE:?Must set SCALE=5000 or SCALE=10000 via --export}"

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

echo "=== Ground-truth evaluation, scale=${SCALE} ==="
echo "Checkpoints: ${CKPT_ROOT}"
echo "Date: $(date)"

python analysis/evaluate_ground_truth.py \
    --ground-truth data/preferences_groundtruth/gca_reward_preferences_groundtruth.jsonl \
    --checkpoints-root "$CKPT_ROOT" \
    --out "reports/campaigns/ground_truth_eval_${SCALE}.json"

python analysis/evaluate_ground_truth.py \
    --ground-truth data/preferences_groundtruth/biased_ground_truth.jsonl \
    --checkpoints-root "$CKPT_ROOT" \
    --out "reports/campaigns/biased_ground_truth_eval_${SCALE}.json"

python analysis/evaluate_ground_truth.py \
    --ground-truth data/preferences_groundtruth/biased_ground_truth_top10.jsonl \
    --checkpoints-root "$CKPT_ROOT" \
    --out "reports/campaigns/biased_top10_eval_${SCALE}.json"

python analysis/evaluate_ground_truth.py \
    --ground-truth data/preferences_groundtruth/biased_ground_truth_top5_allpairs.jsonl \
    --checkpoints-root "$CKPT_ROOT" \
    --out "reports/campaigns/biased_top5_allpairs_eval_${SCALE}.json"

echo "=== Done, scale=${SCALE} ==="
echo "Date: $(date)"
