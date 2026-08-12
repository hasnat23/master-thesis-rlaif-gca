#!/bin/bash
#SBATCH --job-name=retrain_gt
#SBATCH --output=slurm/logs/retrain_gt_%A_%a.out
#SBATCH --error=slurm/logs/retrain_gt_%A_%a.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --array=1-20
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# Retrain holistic + GCA reward models WITH checkpoints saved
#
# The seed-campaign runs (seed_campaign.sh) use run_training.py --kfold 5,
# whose cross-validation path (_kfold_cv) trains and discards five per-fold
# models without ever saving weights -- by design, since only the accuracy
# numbers were needed. Evaluating against the new ground-truth set needs
# actual model weights, so this omits --kfold entirely: run_training.py then
# falls through to train_reward_model() in src/reward_model/train.py, which
# trains once on the FULL preference set and does call _save_checkpoint().
#
# Same training recipe as every other RM result in the thesis (epochs=5,
# lr=2e-5, batch=8, max_length=512, roberta-base), same n=1,000 preference
# files already verified on MOGON. Twenty seeds, matching the resolution of
# the existing seed campaigns.
#
# Submit: sbatch slurm/retrain_for_ground_truth.sh
# ============================================================

set -euo pipefail

SEED="${SLURM_ARRAY_TASK_ID}"

echo "=== Ground-truth retrain: seed=${SEED} ==="
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

HOLISTIC="data/preferences_1000/holistic_reward_preferences_1000.jsonl"
GCA="data/preferences_1000/gca_reward_preferences_1000.jsonl"
OUT="outputs/ground_truth_eval/seed_${SEED}"

if [[ ! -f "$HOLISTIC" || ! -f "$GCA" ]]; then
    echo "ERROR: preference files not found."
    exit 1
fi

if [[ -f "${OUT}/rm_training_summary.json" ]]; then
    echo "Summary already exists for seed ${SEED}; nothing to do."
    exit 0
fi

# No --kfold: single full-data training run per condition, checkpoint saved.
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
