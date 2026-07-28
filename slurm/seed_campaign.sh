#!/bin/bash
#SBATCH --job-name=seed_campaign
#SBATCH --output=slurm/logs/seed_campaign_%A_%a.out
#SBATCH --error=slurm/logs/seed_campaign_%A_%a.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --array=1-20
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# Extended seed campaign — Bradley-Terry RM, n=1,000
#
# Purpose: the original campaign used six runs (five distinct seed values),
# giving a run-level Wilcoxon signed-rank test p = 0.0625 two-sided, which
# cannot reach 0.05 no matter how consistent the sign is: with six paired
# samples the smallest attainable two-sided p is 0.031. This array runs twenty
# seeds so the test has the resolution to either establish the effect or rule
# it out.
#
# Seeds are the consecutive integers 1..20, fixed in advance, so that no
# seed selection is possible after seeing the results.
#
# Requires the deterministic cross-validation path (torch seeded in
# src/reward_model/run_training.py::_kfold_cv). Under that fix a given seed
# reproduces exactly, so these twenty runs are twenty genuinely distinct
# samples rather than repeated draws from an uncontrolled process.
#
# Submit: sbatch slurm/seed_campaign.sh
# ============================================================

set -euo pipefail

SEED="${SLURM_ARRAY_TASK_ID}"

echo "=== Seed campaign run: seed=${SEED} ==="
echo "Job:  ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Node: ${SLURM_NODELIST}"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

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
OUT="outputs/seed_campaign/seed_${SEED}"

if [[ ! -f "$HOLISTIC" || ! -f "$GCA" ]]; then
    echo "ERROR: preference files not found. Run build_preferences_1000.sh first."
    exit 1
fi

# Skip work already done, so a partially failed array can be resubmitted whole.
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
    --kfold 5 \
    --seed "${SEED}"

echo ""
echo "=== Done: seed=${SEED} ==="
echo "Date:   $(date)"
echo "Output: ${OUT}/"
