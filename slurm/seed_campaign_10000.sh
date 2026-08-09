#!/bin/bash
#SBATCH --job-name=seed_10000
#SBATCH --output=slurm/logs/seed_10000_%A_%a.out
#SBATCH --error=slurm/logs/seed_10000_%A_%a.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=07:00:00
#SBATCH --array=1-20
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# Repeated-run campaign at n=10,000 — twenty independent seeds
#
# Same rationale as slurm/seed_campaign_5000.sh, extended from six seeds to
# twenty to match the rigor of the n=1,000 campaign. Seeds 1-6 already have
# results from the first pass and are skipped automatically by the
# idempotency check below; only seeds 7-20 actually run new training.
#
# Historical single run at this scale (job 1411415) took 4h27m; --time is set
# with margin above that.
#
# Submit: sbatch slurm/seed_campaign_10000.sh
#     or: sbatch --array=7-20 slurm/seed_campaign_10000.sh  (skip the no-ops)
# ============================================================

set -euo pipefail

SEED="${SLURM_ARRAY_TASK_ID}"

echo "=== n=10,000 seed campaign: seed=${SEED} ==="
echo "Job:  ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Node: ${SLURM_NODELIST}"
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

HOLISTIC="data/preferences_10000/holistic_reward_preferences_10000.jsonl"
GCA="data/preferences_10000/gca_reward_preferences_10000.jsonl"
OUT="outputs/seed_campaign_10000/seed_${SEED}"

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
    --kfold 5 \
    --seed "${SEED}"

echo ""
echo "=== Done: n=10,000, seed=${SEED} ==="
echo "Date:   $(date)"
echo "Output: ${OUT}/"
