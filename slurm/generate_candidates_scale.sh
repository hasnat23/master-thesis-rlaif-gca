#!/bin/bash
#SBATCH --job-name=gen_scale
#SBATCH --output=slurm/gen_scale_%j.out
#SBATCH --error=slurm/gen_scale_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# MOGON Candidate Generation — scalable 5000/10000 experiment
#
# Usage:
#   sbatch --export=ALL,SAMPLE_SIZE=5000 slurm/generate_candidates_scale.sh
#   sbatch --export=ALL,SAMPLE_SIZE=10000 slurm/generate_candidates_scale.sh
#
# The subset remains deterministic and nested by default because
# the seed is fixed to 200, matching the existing 1000-sample run.
# ============================================================

set -euo pipefail

: "${SAMPLE_SIZE:?Set SAMPLE_SIZE to 5000 or 10000}"
SUBSET_SEED="${SUBSET_SEED:-200}"

echo "=== Candidate Generation Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "Sample:  $SAMPLE_SIZE"
echo "Seed:    $SUBSET_SEED"
echo "Date:    $(date)"
echo ""

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

SUBSET_PATH="data/subset/subset_${SAMPLE_SIZE}.jsonl"
CANDIDATES_PATH="data/candidates/candidates_${SAMPLE_SIZE}.jsonl"

mkdir -p data/subset data/candidates

echo "--- Preparing ${SAMPLE_SIZE}-sample subset ---"
python scripts/01_prepare_subset.py \
    --config configs/subset_1000.yaml \
    --override n_samples=${SAMPLE_SIZE} seed=${SUBSET_SEED} output_filename=subset_${SAMPLE_SIZE}.jsonl

echo ""
WC=$(wc -l < "$SUBSET_PATH" 2>/dev/null || echo 0)
echo "Subset size: $WC lines"
if [[ "$WC" -lt $((SAMPLE_SIZE - 50)) ]]; then
    echo "ERROR: subset too small ($WC vs expected ${SAMPLE_SIZE}). Aborting."
    exit 1
fi

echo "--- Generating ${SAMPLE_SIZE} candidate pairs ---"
python scripts/02_generate_candidates.py \
    --config configs/generation_1000.yaml \
    --override subset_path=${SUBSET_PATH} output_filename=candidates_${SAMPLE_SIZE}.jsonl

echo ""
WC2=$(wc -l < "$CANDIDATES_PATH" 2>/dev/null || echo 0)
echo "Candidates generated: $WC2"

echo "=== Candidate Generation Complete ==="
echo "Date:   $(date)"
echo "Output: $CANDIDATES_PATH"
