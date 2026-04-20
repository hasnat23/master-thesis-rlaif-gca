#!/bin/bash
#SBATCH --job-name=judge_test
#SBATCH --output=slurm/judge_test_%j.out
#SBATCH --error=slurm/judge_test_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL

# ============================================================
# MOGON Judge Test — 20 candidate pairs
#
# Purpose: Test judging prompts on 20 pairs:
#   1. Holistic A/B preference judgments
#   2. Sentence-level factual scoring
#   3. GCA aggregation
#   4. Reliability filtering
#
# This runs in mock mode by default (no API calls).
# Add --live for real GPT-4o calls (requires OPENAI_API_KEY).
#
# No GPU needed — judging uses OpenAI API, not local model.
#
# Submit: sbatch slurm/judge_test.sh
# ============================================================

echo "=== Judge Test Start ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo ""

# Activate conda environment
module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Working directory: $(pwd)"
echo ""

# Run judge test harness in mock mode
echo "--- Mock Judge Test (20 pairs) ---"
python scripts/03_run_judge_test.py \
    --config configs/judging.yaml \
    --n 20

echo ""
echo "=== Judge Test Complete ==="
echo "Date: $(date)"
