#!/bin/bash
#SBATCH --job-name=gen_candidates
#SBATCH --output=slurm/gen_candidates_%j.out
#SBATCH --error=slurm/gen_candidates_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL

# ============================================================
# MOGON Candidate Generation — Full 200 samples
#
# Purpose: Generate 2 candidate summaries per article (200 articles)
# using a 7B instruction-tuned model with 4-bit quantization.
#
# Estimated time: ~1-2 hours on A100
# Estimated GPU memory: ~10-12 GB (4-bit quantized 7B model)
#
# Before running:
#   1. Update configs/generation.yaml model_name
#   2. Ensure data/subset/subset_200.jsonl exists
#   3. Run smoke_test.sh first to validate setup
#
# Submit: sbatch slurm/generate_candidates.sh
# ============================================================

echo "=== Candidate Generation Start ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

# Activate conda environment
module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo ""

# Generate all 200 candidate pairs
echo "--- Generating 200 candidate pairs ---"
python scripts/02_generate_candidates.py \
    --config configs/generation.yaml \
    --override local_dataset_path=data/cnn_dailymail_500

echo ""
echo "=== Candidate Generation Complete ==="
echo "Date: $(date)"
echo "Output: data/candidates/candidates_200.jsonl"
