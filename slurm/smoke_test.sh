#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --output=slurm/smoke_test_%j.out
#SBATCH --error=slurm/smoke_test_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL

# ============================================================
# MOGON Smoke Test — 5 samples
#
# Purpose: Validate that the full pipeline works on MOGON:
#   1. Load 7B model with 4-bit quantization
#   2. Generate candidate pairs for 5 articles
#   3. Compute ROUGE metrics
#
# Before running:
#   1. Update configs/generation.yaml with actual model name
#   2. Ensure data/subset/subset_200.jsonl exists
#      (run scripts/01_prepare_subset.py locally first)
#
# Submit: sbatch slurm/smoke_test.sh
# ============================================================

echo "=== Smoke Test Start ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

# Load modules (adjust to your MOGON environment)
# module load lang/Python/3.11.5-GCCcore-13.2.0
# module load lib/CUDA/12.1.1

# Activate conda environment
module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo ""

# Step 1: Prepare a tiny subset (5 samples) if not already done
echo "--- Step 1: Prepare 5-sample subset ---"
python scripts/01_prepare_subset.py \
    --config configs/subset.yaml \
    --override n_samples=5 output_filename=subset_smoke.jsonl local_dataset_path=data/cnn_dailymail_500

echo ""

# Step 2: Generate candidates for the 5 samples
echo "--- Step 2: Generate candidates ---"
python scripts/02_generate_candidates.py \
    --config configs/generation.yaml \
    --override subset_path=data/subset/subset_smoke.jsonl output_filename=candidates_smoke.jsonl model_name=models/opt-125m load_in_4bit=false

echo ""

# Step 3: Evaluate baseline metrics (ROUGE only for speed)
echo "--- Step 3: Evaluate baseline ---"
python scripts/04_evaluate_baseline.py \
    --candidates data/candidates/candidates_smoke.jsonl \
    --no-bert

echo ""
echo "=== Smoke Test Complete ==="
echo "Date: $(date)"
