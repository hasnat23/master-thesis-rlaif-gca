#!/bin/bash
#SBATCH --job-name=gen_rm500
#SBATCH --output=slurm/gen_rm500_%j.out
#SBATCH --error=slurm/gen_rm500_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# MOGON Candidate Generation — 500-sample RM training set
#
# Purpose: Generate 2 candidate summaries per article for 500 articles
#   drawn from CNN/DM test split with seed=100 (disjoint from the
#   200-sample DPO set which uses seed=42).
#
# Supports resume: if data/candidates/candidates_rm500.jsonl already
#   exists, already-generated sample_ids are skipped automatically.
#
# Model:   Mistral-7B-Instruct-v0.3 (local at ~/thesis/models/)
# Output:  data/candidates/candidates_rm500.jsonl
# Est.:    ~3-4 hours on A100
#
# Submit: sbatch slurm/generate_candidates_rm500.sh
# ============================================================

echo "=== RM-500 Candidate Generation Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:    $(date)"
echo ""

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Working directory: $(pwd)"
echo "Python:            $(which python)"
echo ""

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1

# Step 1: Prepare 500-sample subset (CPU, fast)
echo "--- Preparing 500-sample subset (seed=100) ---"
python scripts/01_prepare_subset.py \
    --config configs/subset_rm500.yaml

echo ""

# Step 2: Generate candidate summaries (GPU, ~3-4h)
echo "--- Generating 500 candidate pairs (resumes if interrupted) ---"
python scripts/02_generate_candidates.py \
    --config configs/generation_rm500.yaml

echo ""
echo "=== RM-500 Candidate Generation Complete ==="
echo "Date:   $(date)"
echo "Output: data/candidates/candidates_rm500.jsonl"
