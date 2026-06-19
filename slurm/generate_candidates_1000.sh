#!/bin/bash
#SBATCH --job-name=gen_1000
#SBATCH --output=slurm/gen_1000_%j.out
#SBATCH --error=slurm/gen_1000_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# MOGON Candidate Generation — 1000-sample expanded experiment
#
# Prof. feedback (02-06-2026): test with more candidates to get
# stronger RM results. This generates 2 summaries per article
# for 1000 articles (seed=200, disjoint from seed=42 and seed=100).
#
# Supports resume: already-generated sample_ids are skipped.
#
# Model:   Mistral-7B-Instruct-v0.3 (local at ~/thesis/models/)
# Output:  data/candidates/candidates_1000.jsonl
# Est.:    ~5-6 hours on A100
#
# Submit: sbatch slurm/generate_candidates_1000.sh
# ============================================================

echo "=== 1000-Sample Candidate Generation Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
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

echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo ""

# Step 1: Prepare 1000-sample subset (CPU, fast)
echo "--- Preparing 1000-sample subset (seed=200) ---"
python scripts/01_prepare_subset.py \
    --config configs/subset_1000.yaml

echo ""
# Validate before proceeding
WC=$(wc -l < data/subset/subset_1000.jsonl 2>/dev/null || echo 0)
echo "Subset size: $WC lines"
if [[ "$WC" -lt 900 ]]; then
    echo "ERROR: subset too small ($WC < 900). Aborting."
    exit 1
fi

# Step 2: Generate candidate summaries (GPU, ~5-6h)
echo "--- Generating 1000 candidate pairs (resumes if interrupted) ---"
python scripts/02_generate_candidates.py \
    --config configs/generation_1000.yaml

echo ""
WC2=$(wc -l < data/candidates/candidates_1000.jsonl 2>/dev/null || echo 0)
echo "Candidates generated: $WC2"

echo "=== 1000-Sample Candidate Generation Complete ==="
echo "Date:   $(date)"
echo "Output: data/candidates/candidates_1000.jsonl"
