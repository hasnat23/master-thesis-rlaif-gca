#!/bin/bash
#SBATCH --job-name=gen_gt_candidates
#SBATCH --output=slurm/logs/gen_gt_candidates_%j.out
#SBATCH --error=slurm/logs/gen_gt_candidates_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# Ground-truth candidate generation — 500 held-out articles
#
# Same model, same two temperatures as every other candidate-generation run.
# The only difference is the article pool: data/subset/subset_groundtruth.jsonl
# is disjoint from the n=10,000 training pool by construction
# (src/data/subset.py::select_disjoint_subset). Run
# scripts/03_prepare_ground_truth_subset.py on the login node first.
#
# Submit: sbatch slurm/generate_candidates_groundtruth.sh
# ============================================================

set -euo pipefail

echo "=== Ground-Truth Candidate Generation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

if [[ ! -f "data/subset/subset_groundtruth.jsonl" ]]; then
    echo "ERROR: data/subset/subset_groundtruth.jsonl not found."
    echo "Run: python scripts/03_prepare_ground_truth_subset.py --config configs/subset_groundtruth.yaml"
    exit 1
fi

export PYTHONUNBUFFERED=1
export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1

python scripts/02_generate_candidates.py --config configs/generation_groundtruth.yaml

echo ""
echo "=== Done ==="
echo "Date: $(date)"
echo "Output: data/candidates/candidates_groundtruth.jsonl"
