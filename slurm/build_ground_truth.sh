#!/bin/bash
#SBATCH --job-name=build_gt
#SBATCH --output=slurm/logs/build_gt_%j.out
#SBATCH --error=slurm/logs/build_gt_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# Ground-truth construction via AlignScore (sentence-level + GCA aggregation)
#
# Per Lingxiao (11 Aug 2026): score each candidate summary with AlignScore's
# sentence-level scoring aggregated via GCA (alpha=0.0, the locked formula),
# and use that single, independent score to rank each article's two
# candidates into "should score higher" (A) / "should score lower" (B).
#
# This calls build_reward_preferences.py in --mode gca only: we want ONE
# common ground-truth ranking derived purely from the GCA-aggregated score,
# used to test BOTH the holistic and the GCA reward model on equal footing
# -- not each RM re-tested against its own training-time labels.
#
# Locked config: nli mode, roberta-base backbone, alpha=0.0, margin=0
# (matches thesis/chapters/04_methodology.tex Table 4.1).
#
# Dependency: run after generate_candidates_groundtruth.sh completes.
# Submit: sbatch slurm/build_ground_truth.sh
# ============================================================

set -euo pipefail

echo "=== Ground-Truth Construction (AlignScore, GCA mode) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"

export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

# The HF mirror at HF_ENDPOINT is currently returning HTTP 500 (checked
# directly: `curl $HF_ENDPOINT` -> 500). FacebookAI/roberta-base is already
# fully cached locally (~/.cache/huggingface/hub/models--FacebookAI--roberta-base),
# so force local-only lookup rather than waiting on a dead mirror for every
# one of 500 samples, which silently SKIPs every sample instead of failing
# loudly. AlignScore's own .ckpt is loaded from --alignscore-ckpt below,
# independent of the HF hub entirely.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

if [[ ! -f "data/candidates/candidates_groundtruth.jsonl" ]]; then
    echo "ERROR: data/candidates/candidates_groundtruth.jsonl not found."
    echo "Run generate_candidates_groundtruth.sh first."
    exit 1
fi

python src/judging/build_reward_preferences.py \
    --candidates data/candidates/candidates_groundtruth.jsonl \
    --output-dir data/preferences_groundtruth \
    --judge-backend alignscore \
    --model-name yzha/AlignScore \
    --alignscore-backbone FacebookAI/roberta-base \
    --alignscore-ckpt models/alignscore/AlignScore-base.ckpt \
    --alignscore-evaluation-mode nli \
    --margin 0 \
    --max-samples 500 \
    --mode gca \
    --alpha 0.0

echo ""
echo "=== Done ==="
echo "Date: $(date)"
echo "Output: data/preferences_groundtruth/gca_reward_preferences_groundtruth.jsonl"
echo "  (chosen = higher GCA-aggregated AlignScore summary = subset A / high)"
echo "  (rejected = lower GCA-aggregated AlignScore summary = subset B / low)"
