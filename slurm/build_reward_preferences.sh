#!/bin/bash
#SBATCH --job-name=reward_prefs
#SBATCH --output=slurm/reward_prefs_%j.out
#SBATCH --error=slurm/reward_prefs_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# MOGON Reward-Model Preference Construction
#
# Purpose: Score all 200 candidate pairs using a fixed
#   factuality/faithfulness model (no GPT/OpenAI).
#   Produces two DPO-ready preference files:
#     data/preferences/holistic_reward_preferences_200.jsonl
#     data/preferences/gca_reward_preferences_200.jsonl
#
# Model:  AlignScore-base (`yzha/AlignScore`, RoBERTa-base backbone)
# Input:  data/candidates/candidates_200.jsonl
# Est.    a few minutes on A100 once checkpoint/backbone are available
#
# Submit: sbatch slurm/build_reward_preferences.sh
# ============================================================

echo "=== Reward Preference Construction Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:    $(date)"
echo ""

# ---- Hugging Face proxy (required on MOGON) ----
export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
echo "HF_ENDPOINT: $HF_ENDPOINT"

# ---- Conda environment ----
module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Working directory: $(pwd)"
echo "Python:            $(which python)"
echo ""

export PYTHONUNBUFFERED=1

# ---- Run preference construction ----
echo "--- Building reward-model preferences (holistic + GCA) ---"
python src/judging/build_reward_preferences.py \
    --candidates data/candidates/candidates_200.jsonl \
    --output-dir data/preferences \
    --judge-backend alignscore \
    --model-name yzha/AlignScore \
    --alignscore-backbone roberta-base \
    --alignscore-filename AlignScore-base.ckpt \
    --margin 0.05 \
    --max-samples 200 \
    --mode both \
    --alpha 0.5

echo ""

# ---- Run analysis if both output files exist ----
HOLISTIC="data/preferences/holistic_reward_preferences_200.jsonl"
GCA="data/preferences/gca_reward_preferences_200.jsonl"

if [[ -f "$HOLISTIC" && -f "$GCA" ]]; then
    echo "--- Generating analysis report ---"
    python src/analysis/analyze_reward_preferences.py \
        --holistic "$HOLISTIC" \
        --gca      "$GCA" \
        --reports-dir reports
    echo "Report saved to reports/reward_model_judging_results.md"
else
    echo "WARNING: one or both preference files missing; skipping analysis."
fi

echo ""
echo "=== Reward Preference Construction Complete ==="
echo "Date: $(date)"
