#!/bin/bash
#SBATCH --job-name=train_rm
#SBATCH --output=slurm/train_rm_%j.out
#SBATCH --error=slurm/train_rm_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# MOGON Bradley-Terry Reward Model Training
#
# Trains two RMs (holistic condition + GCA condition) on the
# 500-sample preference pairs using RoBERTa-base backbone (microsoft/deberta-v3-base
# is not whitelisted on the MOGON HF proxy; FacebookAI/roberta-base is already cached).
#
# Input:  data/preferences_rm500/holistic_reward_preferences_rm500.jsonl
#         data/preferences_rm500/gca_reward_preferences_rm500.jsonl
# Output: outputs/reward_models/{holistic,gca}/best/
#         outputs/reward_models/rm_training_summary.json
#
# Uses 5-fold cross-validation (no held-out split needed with ~380 pairs).
# Est.: ~30-60 min on A100
#
# Dependency: run after build_reward_preferences_rm500.sh completes.
# Submit: sbatch slurm/train_reward_models.sh
# ============================================================

echo "=== Bradley-Terry RM Training Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:    $(date)"
echo ""

export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
echo "HF_ENDPOINT: $HF_ENDPOINT"

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Working directory: $(pwd)"
echo "Python:            $(which python)"
echo ""

export PYTHONUNBUFFERED=1

HOLISTIC="data/preferences_rm500/holistic_reward_preferences_rm500.jsonl"
GCA="data/preferences_rm500/gca_reward_preferences_rm500.jsonl"

if [[ ! -f "$HOLISTIC" || ! -f "$GCA" ]]; then
    echo "ERROR: preference files not found. Run build_reward_preferences_rm500.sh first."
    exit 1
fi

echo "--- Training holistic RM + GCA RM (5-fold CV) ---"
python src/reward_model/run_training.py \
    --holistic "$HOLISTIC" \
    --gca      "$GCA" \
    --output-dir outputs/reward_models \
    --backbone FacebookAI/roberta-base \
    --epochs 5 \
    --batch-size 8 \
    --lr 2e-5 \
    --max-length 512 \
    --max-article-chars 2000 \
    --kfold 5 \
    --seed 42

echo ""
echo "=== Bradley-Terry RM Training Complete ==="
echo "Date:   $(date)"
echo "Output: outputs/reward_models/"
