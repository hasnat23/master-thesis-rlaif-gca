#!/bin/bash
#SBATCH --job-name=train_rm_1000
#SBATCH --output=slurm/train_rm_1000_%j.out
#SBATCH --error=slurm/train_rm_1000_%j.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# MOGON Bradley-Terry RM Training — 1000-sample, no margin
#
# Prof. feedback (02-06-2026): remove DPO, focus purely on RM
# comparison. Also use all pairs (no margin filtering).
#
# Input:  data/preferences_1000/holistic_preferences_1000.jsonl
#         data/preferences_1000/gca_preferences_1000.jsonl
# Output: outputs/reward_models_1000/{holistic,gca}/
#         outputs/reward_models_1000/rm_training_summary.json
#
# Uses 5-fold CV on the full ~1000-pair dataset.
#
# Dependency: run after build_preferences_1000.sh completes.
# Submit: sbatch slurm/train_rm_1000.sh
# ============================================================

echo "=== Bradley-Terry RM Training (1000-sample, no margin) Start ==="
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

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR" || exit 1

export PYTHONUNBUFFERED=1

HOLISTIC="data/preferences_1000/holistic_reward_preferences_1000.jsonl"
GCA="data/preferences_1000/gca_reward_preferences_1000.jsonl"

if [[ ! -f "$HOLISTIC" || ! -f "$GCA" ]]; then
    echo "ERROR: preference files not found. Run build_preferences_1000.sh first."
    exit 1
fi

# Step check before training
echo "--- Pre-training step check ---"
python3 -c "
import json
from collections import Counter
for tag, path in [('holistic', '$HOLISTIC'), ('gca', '$GCA')]:
    rows = [json.loads(l) for l in open(path)]
    c = Counter(r['decision'] for r in rows)
    usable = c['A'] + c['B']
    print(f'{tag}: {len(rows)} total, {usable} usable ({usable/len(rows):.1%})')
"
echo ""

echo "--- Training holistic RM + GCA RM (5-fold CV, n=1000) ---"
python src/reward_model/run_training.py \
    --holistic "$HOLISTIC" \
    --gca      "$GCA" \
    --output-dir outputs/reward_models_1000 \
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
echo "Output: outputs/reward_models_1000/"
