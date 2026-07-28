#!/bin/bash
#SBATCH --job-name=trunc_abl
#SBATCH --output=slurm/logs/trunc_abl_%A_%a.out
#SBATCH --error=slurm/logs/trunc_abl_%A_%a.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --array=0-14
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu0001

# ============================================================
# Truncation ablation — how much of the source does the RM need?
#
# Motivation: both reward models sit near chance (~52%). One candidate
# explanation is that the model never sees enough of the article to recover the
# evaluator's judgement. The evaluator scored summaries against the whole
# article; the reward model reads a truncated prefix.
#
# The binding limit is NOT --max-article-chars. Median article is ~3,643 chars
# and median summary ~880 chars, so at --max-article-chars 2000 the
# concatenation runs to roughly 720 tokens and is then cut to the 512-token cap.
# Raising --max-article-chars alone therefore changes nothing, because
# roberta-base cannot attend past 512 positions at all.
#
# Design (5 configurations x 3 seeds = 15 array tasks):
#
#   A  roberta-base   len  512  chars  500   less context
#   B  roberta-base   len  512  chars 1000   less context
#   C  roberta-base   len  512  chars 2000   baseline (as reported in the thesis)
#   D  deberta-v3     len  512  chars 2000   backbone control, context held fixed
#   E  deberta-v3     len 1024  chars 4000   more context, backbone held fixed
#
# A->C isolates the effect of shortening the prefix under a fixed 512-token cap.
# C->D isolates the backbone change at fixed context. D->E isolates additional
# context at a fixed backbone: deberta-v3 uses relative position embeddings and
# is not limited to 512 positions. Keeping D and E separate is what prevents the
# backbone and the context length from being confounded.
#
# Three seeds per configuration so that a difference between configurations can
# be read against run-to-run noise rather than assumed to exceed it.
#
# Submit: sbatch slurm/truncation_ablation.sh
# ============================================================

set -euo pipefail

# config_id : backbone : max_length : max_article_chars
CONFIGS=(
  "A:FacebookAI/roberta-base:512:500"
  "B:FacebookAI/roberta-base:512:1000"
  "C:FacebookAI/roberta-base:512:2000"
  "D:microsoft/deberta-v3-base:512:2000"
  "E:microsoft/deberta-v3-base:1024:4000"
)
SEEDS=(1 2 3)

N_SEEDS=${#SEEDS[@]}
CFG_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

IFS=':' read -r CFG_ID BACKBONE MAXLEN MAXCHARS <<< "${CONFIGS[$CFG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "=== Truncation ablation ==="
echo "Task:     ${SLURM_ARRAY_TASK_ID}  (config ${CFG_ID}, seed ${SEED})"
echo "Backbone: ${BACKBONE}"
echo "MaxLen:   ${MAXLEN} tokens"
echo "MaxChars: ${MAXCHARS}"
echo "Node:     ${SLURM_NODELIST}"
echo "Date:     $(date)"
echo ""

export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

HOLISTIC="data/preferences_1000/holistic_reward_preferences_1000.jsonl"
GCA="data/preferences_1000/gca_reward_preferences_1000.jsonl"
OUT="outputs/truncation_ablation/cfg_${CFG_ID}/seed_${SEED}"

if [[ ! -f "$HOLISTIC" || ! -f "$GCA" ]]; then
    echo "ERROR: preference files not found. Run build_preferences_1000.sh first."
    exit 1
fi

if [[ -f "${OUT}/rm_training_summary.json" ]]; then
    echo "Summary already exists for ${CFG_ID}/seed_${SEED}; nothing to do."
    exit 0
fi

# Fail loudly and early if the backbone is not in the local HF cache. With
# HF_HUB_OFFLINE=1 a missing checkpoint otherwise surfaces as an opaque error
# partway into training, after the job has already consumed its allocation.
python - "$BACKBONE" <<'PYCHECK'
import sys
from transformers import AutoTokenizer, AutoConfig
name = sys.argv[1]
try:
    AutoConfig.from_pretrained(name)
    AutoTokenizer.from_pretrained(name)
    print(f"[pre-flight] backbone available offline: {name}")
except Exception as exc:
    sys.exit(
        f"[pre-flight] backbone NOT available offline: {name}\n"
        f"  {type(exc).__name__}: {exc}\n"
        f"  Fetch it once on a login node with HF_HUB_OFFLINE=0 before submitting."
    )
PYCHECK

python src/reward_model/run_training.py \
    --holistic "$HOLISTIC" \
    --gca      "$GCA" \
    --output-dir "$OUT" \
    --backbone "$BACKBONE" \
    --epochs 5 \
    --batch-size 8 \
    --lr 2e-5 \
    --max-length "$MAXLEN" \
    --max-article-chars "$MAXCHARS" \
    --kfold 5 \
    --seed "$SEED"

echo ""
echo "=== Done: config ${CFG_ID}, seed ${SEED} ==="
echo "Date:   $(date)"
echo "Output: ${OUT}/"
