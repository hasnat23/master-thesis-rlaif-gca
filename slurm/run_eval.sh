#!/usr/bin/env bash
# =============================================================================
# Evaluate DPO-fine-tuned adapters vs baseline Mistral-7B
#
# Generates summaries from three conditions (baseline, dpo_holistic, dpo_gca)
# and scores with ROUGE, BERTScore, and AlignScore.
#
# Submit:
#   cd ~/thesis && sbatch slurm/run_eval.sh
# =============================================================================
#SBATCH --job-name=run_eval
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=muhhas01@students.uni-mainz.de
#SBATCH --exclude=gpu0001

set -euo pipefail

# --- environment --------------------------------------------------------------
module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd ~/thesis

echo "=== DPO Evaluation Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Date:    $(date)"
echo ""
echo "Python:  $(which python)"
echo "Working directory: $(pwd)"
echo ""

export PYTHONUNBUFFERED=1

python src/eval/run_eval.py \
    --candidates data/candidates/candidates_200.jsonl \
    --model-path models/Mistral-7B-Instruct-v0.3 \
    --holistic-adapter outputs/dpo/holistic/adapter \
    --gca-adapter outputs/dpo/gca/adapter \
    --alignscore-ckpt models/alignscore/AlignScore-base.ckpt \
    --output-dir outputs/eval \
    --n-test 50 \
    --max-new-tokens 120 \
    --gen-batch-size 4 \
    --seed 42

echo ""
echo "=== Evaluation Complete ==="
echo "Date:    $(date)"
echo "Results: outputs/eval/eval_results.json"
