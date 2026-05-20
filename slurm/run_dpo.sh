#!/usr/bin/env bash
# =============================================================================
# DPO fine-tuning — holistic and GCA conditions
# Trains Mistral-7B-Instruct-v0.3 with LoRA on 200-sample preference pairs.
#
# Submit:
#   cd ~/thesis && sbatch slurm/run_dpo.sh
# =============================================================================
#SBATCH --job-name=run_dpo
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=muhhas01@students.uni-mainz.de

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

echo "=== DPO Fine-Tuning Start ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Date:    $(date)"
echo ""
echo "Python:  $(which python)"
echo "Working directory: $(pwd)"
echo ""

# --- holistic condition -------------------------------------------------------
echo "--- Training DPO holistic ---"
python src/dpo/run_dpo.py \
    --condition holistic \
    --preferences data/preferences/holistic_reward_preferences_200.jsonl \
    --model-path models/Mistral-7B-Instruct-v0.3 \
    --output-dir outputs/dpo/holistic \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 5e-7 \
    --beta 0.1 \
    --max-length 1024 \
    --lora-r 16 \
    --lora-alpha 32 \
    --seed 42

echo ""
echo "--- Training DPO GCA ---"
python src/dpo/run_dpo.py \
    --condition gca \
    --preferences data/preferences/gca_reward_preferences_200.jsonl \
    --model-path models/Mistral-7B-Instruct-v0.3 \
    --output-dir outputs/dpo/gca \
    --epochs 1 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 5e-7 \
    --beta 0.1 \
    --max-length 1024 \
    --lora-r 16 \
    --lora-alpha 32 \
    --seed 42

echo ""
echo "=== DPO Fine-Tuning Complete ==="
echo "Date:    $(date)"
echo "Outputs: outputs/dpo/"
