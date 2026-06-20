#!/bin/bash
#SBATCH --job-name=gca-hpsearch
#SBATCH --output=logs/gca_hpsearch_%A_%a.log
#SBATCH --error=logs/gca_hpsearch_%A_%a.err
#SBATCH --partition=a100dl
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=1-9
#SBATCH --mail-type=END,FAIL

set -e

WORK_DIR="/fshpc/muhhas01/thesis_git"
cd "$WORK_DIR"

# Initialize environment
. /cluster/easybuild/profile.sh
export PATH="/cluster/easybuild/broadwell/software/Anaconda3/2023.03/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate thesis 2>/dev/null || true

# Define hyperparameter configurations (9 total)
# Grid: 3 learning rates × 3 epoch values
CONFIGS=(
    "1e-5:3"    # Config 1
    "1e-5:5"    # Config 2
    "1e-5:7"    # Config 3
    "2e-5:3"    # Config 4
    "2e-5:5"    # Config 5
    "2e-5:7"    # Config 6
    "5e-5:3"    # Config 7
    "5e-5:5"    # Config 8
    "5e-5:7"    # Config 9
)

# Get this job's configuration
CONFIG=${CONFIGS[$((SLURM_ARRAY_TASK_ID - 1))]}
LR=$(echo $CONFIG | cut -d: -f1)
EPOCHS=$(echo $CONFIG | cut -d: -f2)

echo "=========================================="
echo "GCA RM Hyperparameter Search"
echo "Job ID: $SLURM_ARRAY_TASK_ID / 9"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Paths to optimized preferences (from main optimization job)
HOLISTIC_PREFS="data/preferences_1000_opt_alpha_0.0/holistic_reward_preferences_1000_opt_alpha_0.0.jsonl"
GCA_PREFS="data/preferences_1000_opt_alpha_0.0/gca_reward_preferences_1000_opt_alpha_0.0.jsonl"

# Check preferences exist
if [ ! -f "$HOLISTIC_PREFS" ] || [ ! -f "$GCA_PREFS" ]; then
    echo "ERROR: Optimized preferences not found"
    echo "  Holistic: $HOLISTIC_PREFS"
    echo "  GCA: $GCA_PREFS"
    echo "Run main optimization job first (submit_gca_opt_v2.sh)"
    exit 1
fi

# Output directory for this configuration
OUTPUT_DIR="outputs/reward_models_hpsearch_lr${LR}_ep${EPOCHS}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Training RM with hyperparameters:"
echo "  LR: $LR"
echo "  Epochs: $EPOCHS"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run RM training with these hyperparameters
python3 src/reward_model/run_training.py \
    --holistic "$HOLISTIC_PREFS" \
    --gca "$GCA_PREFS" \
    --output-dir "$OUTPUT_DIR" \
    --kfold 5 \
    --backbone FacebookAI/roberta-base \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size 8 \
    --max-length 512 \
    --device cuda

echo ""
echo "=========================================="
echo "Configuration $SLURM_ARRAY_TASK_ID completed"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
