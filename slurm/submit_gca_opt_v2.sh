#!/bin/bash
#SBATCH --job-name=gca-opt-v2
#SBATCH --output=logs/gca_opt_v2_%A.log
#SBATCH --error=logs/gca_opt_v2_%A.err
#SBATCH --partition=a100dl
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=h.ahmed@uni-mainz.de

set -e

WORK_DIR="/fshpc/muhhas01/thesis_git"
cd "$WORK_DIR"

echo "=========================================="
echo "GCA Optimization Validation Pipeline v2"
echo "=========================================="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Working directory: $WORK_DIR"
echo ""

# Initialize module environment
. /cluster/easybuild/profile.sh

# Load Python from environment
export PATH="/cluster/easybuild/broadwell/software/Anaconda3/2023.03/bin:$PATH"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate thesis || source activate thesis || {
    echo "Conda activation failed, attempting direct python..."
}

echo "Python version: $(python3 --version)"
echo "Working directory: $(pwd)"
echo ""

# Step 1: Backup previous results
echo "Step 1: Backing up previous results..."
mkdir -p outputs/reward_models_opt_v1_backup
if [ -d "outputs/reward_models_1000" ]; then
    cp -r outputs/reward_models_1000 outputs/reward_models_opt_v1_backup/models_old_alpha_0.5
    echo "  ✓ Backed up old models"
fi

# Step 2: Create output directory
echo ""
echo "Step 2: Creating output directories..."
mkdir -p data/preferences_1000_opt_alpha_0.0
mkdir -p outputs/reward_models_1000_opt_alpha_0.0
echo "  ✓ Created directories"

# Step 3: Build preferences with optimized alpha=0.0
echo ""
echo "Step 3: Building preferences with optimized alpha=0.0..."
python3 src/judging/build_reward_preferences.py \
    --candidates data/candidates/candidates_1000.jsonl \
    --output-dir data/preferences_1000_opt_alpha_0.0 \
    --output-suffix "1000_opt_alpha_0.0" \
    --judge-backend alignscore \
    --model-name yzha/AlignScore \
    --alignscore-backbone FacebookAI/roberta-base \
    --alignscore-filename AlignScore-base.ckpt \
    --margin 0 \
    --max-samples 1000 \
    --mode both \
    --alpha 0.0 \
    2>&1 | tee build_prefs_log.txt

# Check if preferences were generated
HOLISTIC_PREFS="data/preferences_1000_opt_alpha_0.0/holistic_reward_preferences_1000_opt_alpha_0.0.jsonl"
GCA_PREFS="data/preferences_1000_opt_alpha_0.0/gca_reward_preferences_1000_opt_alpha_0.0.jsonl"

if [ ! -f "$HOLISTIC_PREFS" ] || [ ! -f "$GCA_PREFS" ]; then
    echo "ERROR: Preference files not generated!"
    ls -lah data/preferences_1000_opt_alpha_0.0/
    exit 1
fi

echo "  ✓ Holistic preferences: $(wc -l < "$HOLISTIC_PREFS") lines"
echo "  ✓ GCA preferences: $(wc -l < "$GCA_PREFS") lines"

# Step 4: Train RM models with new preferences
echo ""
echo "Step 4: Training RM models with optimized preferences..."
python3 src/reward_model/run_training.py \
    --holistic "$HOLISTIC_PREFS" \
    --gca "$GCA_PREFS" \
    --output-dir outputs/reward_models_1000_opt_alpha_0.0 \
    --kfold 5 \
    --backbone FacebookAI/roberta-base \
    --epochs 5 \
    --lr 2e-5 \
    --batch-size 8 \
    --max-length 512 \
    --device cuda \
    2>&1 | tee train_rm_log.txt

echo "  ✓ RM models trained"

# Step 5: Compare results
echo ""
echo "Step 5: Analyzing results..."
python3 analysis/analyze_optimization_results.py

echo ""
echo "=========================================="
echo "Pipeline completed: $(date)"
echo "=========================================="
