#!/bin/bash
#SBATCH --job-name=gca-opt-v3-200
#SBATCH --output=logs/gca_opt_v3_200_%A.log
#SBATCH --error=logs/gca_opt_v3_200_%A.err
#SBATCH --partition=a100dl
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --mail-type=END,FAIL

# Load conda environment with all dependencies
WORK_DIR="/fshpc/muhhas01/thesis_git"
cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }

# Load conda and activate thesis environment (has torch, transformers, etc.)
module load lang/Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"
conda activate thesis_env

echo "=========================================="
echo "GCA Optimization v3 (200-sample validation)"
echo "=========================================="
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python3)"
python3 --version
echo ""

# Step 1: Backup old results
echo "Step 1: Backing up old results..."
mkdir -p outputs/backup_alpha_0.5
if [ -d "outputs/reward_models_200" ]; then
    cp -r outputs/reward_models_200 outputs/backup_alpha_0.5/ 2>/dev/null && echo "  ✓ Backed up old models" || true
fi

# Step 2: Create output directories
echo ""
echo "Step 2: Creating output directories..."
mkdir -p data/preferences_200_opt_alpha_0.0
mkdir -p outputs/reward_models_200_opt_alpha_0.0
echo "  ✓ Directories created"

# Step 3: Build preferences
echo ""
echo "Step 3: Building preferences with alpha=0.0..."
python3 -u src/judging/build_reward_preferences.py \
    --candidates data/candidates/candidates_200.jsonl \
    --output-dir data/preferences_200_opt_alpha_0.0 \
    --output-suffix "200_opt_alpha_0.0" \
    --judge-backend alignscore \
    --model-name yzha/AlignScore \
    --alignscore-backbone FacebookAI/roberta-base \
    --alignscore-filename AlignScore-base.ckpt \
    --margin 0 \
    --max-samples 200 \
    --mode both \
    --alpha 0.0 \
    2>&1 | tee -a build_prefs_$(date +%s).log

echo ""
echo "Step 3b: Verifying preference files..."
HOLISTIC="data/preferences_200_opt_alpha_0.0/holistic_reward_preferences_200_opt_alpha_0.0.jsonl"
GCA="data/preferences_200_opt_alpha_0.0/gca_reward_preferences_200_opt_alpha_0.0.jsonl"

if [ -f "$HOLISTIC" ] && [ -f "$GCA" ]; then
    H_COUNT=$(wc -l < "$HOLISTIC")
    G_COUNT=$(wc -l < "$GCA")
    echo "  ✓ Holistic: $H_COUNT pairs"
    echo "  ✓ GCA: $G_COUNT pairs"
else
    echo "  ✗ ERROR: Preference files not generated!"
    ls -la data/preferences_200_opt_alpha_0.0/
    exit 1
fi

# Step 4: Train RM models
echo ""
echo "Step 4: Training RM models with 5-fold cross-validation..."
python3 -u src/reward_model/run_training.py \
    --holistic "$HOLISTIC" \
    --gca "$GCA" \
    --output-dir "outputs/reward_models_200_opt_alpha_0.0" \
    --num-folds 5 \
    --learning-rate 2e-5 \
    --epochs 5 \
    --batch-size 8 \
    --max-length 512 \
    --seed 42 \
    2>&1 | tee -a train_rm_$(date +%s).log

echo ""
echo "Step 5: Running result analysis..."
python3 -u analysis/analyze_optimization_results.py \
    --old-accuracy 0.5600 \
    --old-holistic 0.5720 \
    --new-dir outputs/reward_models_200_opt_alpha_0.0 \
    --output-file outputs/optimization_results_200.txt \
    2>&1 | tee -a analysis_$(date +%s).log

echo ""
echo "=========================================="
echo "✓ GCA Optimization Complete"
echo "=========================================="
echo "Results saved to:"
echo "  - Preferences: data/preferences_200_opt_alpha_0.0/"
echo "  - Models: outputs/reward_models_200_opt_alpha_0.0/"
echo "  - Analysis: outputs/optimization_results_200.txt"
echo ""
echo "Completion time: $(date)"
