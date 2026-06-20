#!/bin/bash
#SBATCH --job-name=gca-opt-v3-simple
#SBATCH --output=logs/gca_opt_v3_%A.log
#SBATCH --error=logs/gca_opt_v3_%A.err
#SBATCH --partition=a100dl
#SBATCH --account=nhr-haloed
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --mail-type=END,FAIL

# Use proven working env vars from old scripts
export HF_ENDPOINT=http://10.81.2.171:8090
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

WORK_DIR="/fshpc/muhhas01/thesis_git"
cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }

# Load conda and activate thesis environment (has torch, transformers, etc.)
module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

echo "=========================================="
echo "GCA Optimization v3 (Simplified)"
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
if [ -d "outputs/reward_models_1000" ]; then
    cp -r outputs/reward_models_1000 outputs/backup_alpha_0.5/ 2>/dev/null && echo "  ✓ Backed up old models" || true
fi

# Step 2: Create output directories
echo ""
echo "Step 2: Creating output directories..."
mkdir -p data/preferences_1000_opt_alpha_0.0
mkdir -p outputs/reward_models_1000_opt_alpha_0.0
echo "  ✓ Directories created"

# Step 3: Build preferences
echo ""
echo "Step 3: Building preferences with alpha=0.0..."
python src/judging/build_reward_preferences.py \
    --candidates data/candidates/candidates_1000.jsonl \
    --output-dir data/preferences_1000_opt_alpha_0.0 \
    --output-suffix 1000_opt_alpha_0.0 \
    --judge-backend alignscore \
    --model-name yzha/AlignScore \
    --alignscore-backbone FacebookAI/roberta-base \
    --alignscore-filename AlignScore-base.ckpt \
    --margin 0 \
    --max-samples 1000 \
    --mode both \
    --alpha 0.0

echo ""
echo "Step 3b: Verifying preference files..."
HOLISTIC="data/preferences_1000_opt_alpha_0.0/holistic_reward_preferences_1000_opt_alpha_0.0.jsonl"
GCA="data/preferences_1000_opt_alpha_0.0/gca_reward_preferences_1000_opt_alpha_0.0.jsonl"

if [ -f "$HOLISTIC" ] && [ -f "$GCA" ]; then
    H_COUNT=$(wc -l < "$HOLISTIC")
    G_COUNT=$(wc -l < "$GCA")
    echo "  ✓ Holistic: $H_COUNT pairs"
    echo "  ✓ GCA: $G_COUNT pairs"
else
    echo "  ✗ ERROR: Preference files not generated!"
    ls -la data/preferences_1000_opt_alpha_0.0/
    exit 1
fi

# Step 4: Train RM models
echo ""
echo "Step 4: Training RM models (5-fold CV, n=1000)..."
python src/reward_model/run_training.py \
    --holistic "$HOLISTIC" \
    --gca      "$GCA" \
    --output-dir outputs/reward_models_1000_opt_alpha_0.0 \
    --backbone FacebookAI/roberta-base \
    --epochs 5 \
    --batch-size 8 \
    --lr 2e-5 \
    --max-length 512 \
    --max-article-chars 2000 \
    --kfold 5 \
    --seed 42

echo ""
echo "Step 4b: Verifying training results..."
if [ -f "outputs/reward_models_1000_opt_alpha_0.0/rm_training_summary.json" ]; then
    echo "  ✓ Training summary found"
    echo ""
    cat outputs/reward_models_1000_opt_alpha_0.0/rm_training_summary.json
    echo ""
    echo "Step 5: Analyzing results vs baseline..."
    python3 analysis/analyze_optimization_results.py 2>&1
else
    echo "  ✗ Training summary not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Pipeline completed: $(date)"
echo "=========================================="
