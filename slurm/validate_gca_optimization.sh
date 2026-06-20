#!/bin/bash

# Regenerate preferences with optimized alpha=0.0 and retrain RM models
# This validates that the empirically-proven formula actually improves RM performance

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=========================================="
echo "GCA Optimization Validation Pipeline"
echo "Starting: $TIMESTAMP"
echo "=========================================="
echo ""

# Step 1: Backup previous results
echo "Step 1: Backing up previous results..."
mkdir -p outputs/reward_models_opt_v1_backup
if [ -d "outputs/reward_models_1000" ]; then
    cp -r outputs/reward_models_1000 outputs/reward_models_opt_v1_backup/models_old_alpha_0.5
    echo "  ✓ Backed up old models"
fi
if [ -f "data/preferences_1000/holistic_reward_preferences_1000.jsonl" ]; then
    cp data/preferences_1000/holistic_reward_preferences_1000.jsonl outputs/reward_models_opt_v1_backup/holistic_old_alpha_0.5.jsonl
    cp data/preferences_1000/gca_reward_preferences_1000.jsonl outputs/reward_models_opt_v1_backup/gca_old_alpha_0.5.jsonl
    echo "  ✓ Backed up old preferences"
fi

# Step 2: Create new output directory for optimized run
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
    --alpha 0.0

HOLISTIC_PREFS="data/preferences_1000_opt_alpha_0.0/holistic_reward_preferences_1000_opt_alpha_0.0.jsonl"
GCA_PREFS="data/preferences_1000_opt_alpha_0.0/gca_reward_preferences_1000_opt_alpha_0.0.jsonl"

if [ ! -f "$HOLISTIC_PREFS" ] || [ ! -f "$GCA_PREFS" ]; then
    echo "ERROR: Preference files not generated!"
    exit 1
fi
echo "  ✓ Preferences built"

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
    --device cuda

echo "  ✓ RM models trained"

# Step 5: Analyze and compare results
echo ""
echo "Step 5: Analyzing results..."
python3 -c "
import json
import os

def get_metrics(summary_path):
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        return json.load(f)

old_summary = 'outputs/reward_models_1000/rm_training_summary.json'
new_summary = 'outputs/reward_models_1000_opt_alpha_0.0/rm_training_summary.json'

old_metrics = get_metrics(old_summary)
new_metrics = get_metrics(new_summary)

if not old_metrics or not new_metrics:
    print('ERROR: Could not load metrics')
    exit(1)

print('\\n========================================')
print('OPTIMIZATION RESULTS COMPARISON')
print('========================================\\n')

print('HOLISTIC ACCURACY:')
old_h = old_metrics.get('holistic', {}).get('mean_accuracy', 0)
new_h = new_metrics.get('holistic', {}).get('mean_accuracy', 0)
print(f'  Old (alpha=0.5): {old_h:.4f}')
print(f'  New (alpha=0.0): {new_h:.4f}')
print(f'  Change: {(new_h - old_h)*100:+.2f}%')

print('\\nGCA ACCURACY:')
old_g = old_metrics.get('gca', {}).get('mean_accuracy', 0)
new_g = new_metrics.get('gca', {}).get('mean_accuracy', 0)
print(f'  Old (alpha=0.5): {old_g:.4f}')
print(f'  New (alpha=0.0): {new_g:.4f}')
print(f'  Change: {(new_g - old_g)*100:+.2f}%')

print('\\nGCA vs HOLISTIC:')
old_gap = old_h - old_g
new_gap = new_h - new_g
print(f'  Old gap (Holistic - GCA): {old_gap:+.4f}')
print(f'  New gap (Holistic - GCA): {new_gap:+.4f}')
print(f'  Gap change: {(new_gap - old_gap)*100:+.2f}%')

if new_g > old_g:
    print(f'\\n✓ SUCCESS: GCA accuracy improved by {(new_g - old_g)*100:.2f}%')
else:
    print(f'\\n⚠ WARNING: GCA accuracy decreased by {(old_g - new_g)*100:.2f}%')

if new_g > new_h:
    print('✓ VICTORY: GCA now outperforms Holistic!')
else:
    print(f'⚠ Gap remaining: {abs(new_h - new_g)*100:.2f}%')
"

echo ""
echo "=========================================="
echo "Pipeline completed: $(date +%Y%m%d_%H%M%S)"
echo "=========================================="
