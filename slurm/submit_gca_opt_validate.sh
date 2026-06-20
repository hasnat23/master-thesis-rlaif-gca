#!/bin/bash
#SBATCH --job-name=gca-opt-validate
#SBATCH --output=logs/gca_opt_validate_%A.log
#SBATCH --error=logs/gca_opt_validate_%A.err
#SBATCH --partition=a100dl
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=h.ahmed@uni-mainz.de

set -e

# Use the cloned git repository
WORK_DIR="/fshpc/muhhas01/thesis_git"
cd "$WORK_DIR"

echo "Node: $(hostname)"
echo "Working directory: $WORK_DIR"
echo "Time: $(date)"
echo ""

# Load environment (allow failure for missing modules)
module load cuda 2>/dev/null || true
source ~/.bashrc
conda activate thesis 2>/dev/null || true

# Run validation pipeline
bash slurm/validate_gca_optimization.sh

# Final status
echo ""
echo "PIPELINE STATUS: $?"
echo "Completion time: $(date)"
