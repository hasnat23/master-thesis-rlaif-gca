#!/bin/bash
#SBATCH --job-name=gca-opt-validate
#SBATCH --output=logs/gca_opt_validate_%A.log
#SBATCH --error=logs/gca_opt_validate_%A.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=h.ahmed@uni-mainz.de

set -e

cd ~/thesis

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=index,name --format=csv,noheader)"
echo "Time: $(date)"
echo ""

# Load environment
module load cuda/12.0
source ~/.bashrc
conda activate thesis

# Run validation pipeline
bash slurm/validate_gca_optimization.sh

# Final status
echo ""
echo "PIPELINE STATUS: $?"
echo "Completion time: $(date)"
