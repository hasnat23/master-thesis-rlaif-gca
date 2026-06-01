#!/bin/bash
#SBATCH --job-name=bootstrap_ci
#SBATCH --partition=smallcpu
#SBATCH --account=nhr-haloed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd ~/thesis

module load lang/Anaconda3/2024.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis_env

echo "=== Bootstrap CI + Wilcoxon tests ==="
echo "Started at $(date)"

python src/eval/bootstrap_ci.py \
    --generations outputs/eval/generations.jsonl \
    --eval-results outputs/eval/eval_results.json \
    --output-dir outputs/eval \
    --n-bootstrap 10000 \
    --seed 42

echo "=== Done at $(date) ==="
echo "Results: outputs/eval/bootstrap_ci.json"
