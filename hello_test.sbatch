#!/bin/bash
#SBATCH --job-name=my_helloworld_but_who_am_i
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00

set -euo pipefail

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"

source $HOME/deep_learning/my_env/bin/activate

echo "=== Python ==="
echo "Python executable location: $(which python)"
echo "Python version: $(python --version)"

echo "=== Execution ==="
python $HOME/deep_learning/src/test.py