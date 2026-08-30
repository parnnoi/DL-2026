#!/bin/bash
#SBATCH --job-name=lab_2
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

set -euo pipefail

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"

source $HOME/deep_learning/env_dl/bin/activate

echo "=== Python ==="
echo "Python executable location: $(which python)"
echo "Python version: $(python --version)"

echo "=== GPU ==="
nvidia-smi

echo "=== Execution ==="
python $HOME/deep_learning/src/lab_2/example_memory_profile.py 50000 20000 20000 40000 --dtype bfloat16 --output $HOME/deep_learning/results/lab_2/profile.pickle
python $HOME/deep_learning/src/lab_2/memory_viz.py trace_plot $HOME/deep_learning/results/lab_2/profile.pickle -o $HOME/deep_learning/results/lab_2/profile.html