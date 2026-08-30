#!/bin/bash
#SBATCH --job-name=my_helloworld_but_who_am_i
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00

DIRECTORY_VENV="$HOME/deep_learning/my_env"
DIRECTORY_REPO="$HOME/deep_learning/src/DL-2026"
DIRECTORY_LAB="$HOME/deep_learning/src/DL-2026/Lab"
LAB_SPECIFY="2"

set -euo pipefail

cd "$DIRECTORY_REPO"

echo "=== Git ==="
git pull

cd $HOME

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"

source "$DIRECTORY_VENV/bin/activate"

echo "=== Python ==="
echo "Python executable location: $(which python)"
echo "Python version: $(python --version)"

echo "=== Execution ==="
python "$DIRECTORY_LAB/$LAB_SPECIFY/lab.py"