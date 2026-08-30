#!/bin/bash
#SBATCH --job-name=Pipat_Lab2
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --partition=gpu4090
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00

DIRECTORY_VENV_ACTIVATE="$HOME/DeepLearning/venv/bin/activate"
DIRECTORY_LAB="$HOME/DeepLearning/src/Lab"
LAB_SPECIFY="2"

set -euo pipefail

cd $HOME

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"

source $DIRECTORY_VENV_ACTIVATE

echo "=== Python ==="
echo "Python executable location: $(which python)"
echo "Python version: $(python --version)"

echo "=== Execution ==="
python "$DIRECTORY_LAB$LAB_SPECIFY/lab.py"