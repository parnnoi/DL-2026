#!/bin/bash
# RunPod version of the Slurm job (no #SBATCH lines: the pod already has the GPU)

DIRECTORY_PROJECT="/DeepLearning"
DIRECTORY_VENV_ACTIVATE="$DIRECTORY_PROJECT/.venv/bin/activate"
DIRECTORY_LAB="$DIRECTORY_PROJECT/src/Lab"
LAB_SPECIFY="9"

export HF_HOME="$DIRECTORY_PROJECT/hf_cache"

# replaces "#SBATCH --output/--error": one log file per run, named by date/time
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$DIRECTORY_PROJECT/logs/lab9"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/$RUN_ID.log") 2> >(tee "$LOG_DIR/$RUN_ID.err" >&2)

set -euo pipefail

# run from the lab folder so model.pth is saved next to lab.py
cd "$DIRECTORY_LAB$LAB_SPECIFY"

START_TIME=$SECONDS

print_elapsed() {
    local elapsed=$(( SECONDS - START_TIME ))
    printf "Elapsed time: %02d:%02d:%02d\n" \
        $(( elapsed / 3600 )) $(( (elapsed % 3600) / 60 )) $(( elapsed % 60 ))
}
trap print_elapsed EXIT

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$DIRECTORY_VENV_ACTIVATE"

echo "=== Python ==="
echo "Python executable location: $(which python)"
echo "Python version: $(python --version)"

echo "=== Execution ==="
python -u lab.py          # -u = print logs immediately (no buffering)

echo "Job finished on: $(date)"