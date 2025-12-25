#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --job-name=debug_repro
#SBATCH --output=logs/reproduction/debug_repro_%j.log
#SBATCH --error=logs/reproduction/debug_repro_%j.log

# run_debug_reproduction.sh: SLURM script for automated reproduction quality debugging
# Written by: Dani
# Created: 2025-12-22 01:15 EST
# Last Modified: 2025-12-22 01:15 EST

echo "=========================================="
echo "REPRODUCTION QUALITY DEBUG"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Activate environment
source .venv/bin/activate

# Log GPU info
echo "GPU Information:"
nvidia-smi

echo ""
echo "=========================================="
echo "Running debug script..."
echo "=========================================="

# Run debugging script with all three families
python3 debug_reproduction_quality.py \
    --families llama qwen gemma \
    --benchmark math500 \
    --n_samples 100

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Debug session completed"
else
    echo "ERROR: Debug session failed with exit code $EXIT_CODE"
fi
echo "Completed: $(date)"
echo "=========================================="

exit $EXIT_CODE
