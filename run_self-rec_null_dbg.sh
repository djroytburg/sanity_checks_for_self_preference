#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=logs/self_rec_null_dbg/self_rec_null_dbg_%j.log
#SBATCH --error=logs/self_rec_null_dbg/self_rec_null_dbg_%j.log
#SBATCH --job-name=self_rec_null_dbg

# run_self-rec_null_dbg.sh: SLURM script for self-recognition null hypothesis test
# Written by: Dani
# Created: Dec 24, 2025, 17:15 EST
# Last Modified: Dec 24, 2025, 17:15 EST

echo "=========================================="
echo "SELF-RECOGNITION NULL HYPOTHESIS TEST"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Activate virtual environment
source .venv/bin/activate

# Create log directory
mkdir -p logs/self_rec_null_dbg

# Run for each dataset
for dataset in alpaca_eval translation truthfulness; do
    echo ""
    echo "=== Processing dataset: $dataset ==="
    echo "Start time: $(date)"
    
    python3 run_self-rec_null_dbg.py \
        --dataset $dataset \
        --output_dir self_rec_null_dbg_results \
        --batch_size 1
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed $dataset"
    else
        echo "✗ Error processing $dataset"
    fi
    
    echo "End time: $(date)"
done

echo ""
echo "=========================================="
echo "=== Self-recognition generation complete ==="
echo "End time: $(date)"
echo "=========================================="
