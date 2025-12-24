#!/bin/bash
#SBATCH --job-name=judge_swap_null_dbg
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=logs/judge_swap_null_dbg/%x_%j.log
#SBATCH --error=logs/judge_swap_null_dbg/%x_%j.log

echo "=========================================="
echo "JUDGE SWAP NULL HYPOTHESIS TEST: DBG"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="

# Activate environment
source .venv/bin/activate

# Create log directory
mkdir -p logs/judge_swap_null_dbg

# Parse arguments (default to testing with 10 examples)
DATASET=${1:-alpaca_eval}
BATCH_SIZE=${2:-1}
MAX_EXAMPLES=${3:-}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Max examples: $MAX_EXAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Run pipeline
echo "Starting preference generation..."
if [ -z "$MAX_EXAMPLES" ]; then
    python -u run_judge_swap_null_dbg.py \
        --dataset "$DATASET" \
        --batch_size "$BATCH_SIZE" \
        --proxy_data_dir "dbg-score-paper/proxy_preference_data" \
        --output_dir "judge_swap_null_dbg_results"
else
    python -u run_judge_swap_null_dbg.py \
    --dataset "$DATASET" \
    --max_examples "$MAX_EXAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --proxy_data_dir "dbg-score-paper/proxy_preference_data" \
    --output_dir "judge_swap_null_dbg_results"
fi


echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
