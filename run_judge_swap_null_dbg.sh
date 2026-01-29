

echo "=========================================="
echo "JUDGE SWAP NULL HYPOTHESIS TEST: DBG"
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
