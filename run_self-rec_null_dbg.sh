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
