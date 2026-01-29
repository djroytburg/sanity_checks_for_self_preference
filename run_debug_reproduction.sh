
echo "=========================================="
echo "REPRODUCTION QUALITY DEBUG"
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
