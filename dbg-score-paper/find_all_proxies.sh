# find_all_proxies.sh: Find best proxy models for all judges
# Written by: Dani
# Created: Dec 23, 2024, 11:10 EST
# Last Modified: Dec 23, 2024, 11:10 EST

echo "=========================================="
echo "Finding proxy models for all judges"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

# Activate virtual environment
echo "Activating virtual environment..."
cd ../
source .venv/bin/activate

# Create log directory
mkdir -p dbg-score-paper/proxy_preference_data/logs

for DATASET in alpaca_eval truthfulness translation; do
    OUTPUT_FILE="dbg-score-paper/proxy_preference_data/logs/$(date +%Y%m%d_%H%M)-$DATASET.txt"
    python3 dbg-score-paper/find_proxy_models.py "$DATASET" dbg-score-paper/model_preferences_fullset 2>&1 | tee -a "$OUTPUT_FILE"
done

echo "=========================================="
echo "Proxy search completed"
echo "Logs saved to: dbg-score-paper/proxy_preference_data/logs/"
echo "End time: $(date)"
echo "=========================================="
