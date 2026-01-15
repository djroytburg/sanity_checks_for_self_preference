#!/bin/bash
#SBATCH --job-name=judge_swap_null_author_obfuscation
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=logs/judge_swap_null_author_obfuscation/%x_%j.log
#SBATCH --error=logs/judge_swap_null_author_obfuscation/%x_%j.log

echo "=========================================="
echo "JUDGE SWAP NULL HYPOTHESIS TEST - AUTHOR_OBFUSCATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="

# Activate environment
source .venv/bin/activate

# Create log directory
mkdir -p logs/judge_swap_null_author_obfuscation

# Parse arguments (default to smoke test with small models)
SMOKE_TEST=${1:-true}
# SMOKE_MODELS=${2:-"Meta-Llama-3.1-8B-Instruct-Turbo"}
SMOKE_MODELS=${2:-"Llama-4-Scout-17B-16E-Instruct"}
SMOKE_EXAMPLES=${3:-10}
OUTPUT_DIR=${4:-judge_swap_null_author_obfuscation}

echo "Configuration:"
echo "  Smoke test: $SMOKE_TEST"
echo "  Smoke models: $SMOKE_MODELS"
echo "  Smoke examples: $SMOKE_EXAMPLES"
echo "  Output dir: $OUTPUT_DIR"

# Run the script
if [ "$SMOKE_TEST" = true ]; then
    echo "Running smoke test..."
    python -u run_judge_swap_null_author_obfuscation.py \
        --smoke_test \
        --smoke_models $SMOKE_MODELS \
        --smoke_examples $SMOKE_EXAMPLES \
        --output_dir "$OUTPUT_DIR"
else
    echo "Running full judge swap test..."
    python -u run_judge_swap_null_author_obfuscation.py \
        --judge_models $SMOKE_MODELS \
        --output_dir "$OUTPUT_DIR"
fi

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="