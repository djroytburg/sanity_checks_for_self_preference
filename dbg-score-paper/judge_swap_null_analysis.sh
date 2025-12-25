#!/bin/bash
#SBATCH --job-name=judge_swap_null
#SBATCH --output=logs/judge_swap_null_%j.log
#SBATCH --error=logs/judge_swap_null_%j.log
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# judge_swap_null_analysis.sh: Run judge swap null hypothesis test
# Tests if self-preference is driven by identity or response quality

echo "=========================================="
echo "JUDGE SWAP NULL HYPOTHESIS TEST"
echo "Started: $(date '+%a %b %d %H:%M:%S %Z %Y')"
echo "=========================================="
echo ""
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo ""

# Activate virtual environment
source ../.venv/bin/activate

# Configuration
DATASET="alpaca_eval_500_id_wr"
PREF_DIR="model_preferences_fullset/${DATASET}"

# Test 1: Llama-3.1-8B-Instruct vs Qwen2.5-7B-Instruct (both judging gemma-2-9b-it)
echo "==========================================  "
echo "TEST 1: Llama-3.1-8B-Instruct (judge) vs Qwen2.5-7B-Instruct (proxy)"
echo "=========================================="
echo ""

JUDGE="Llama-3.1-8B-Instruct"
PROXY="Qwen2.5-7B-Instruct"
OTHER="gemma-2-9b-it"
OUTPUT_DIR="../judge_swap_results/test1_llama8b_qwen7b"

python3 ../judge_swap_null_test.py \
    --pref_dir "$PREF_DIR" \
    --judge_name "$JUDGE" \
    --proxy_name "$PROXY" \
    --other_name "$OTHER" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "Test 1 completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "ERROR: Test 1 failed"
    exit 1
fi

echo ""
echo ""

# Test 2: Qwen2.5-7B-Instruct vs Llama-3.1-8B-Instruct (reversed)
echo "=========================================="
echo "TEST 2: Qwen2.5-7B-Instruct (judge) vs Llama-3.1-8B-Instruct (proxy)"
echo "=========================================="
echo ""

JUDGE="Qwen2.5-7B-Instruct"
PROXY="Llama-3.1-8B-Instruct"
OTHER="gemma-2-9b-it"
OUTPUT_DIR="../judge_swap_results/test2_qwen7b_llama8b"

python3 ../judge_swap_null_test.py \
    --pref_dir "$PREF_DIR" \
    --judge_name "$JUDGE" \
    --proxy_name "$PROXY" \
    --other_name "$OTHER" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "Test 2 completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "ERROR: Test 2 failed"
    exit 1
fi

echo ""
echo ""

# Test 3: Llama-3.1-70B-Instruct vs Qwen2.5-72B-Instruct (large models)
echo "=========================================="
echo "TEST 3: Llama-3.1-70B-Instruct (judge) vs Qwen2.5-72B-Instruct (proxy)"
echo "=========================================="
echo ""

JUDGE="Llama-3.1-70B-Instruct"
PROXY="Qwen2.5-72B-Instruct"
OTHER="Llama-3.1-8B-Instruct"
OUTPUT_DIR="../judge_swap_results/test3_llama70b_qwen72b"

python3 ../judge_swap_null_test.py \
    --pref_dir "$PREF_DIR" \
    --judge_name "$JUDGE" \
    --proxy_name "$PROXY" \
    --other_name "$OTHER" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "Test 3 completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "ERROR: Test 3 failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "ALL TESTS COMPLETED"
echo "Ended: $(date '+%a %b %d %H:%M:%S %Z %Y')"
echo "=========================================="
echo ""
echo "Results directories:"
echo "  Test 1: ../judge_swap_results/test1_llama8b_qwen7b"
echo "  Test 2: ../judge_swap_results/test2_qwen7b_llama8b"
echo "  Test 3: ../judge_swap_results/test3_llama70b_qwen72b"
