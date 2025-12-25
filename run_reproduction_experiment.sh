#!/bin/bash
#SBATCH --job-name=repro_exp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/reproduction/repro_exp_%j.log
#SBATCH --error=logs/reproduction/repro_exp_%j.log

# run_reproduction_experiment.sh: SLURM runner for full reproduction experiments
# Written by: Dani
# Created: 2025-12-21 20:50 EST
# Last Modified: 2025-12-21 23:30 EST

set -euo pipefail

mkdir -p logs/reproduction

source .venv/bin/activate

echo "[repro] starting: $(date)"
echo "[repro] python: $(which python)"

# Function to map full model name to short name
# Maps HuggingFace model IDs to the short names used in llm-sp/sp filenames
get_model_short_name() {
    local model="$1"
    case "$model" in
        # Qwen models (qwen-2.5-{3b,7b,14b,32b,72b} available)
        "Qwen/Qwen2.5-3B-Instruct") echo "qwen-2.5-3b" ;;
        "Qwen/Qwen2.5-7B-Instruct") echo "qwen-2.5-7b" ;;
        "Qwen/Qwen2.5-14B-Instruct") echo "qwen-2.5-14b" ;;
        "Qwen/Qwen2.5-32B-Instruct") echo "qwen-2.5-32b" ;;
        "Qwen/Qwen2.5-72B-Instruct") echo "qwen-2.5-72b" ;;
        
        # Llama models (llama-3.1-{8b,70b}, llama-3.2-3b, llama-3.3-70b available)
        "meta-llama/Llama-3.1-8B-Instruct") echo "llama-3.1-8b" ;;
        "meta-llama/Llama-3.1-70B-Instruct") echo "llama-3.1-70b" ;;
        "meta-llama/Llama-3.2-1B-Instruct") echo "llama-3.2-1b" ;;
        "meta-llama/Llama-3.2-3B-Instruct") echo "llama-3.2-3b" ;;
        "meta-llama/Llama-3.3-70B-Instruct") echo "llama-3.3-70b" ;;
        
        # Gemma models (gemma-2-{9b,27b} available)
        "google/gemma-2-2b-it") echo "gemma-2-2b" ;;
        "google/gemma-2-9b-it") echo "gemma-2-9b" ;;
        "google/gemma-2-27b-it") echo "gemma-2-27b" ;;
        
        # Mistral models (used as evaluatees, not judges in llm-sp)
        "mistralai/Mistral-7B-Instruct-v0.3") echo "mistral-7b-v0.3" ;;
        "mistralai/Mistral-Small-Instruct-2409") echo "mistral-small" ;;
        
        # Phi models (used as evaluatees, not judges in llm-sp)
        "microsoft/Phi-3.5-mini-instruct") echo "phi-3.5-mini" ;;
        
        # Default: try to extract from path and normalize
        *) 
            echo "[repro] WARNING: Unknown model '$model', attempting auto-extraction" >&2
            echo "$model" | sed 's/.*\///g' | tr '[:upper:]' '[:lower:]' | sed 's/-instruct//g' 
            ;;
    esac
}

# Parse arguments
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
BENCHMARK="${2:-math500}"
JUDGE_FAMILY="${3:-qwen}"
N_SAMPLES="${4:-100}"

# Get judge short name from MODEL
JUDGE_SHORT=$(get_model_short_name "$MODEL")

# Auto-detect data file from llm-sp/sp directory
# Expected pattern: llm-sp/sp/{benchmark}/{judge_family}/*.jsonl
SP_DIR="llm-sp/sp/${BENCHMARK}/${JUDGE_FAMILY}"

if [ ! -d "$SP_DIR" ]; then
    echo "[repro] ERROR: Directory not found: $SP_DIR"
    exit 1
fi

# Find first available jsonl file in the directory
DATA=$(find "$SP_DIR" -name "*.jsonl" | head -1)

if [ -z "$DATA" ]; then
    echo "[repro] ERROR: No JSONL files found in $SP_DIR"
    exit 1
fi

# Extract evaluatee model name from the data filename
# Pattern: {judge_short}_{evaluatee_short}_eval.jsonl
# We need to find a file that matches our judge_short
DATA=$(find "$SP_DIR" -name "${JUDGE_SHORT}_*_eval.jsonl" | head -1)

if [ -z "$DATA" ]; then
    echo "[repro] WARNING: No data file found for judge '$JUDGE_SHORT' in $SP_DIR"
    echo "[repro] Available files:"
    ls -1 "$SP_DIR"/*.jsonl 2>/dev/null || echo "  (none)"
    echo "[repro] Falling back to first available file"
    DATA=$(find "$SP_DIR" -name "*.jsonl" | head -1)
fi

# Extract evaluatee from filename
FILENAME=$(basename "$DATA" "_eval.jsonl")
DATA_JUDGE_SHORT=$(echo "$FILENAME" | cut -d'_' -f1)
EVALUATEE_SHORT=$(echo "$FILENAME" | cut -d'_' -f2-)

# Validate judge match
if [ "$DATA_JUDGE_SHORT" != "$JUDGE_SHORT" ]; then
    echo "[repro] ERROR: Judge mismatch!"
    echo "[repro]   Model requested: $MODEL"
    echo "[repro]   Model short name: $JUDGE_SHORT"
    echo "[repro]   Data judge: $DATA_JUDGE_SHORT"
    echo "[repro]"
    echo "[repro] Available judges in $JUDGE_FAMILY family:"
    ls -1 "$SP_DIR"/*.jsonl 2>/dev/null | sed 's/.*\///g' | sed 's/_eval.jsonl//g' | cut -d'_' -f1 | sort -u | sed 's/^/[repro]     /g'
    echo "[repro]"
    echo "[repro] Please use one of the following models for $JUDGE_FAMILY:"
    case "$JUDGE_FAMILY" in
        qwen)
            echo "[repro]   - Qwen/Qwen2.5-3B-Instruct"
            echo "[repro]   - Qwen/Qwen2.5-7B-Instruct"
            echo "[repro]   - Qwen/Qwen2.5-14B-Instruct"
            echo "[repro]   - Qwen/Qwen2.5-32B-Instruct"
            echo "[repro]   - Qwen/Qwen2.5-72B-Instruct"
            ;;
        llama)
            echo "[repro]   - meta-llama/Llama-3.1-8B-Instruct"
            echo "[repro]   - meta-llama/Llama-3.1-70B-Instruct"
            echo "[repro]   - meta-llama/Llama-3.2-3B-Instruct"
            echo "[repro]   - meta-llama/Llama-3.3-70B-Instruct"
            ;;
        gemma)
            echo "[repro]   - google/gemma-2-9b-it"
            echo "[repro]   - google/gemma-2-27b-it"
            ;;
        *)
            echo "[repro]   (unknown family)"
            ;;
    esac
    exit 1
fi

echo "[repro] running reproduction experiment"
echo "[repro] model: $MODEL"
echo "[repro] benchmark: $BENCHMARK"
echo "[repro] judge family: $JUDGE_FAMILY"
echo "[repro] judge short: $JUDGE_SHORT"
echo "[repro] evaluatee short: $EVALUATEE_SHORT"
echo "[repro] data source: $DATA"
echo "[repro] n_samples: $N_SAMPLES"

python reproduce_paper_experiments.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output "reproduction_results" \
  --benchmark "$BENCHMARK" \
  --judge_family "$JUDGE_FAMILY" \
  --judge_short "$JUDGE_SHORT" \
  --evaluatee_short "$EVALUATEE_SHORT" \
  --n_samples "$N_SAMPLES" \
  --seed 42

echo "[repro] done: $(date)"
