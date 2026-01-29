#!/bin/bash

# Script to run analyze_judge_swap for comparing J(J vs R) with J(K vs R)
# where J = GPT-3.5, K = other models, R = llama2 (reference)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define model comparisons and datasets
MODELS=("hermes-3-8b" "llama3.1-8b")
DATASETS=("cnn" "xsum")

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        # Paths to statistics files
        j_vs_r_stats="${dataset}/gpt_3.5_vs_llama2/cnn_self_preference_statistics.json"
        k_vs_r_stats="${dataset}/${model}_vs_llama2/self_preference_statistics.json"
        output_dir="${dataset}/judge_swap_${model}"

        echo "=========================================="
        echo "Processing judge swap: ${dataset} dataset"
        echo "  J(J vs R): ${j_vs_r_stats}"
        echo "  J(K vs R): ${k_vs_r_stats}"
        echo "  Output dir: ${output_dir}"
        echo "=========================================="

        # Determine which script to use based on dataset
        if [ "$dataset" == "cnn" ]; then
            script="analyze_judge_swap_cnn.py"
        else
            script="analyze_judge_swap_cnn.py"  # Assuming same script works for both
        fi

        python3 "$script" \
            --j_vs_r_stats "${j_vs_r_stats}" \
            --k_vs_r_stats "${k_vs_r_stats}" \
            --output_dir "${output_dir}"

        echo ""
        echo "Completed judge swap analysis for ${model} on ${dataset}"
        echo ""
    done
done

echo "All judge swap analyses complete!"
