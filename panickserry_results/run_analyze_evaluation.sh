#!/bin/bash

# Script to run analyze_evaluation_self_preference.py for multiple model comparisons

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define model comparisons
MODELS=("hermes-3-8b" "llama3.1-8b")
DATASETS=("cnn" "xsum")

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        # Construct file paths
        eval_results="gpt3.5_${model}_vs_llama2_${dataset}.json"
        ground_truth="gpt3.5_${model}vsllama2_${dataset}__gpt5_judging2.json"
        output_dir="${dataset}/${model}_vs_llama2"
        model_name="${model} vs llama2 (${dataset})"

        echo "=========================================="
        echo "Processing: ${model} vs llama2 on ${dataset}"
        echo "  Eval results: ${eval_results}"
        echo "  Ground truth: ${ground_truth}"
        echo "  Output dir: ${output_dir}"
        echo "=========================================="

        python3 analyze_evaluation_self_preference.py \
            --eval_results "${eval_results}" \
            --ground_truth "${ground_truth}" \
            --model_name "${model_name}" \
            --output_dir "${output_dir}"

        echo ""
        echo "Completed: ${model} vs llama2 on ${dataset}"
        echo ""
    done
done

echo "All analyses complete!"
