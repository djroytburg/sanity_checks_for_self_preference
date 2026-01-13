

source .venv/bin/activate
models=("Llama-3.1-8B-Instruct" "gemma-2-9b-it" "Qwen2.5-7B-Instruct")
datasets=("alpaca_eval" "translation" "truthfulness")
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    if [ -f "./activation_cache/dbg/${dataset}/${model}_activations.pkl" ]; then
      echo "Activation cache for model $model and dataset $dataset already exists. Skipping..."
      continue
    fi
    python3 residual_classifier/activations_cache.py --model "$model" --paper dbg --dataset "$dataset" --data_dir dbg-score-paper --output_dir activation_cache 
  done
done