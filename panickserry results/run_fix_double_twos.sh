#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FILES=(
    "gpt3.5_hermes3-8vsllama2_cnn__gpt5_judging.json"
    "gpt3.5_hermes3-8vsllama2_xsum__gpt5_judging.json"
    "gpt3.5_llama3.1-8bvsllama2_cnn__gpt5_judging.json"
    "gpt3.5_llama3.1-8bvsllama2_xsum__gpt5_judging.json"
    "gpt3.5vsllama2_xsum__gpt5_judging.json"
)

for file in "${FILES[@]}"; do
    input="$file"
    output="${input%.json}2.json"
    echo "Processing: $file"
    python3 "fix_double_twos.py" "$input" -o "$output"
    echo ""
done
