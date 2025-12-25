# extract_responses_from_preferences.py: Extract model responses from preference files
# Written by: Dani
# Created: 2025-12-23, 14:45 EST
# Last Modified: 2025-12-23, 14:45 EST

"""
Extracts model responses from existing preference files and saves them
in the format expected by gen_answering.py output.

This avoids re-running expensive model inference when responses already exist
in the preference evaluation files.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(path, data):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def extract_responses_from_preferences(pref_dir, output_dir, dataset):
    """
    Extract model responses from preference files.
    
    Args:
        pref_dir: Directory containing preference files (e.g., model_preferences_fullset/alpaca_eval_500_id_wr/)
        output_dir: Output directory for responses (e.g., model_responses_fullset/question_answering/alpaca_eval_500_id_wr/)
        dataset: Dataset name (e.g., alpaca_eval)
    """
    pref_path = Path(pref_dir) / f"{dataset}_500_id_wr"
    
    if not pref_path.exists():
        print(f"ERROR: Preference directory not found: {pref_path}")
        return
    
    # Dictionary to collect responses by model
    model_responses = defaultdict(dict)
    
    # Scan all evaluator directories
    print(f"Scanning preference files in {pref_path}")
    evaluator_dirs = [d for d in pref_path.iterdir() if d.is_dir() and d.name.startswith('evaluator_')]
    
    for eval_dir in tqdm(evaluator_dirs, desc="Processing evaluators"):
        # Process all preference files
        pref_files = list(eval_dir.glob('average_*.jsonl')) + list(eval_dir.glob('merge_*.jsonl'))
        
        for pref_file in pref_files:
            data = load_jsonl(pref_file)
            
            for item in data:
                # Extract model names and responses
                for key, value in item.items():
                    if key.endswith('_response'):
                        model_name = key.replace('_response', '')
                        example_id = item['id']
                        
                        # Store response with metadata
                        if example_id not in model_responses[model_name]:
                            # Need to extract query/golden_response from somewhere
                            # For now, just store the response
                            model_responses[model_name][example_id] = {
                                'id': example_id,
                                'model_response': value
                            }
    
    # Save extracted responses
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracted responses for {len(model_responses)} models")
    
    for model_name, responses in model_responses.items():
        # Convert dict to sorted list
        response_list = [responses[id] for id in sorted(responses.keys())]
        
        output_file = output_path / f"{model_name}.jsonl"
        save_jsonl(output_file, response_list)
        print(f"  {model_name}: {len(response_list)} responses → {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract responses from preference files")
    parser.add_argument("--pref_dir", type=str, default="model_preferences_fullset/",
                        help="Directory containing preference files")
    parser.add_argument("--output_dir", type=str, default="model_responses_fullset/",
                        help="Output directory for response files")
    parser.add_argument("--dataset", type=str, default="alpaca_eval",
                        help="Dataset name")
    
    args = parser.parse_args()
    
    extract_responses_from_preferences(args.pref_dir, args.output_dir, args.dataset)
    print("\n✓ Response extraction complete!")


if __name__ == "__main__":
    main()
