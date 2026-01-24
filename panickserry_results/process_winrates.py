#!/usr/bin/env python3
"""
Process panickserry_results to calculate reference winrates.
Creates {dataset}_winrates folders with JSON files for each judge-reference pair.

Reference winrate is the rate at which the evaluator (first model) wins according to GPT-5 oracle.
- If original_order.answer == "1" and flipped_order.answer == "2", the evaluator consistently wins.

Only processes files from main judge folders:
- GPT-3.5/judge/GPT-3.5/
- GPT-4/judge/GPT-4/
"""

import json
import os
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/home/ubuntu/sanity_checks_for_self_preference/panickserry_results")


def calculate_reference_winrate(data):
    """
    Calculate reference winrate based on original_order and flipped_order answers.
    Reference (evaluator) wins if original_order.answer == "1" and flipped_order.answer == "2"
    This means summary1 was consistently chosen regardless of presentation order.
    """
    total = 0
    reference_wins = 0

    for item in data:
        original_answer = item.get("original_order", {}).get("answer")
        flipped_answer = item.get("flipped_order", {}).get("answer")

        if original_answer and flipped_answer:
            total += 1
            if original_answer == "1" and flipped_answer == "2":
                reference_wins += 1

    if total == 0:
        return 0.0

    return reference_wins / total


def normalize_model_name(name):
    """Normalize model names for consistency and readability."""
    name = name.lower().strip()

    # Remove common suffixes/patterns that shouldn't be part of model names
    patterns_to_remove = [
        "_judging", "judging",
        "_orig_paper", "orig_paper",
        "_cnn", "_xsum",
        "_judge", "judge",
        "gpt5", "_"
    ]

    for pattern in patterns_to_remove:
        name = name.replace(pattern, "")

    name = name.strip()

    # Common normalizations
    mappings = {
        "gpt35": "GPT-3.5",
        "gpt3.5": "GPT-3.5",
        "gpt4": "GPT-4",
        "llama2": "llama2",
        "llama": "llama2",
        "human": "human",
    }

    # Try exact match first
    if name in mappings:
        return mappings[name]

    # Try without special chars
    name_clean = name.replace("-", "").replace(".", "").replace("_", "")
    for key, val in mappings.items():
        key_clean = key.replace("-", "").replace(".", "").replace("_", "")
        if name_clean == key_clean:
            return val

    return name


def parse_oracle_file(filename):
    """
    Parse oracle (GPT-5) judge files to extract evaluator and evaluatee.

    Expected patterns:
    - gpt5_gpt35vshuman.json -> evaluator: GPT-3.5, evaluatee: human
    - gpt3.5vsllama2_cnn_gpt5_judging.json -> evaluator: GPT-3.5, evaluatee: llama2
    - gpt4_vs_gpt35_cnn_gpt5_judging.json -> evaluator: GPT-4, evaluatee: GPT-3.5
    """
    fname = filename.lower()

    # Only process GPT-5 oracle files
    if not ("gpt5" in fname):
        return None, None

    # Remove file extension
    core = fname.replace(".json", "")

    # Handle different "vs" patterns - split first, then clean each part
    vs_patterns = ["_vs_", "_vs", "vs"]
    for pattern in vs_patterns:
        if pattern in core:
            parts = core.split(pattern, 1)
            if len(parts) == 2:
                evaluator = normalize_model_name(parts[0])
                evaluatee = normalize_model_name(parts[1])
                return evaluator, evaluatee

    return None, None


def find_oracle_files():
    """
    Find GPT-5 oracle judge files from main judge folders only.
    Only processes:
    - GPT-3.5/judge/GPT-3.5/
    - GPT-4/judge/GPT-4/
    """
    datasets = {
        "cnn": BASE_DIR / "cnn_results" / "cnn_eval",
        "xsum": BASE_DIR / "xsum_result" / "xsum_eval"
    }

    results = defaultdict(list)

    for dataset, base_path in datasets.items():
        for judge_model in ["GPT-3.5", "GPT-4"]:
            # Only look in the main judge folder (e.g., GPT-3.5/judge/GPT-3.5/)
            main_judge_path = base_path / judge_model / "judge" / judge_model

            if main_judge_path.exists() and main_judge_path.is_dir():
                for file in main_judge_path.glob("*.json"):
                    fname = file.name.lower()
                    # Only GPT-5 oracle files
                    if "gpt5" in fname and "vs" in fname:
                        evaluator, evaluatee = parse_oracle_file(file.name)
                        if evaluator and evaluatee:
                            results[dataset].append((evaluator, evaluatee, file, judge_model))
            else:
                # For XSUM, files might be directly in the judge folder
                judge_path = base_path / judge_model / "judge"
                if judge_path.exists():
                    for file in judge_path.glob("*.json"):
                        fname = file.name.lower()
                        # Only GPT-5 oracle files that match the judge model
                        if "gpt5" in fname and "vs" in fname:
                            evaluator, evaluatee = parse_oracle_file(file.name)
                            # Only include if evaluator matches the judge model
                            if evaluator and evaluatee:
                                judge_model_normalized = normalize_model_name(judge_model)
                                if evaluator == judge_model_normalized:
                                    results[dataset].append((evaluator, evaluatee, file, judge_model))

    return results


def process_all():
    """Process all files and create output JSON files."""
    all_files = find_oracle_files()

    for dataset, file_list in all_files.items():
        output_dir = BASE_DIR / f"{dataset}_winrates"
        output_dir.mkdir(exist_ok=True)

        # Track processed pairs to avoid duplicates
        processed_pairs = set()

        print(f"\n=== Processing {dataset} ===")

        for evaluator, evaluatee, filepath, parent_judge in file_list:
            pair_key = (evaluator, evaluatee)

            # Skip duplicates (files ending with 2, 3, etc.)
            fname = filepath.name
            if any(fname.endswith(f"{i}.json") for i in range(2, 10)):
                continue

            # Skip if already processed this pair
            if pair_key in processed_pairs:
                continue

            print(f"  Processing: {evaluator} vs {evaluatee}")
            print(f"    Source: {filepath.name}")

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                ref_winrate = calculate_reference_winrate(data)

                # Create output file in format expected by analyze script
                output_data = {
                    "reference_evaluator": evaluator,
                    "reference_evaluatee": evaluatee,
                    "proxies": {
                        "oracle_gpt5": {
                            "reference_winrate": ref_winrate
                        }
                    }
                }

                # Clean filename
                safe_evaluator = evaluator.replace("/", "-").replace(" ", "_")
                safe_evaluatee = evaluatee.replace("/", "-").replace(" ", "_")
                output_filename = f"{safe_evaluator}_vs_{safe_evaluatee}.json"
                output_path = output_dir / output_filename

                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)

                print(f"    -> Created: {output_filename} (winrate: {ref_winrate:.4f})")
                processed_pairs.add(pair_key)

            except Exception as e:
                print(f"    ERROR processing {filepath}: {e}")


if __name__ == "__main__":
    process_all()
    print("\nDone!")
