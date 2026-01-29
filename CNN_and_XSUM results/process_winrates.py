#!/usr/bin/env python3
"""
Process CNN_and_XSUM results to calculate reference winrates.
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

BASE_DIR = Path("/home/ubuntu/sanity_checks_for_self_preference/CNN_and_XSUM results")


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


def calculate_lsp_ilsp_data(eval_data, oracle_data):
    """
    Calculate LSP and ILSP data by comparing proxy judgments with oracle judgments.

    LSP (Logprob-Supported Prediction): Cases where proxy's higher logprob choice matches oracle
    ILSP (Inverse LSP): Cases where proxy's higher logprob choice differs from oracle

    Returns dict with n_lsp, n_ilsp, lsp_ids, ilsp_ids
    """
    # Create a mapping from article_index to oracle answer
    oracle_map = {}
    for item in oracle_data:
        article_idx = item.get("article_index")
        orig_answer = item.get("original_order", {}).get("answer")
        flip_answer = item.get("flipped_order", {}).get("answer")

        if article_idx and orig_answer and flip_answer:
            # Oracle reference wins if orig=1 and flip=2
            oracle_wins = (orig_answer == "1" and flip_answer == "2")
            oracle_map[article_idx] = oracle_wins

    lsp_ids = []
    ilsp_ids = []

    for item in eval_data:
        article_idx = item.get("article_index")
        if article_idx not in oracle_map:
            continue

        oracle_wins = oracle_map[article_idx]

        # Get proxy's logprobs
        orig_logprobs = item.get("original_order", {}).get("top_logprobs", [])
        flip_logprobs = item.get("flipped_order", {}).get("top_logprobs", [])

        if not orig_logprobs or not flip_logprobs:
            continue

        # Find which choice has higher probability in each order
        # orig_logprobs[0] is the top choice for original order
        # We need to check if choice "1" or "2" has higher prob
        orig_choice_1_prob = 0
        orig_choice_2_prob = 0
        for lp in orig_logprobs:
            if lp.get("token") == "1":
                orig_choice_1_prob = lp.get("probability", 0)
            elif lp.get("token") == "2":
                orig_choice_2_prob = lp.get("probability", 0)

        flip_choice_1_prob = 0
        flip_choice_2_prob = 0
        for lp in flip_logprobs:
            if lp.get("token") == "1":
                flip_choice_1_prob = lp.get("probability", 0)
            elif lp.get("token") == "2":
                flip_choice_2_prob = lp.get("probability", 0)

        # Proxy predicts reference wins if:
        # - In original order, "1" has higher prob
        # - In flipped order, "2" has higher prob
        proxy_predicts_ref_wins = (orig_choice_1_prob > orig_choice_2_prob and
                                   flip_choice_2_prob > flip_choice_1_prob)

        # LSP: proxy prediction matches oracle
        # ILSP: proxy prediction differs from oracle
        if proxy_predicts_ref_wins == oracle_wins:
            lsp_ids.append(article_idx)
        else:
            ilsp_ids.append(article_idx)

    return {
        "n_lsp": len(lsp_ids),
        "n_ilsp": len(ilsp_ids),
        "lsp_ids": lsp_ids,
        "ilsp_ids": ilsp_ids
    }


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


def find_proxy_subdirs(base_path, judge_model):
    """
    Find all proxy subdirectories within the judge and eval folders.
    Excludes the main judge model's own subdirectory.

    Returns list of (proxy_name, judge_path, eval_path) tuples
    """
    judge_base = base_path / judge_model / "judge"
    eval_base = base_path / judge_model / "eval"

    proxies = []

    if judge_base.exists():
        for subdir in judge_base.iterdir():
            if subdir.is_dir() and subdir.name != judge_model:
                proxy_name = subdir.name
                judge_path = subdir
                eval_path = eval_base / proxy_name if eval_base.exists() else None

                if eval_path and eval_path.exists():
                    proxies.append((proxy_name, judge_path, eval_path))

    return proxies


def infer_proxy_names_from_files(judge_path, eval_path, judge_model):
    """
    For datasets where proxy files are not in subdirectories (like xsum),
    infer proxy names from filenames.

    Returns list of (proxy_name, judge_path, eval_path) tuples
    """
    proxies_found = set()

    # Common proxy model patterns to look for
    proxy_patterns = [
        'hermes-3-8b', 'hermes3-8b', 'hermes8b',
        'hermes-3-405b', 'hermes3-405b', 'hermes405b',
        'llama-3.1-8b', 'llama3.1-8b', 'llama31-8b',
        'llama-3.2-3b', 'llama3.2-3b', 'llama32-3b',
        'llama-3.1-405b', 'llama3.1-405b', 'llama405b',
        'deepseekv3', 'deepseek-v3'
    ]

    # Scan judge directory for proxy files
    if judge_path.exists():
        for file in judge_path.glob("*.json"):
            fname = file.name.lower()
            # Skip oracle files
            if "gpt5" in fname and judge_model.lower().replace("-", "").replace(".", "") in fname:
                continue

            # Check for proxy patterns
            for pattern in proxy_patterns:
                pattern_clean = pattern.lower().replace("-", "").replace(".", "")
                fname_clean = fname.replace("-", "").replace(".", "").replace("_", "")
                if pattern_clean in fname_clean:
                    proxies_found.add(pattern)
                    break

    # Normalize proxy names and return
    proxies = []
    for proxy_name in proxies_found:
        proxies.append((proxy_name, judge_path, eval_path))

    return proxies


def find_matching_file(directory, evaluator, evaluatee, is_judge_file=False):
    """
    Find a file in the directory that matches the evaluator vs evaluatee pattern.
    Handles various filename formats.

    Args:
        directory: Directory to search in
        evaluator: The evaluator model name (e.g., "GPT-4")
        evaluatee: The evaluatee model name (e.g., "llama2", "human")
        is_judge_file: If True, prefer files with gpt5 in the name (oracle judge files)
    """
    if not directory.exists():
        return None

    # Create variations of the names to match against
    def get_name_variations(name):
        """Generate various possible name formats"""
        variations = set()
        name_lower = name.lower()

        # Original
        variations.add(name_lower)

        # Without spaces, dashes, dots, underscores
        clean = name_lower.replace("-", "").replace(".", "").replace("_", "").replace(" ", "")
        variations.add(clean)

        # Common model name patterns
        if "gpt" in name_lower:
            if "3.5" in name_lower or "35" in name_lower:
                variations.update(["gpt35", "gpt3.5", "gpt_3.5", "gpt-3.5"])
            elif "4" in name_lower:
                variations.update(["gpt4", "gpt-4", "gpt_4"])

        if "llama" in name_lower:
            if "2" in name_lower:
                variations.update(["llama2", "llama-2", "llama_2"])
            variations.add("llama")

        if "hermes" in name_lower:
            variations.update(["hermes", "hermes3", "hermes-3"])
            if "405b" in name_lower:
                variations.update(["hermes405b", "hermes-405b", "hermes3-405b"])
            if "8b" in name_lower:
                variations.update(["hermes8b", "hermes-8b", "hermes3-8b"])

        return variations

    eval_variations = get_name_variations(evaluator)
    evalee_variations = get_name_variations(evaluatee)

    # Try to find matching files
    candidates = []

    for file in directory.glob("*.json"):
        fname = file.name.lower()

        # Skip numbered duplicates (but not if it's the only match)
        is_duplicate = any(fname.endswith(f"{i}.json") for i in range(2, 10))

        fname_clean = fname.replace("-", "").replace(".", "").replace("_", "")

        # Check if both evaluator and evaluatee appear in filename
        eval_match = any(var in fname_clean for var in eval_variations)
        evalee_match = any(var in fname_clean for var in evalee_variations)

        if eval_match and evalee_match:
            # Score the match
            score = 0
            if is_judge_file and "gpt5" in fname:
                score += 10  # Prefer oracle judge files for judge directory
            if not is_duplicate:
                score += 5  # Prefer non-duplicates

            # Prefer more specific matches
            for var in eval_variations:
                if var in fname_clean and len(var) > 3:
                    score += len(var)
            for var in evalee_variations:
                if var in fname_clean and len(var) > 3:
                    score += len(var)

            candidates.append((score, file, is_duplicate))

    if not candidates:
        return None

    # Sort by score (highest first), then by whether it's a duplicate (non-duplicates first)
    candidates.sort(key=lambda x: (x[0], not x[2]), reverse=True)

    return candidates[0][1]


def process_all():
    """Process all files and create output JSON files."""
    all_files = find_oracle_files()

    datasets = {
        "cnn": BASE_DIR / "cnn_results" / "cnn_eval",
        "xsum": BASE_DIR / "xsum_result" / "xsum_eval"
    }

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
                # Load oracle data
                with open(filepath, 'r') as f:
                    oracle_data = json.load(f)

                ref_winrate = calculate_reference_winrate(oracle_data)

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

                # Process proxies
                base_path = datasets[dataset]
                proxies = find_proxy_subdirs(base_path, parent_judge)

                # If no proxy subdirectories found, try flat file structure
                if not proxies:
                    judge_path = base_path / parent_judge / "judge"
                    eval_path = base_path / parent_judge / "eval"
                    proxies = infer_proxy_names_from_files(judge_path, eval_path, parent_judge)

                for proxy_name, proxy_judge_path, proxy_eval_path in proxies:
                    print(f"    Processing proxy: {proxy_name}")

                    try:
                        # For proxy files in GPT-4/judge/hermes-3-405b/, we're looking for:
                        # Files where parent_judge (GPT-4) evaluates proxy vs evaluatee
                        # Example: "hermes405bvshuman_gpt4.json" = GPT-4 judging hermes405b vs human
                        # So we look for files matching: proxy_name vs evaluatee
                        proxy_judge_file = find_matching_file(proxy_judge_path, proxy_name, evaluatee, is_judge_file=True)
                        proxy_eval_file = find_matching_file(proxy_eval_path, proxy_name, evaluatee, is_judge_file=False)

                        if not proxy_judge_file or not proxy_eval_file:
                            print(f"      -> Skipping {proxy_name}: missing files (judge={proxy_judge_file is not None}, eval={proxy_eval_file is not None})")
                            continue

                        # Load proxy data
                        with open(proxy_judge_file, 'r') as f:
                            proxy_judge_data = json.load(f)

                        with open(proxy_eval_file, 'r') as f:
                            proxy_eval_data = json.load(f)

                        # Calculate proxy winrate
                        proxy_winrate = calculate_reference_winrate(proxy_judge_data)

                        # Calculate LSP/ILSP data
                        lsp_ilsp_data = calculate_lsp_ilsp_data(proxy_eval_data, oracle_data)

                        # Calculate additional metrics
                        n_common = lsp_ilsp_data["n_lsp"] + lsp_ilsp_data["n_ilsp"]
                        lsp_rate = lsp_ilsp_data["n_lsp"] / n_common if n_common > 0 else 0
                        ilsp_rate = lsp_ilsp_data["n_ilsp"] / n_common if n_common > 0 else 0

                        # Add proxy data to output
                        output_data["proxies"][proxy_name] = {
                            "n_common": n_common,
                            "n_lsp": lsp_ilsp_data["n_lsp"],
                            "n_ilsp": lsp_ilsp_data["n_ilsp"],
                            "lsp_rate": lsp_rate,
                            "ilsp_rate": ilsp_rate,
                            "reference_winrate": ref_winrate,
                            "proxy_winrate": proxy_winrate,
                            "lsp_ids": lsp_ilsp_data["lsp_ids"],
                            "ilsp_ids": lsp_ilsp_data["ilsp_ids"]
                        }

                        print(f"      -> Added {proxy_name}: proxy_winrate={proxy_winrate:.4f}, lsp_rate={lsp_rate:.4f}")

                    except Exception as e:
                        print(f"      -> ERROR processing proxy {proxy_name}: {e}")

                # Clean filename
                safe_evaluator = evaluator.replace("/", "-").replace(" ", "_")
                safe_evaluatee = evaluatee.replace("/", "-").replace(" ", "_")
                output_filename = f"{safe_evaluator}_vs_{safe_evaluatee}.json"
                output_path = output_dir / output_filename

                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)

                print(f"    -> Created: {output_filename} (reference winrate: {ref_winrate:.4f})")
                processed_pairs.add(pair_key)

            except Exception as e:
                print(f"    ERROR processing {filepath}: {e}")


if __name__ == "__main__":
    process_all()
    print("\nDone!")
