#!/usr/bin/env python3
"""
proxy_robustness_plot2.py: Analyze average mean difference across (example, judge, reference) triplets.

Creates a plot showing the average mean difference (self-preference minus proxy-preference)
for examples with at least N valid proxies, to visualize how mean difference changes as we
require more proxies per example.

Supports four data sources:
- author_obfuscation: quality dataset from author obfuscation experiments
- dbg: DBG score datasets (alpaca_eval, translation, truthfulness)
- verifiable: Verifiable datasets (math500, mbpp-plus, mmlu)
- panickserry: Summarization datasets (cnn, xsum)

Usage:
    python proxy_robustness_plot2.py --source author_obfuscation
    python proxy_robustness_plot2.py --source dbg --dataset alpaca_eval
    python proxy_robustness_plot2.py --source dbg --dataset all
    python proxy_robustness_plot2.py --source verifiable --dataset math500
    python proxy_robustness_plot2.py --source verifiable --dataset all
    python proxy_robustness_plot2.py --source panickserry --dataset cnn
    python proxy_robustness_plot2.py --source panickserry --dataset all
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Default paths for each source
DEFAULT_PATHS = {
    "author_obfuscation": "author_obfuscation/data/quality/proxies",
    "dbg": "dbg-score-paper/proxy_preference_data",
    "verifiable": "llm-sp-verif",
    "panickserry": "panickserry_results",  # For panickserry, proxy info is in cache files
}

# Default cache paths for preference scores
# For dbg: cache is at judge_swap_null_dbg_results/{dataset}/cache
# For verifiable: cache is at judge_swap_null_verif_smoke2/{dataset}/cache
# For author_obfuscation: preference results in author_obfuscation/data/quality/preference_results/
# For panickserry: cache is at panickserry_results/{dataset}_result/{dataset}/cache
DEFAULT_CACHE_PATHS = {
    "author_obfuscation": "author_obfuscation/data/quality/preference_results",
    "dbg": "judge_swap_null_dbg_results",
    "verifiable": "judge_swap_null_verif_smoke2",
    "panickserry": "panickserry_results",
}

DBG_DATASETS = ["alpaca_eval", "translation", "truthfulness"]
VERIFIABLE_DATASETS = ["math500", "mbpp-plus", "mmlu"]
PANICKSERRY_DATASETS = ["cnn", "xsum"]


def load_proxy_files_author_obfuscation(proxy_dir: Path) -> list:
    """
    Load proxy files from author_obfuscation format.
    
    Files: evaluator_{judge}_vs_{reference}.json
    Structure: reference_evaluator, reference_evaluatee, proxies with lsp_ids/ilsp_ids
    """
    proxy_files = list(proxy_dir.glob("evaluator_*.json"))
    data = []
    
    for pf in proxy_files:
        with open(pf) as f:
            content = json.load(f)
        
        # Convert to common format
        proxies_converted = {}
        for proxy_name, proxy_info in content["proxies"].items():
            proxies_converted[proxy_name] = {
                "lsp_ids": proxy_info.get("lsp_ids", []),
                "ilsp_ids": proxy_info.get("ilsp_ids", []),
            }
        
        data.append({
            "judge": content["reference_evaluator"],
            "reference": content["reference_evaluatee"],
            "proxies": proxies_converted,
            "dataset": "quality",
            "file": pf.name
        })
    
    return data


def load_proxy_files_dbg(proxy_dir: Path, datasets: list) -> list:
    """
    Load proxy files from dbg-score format.
    
    Files: judge_{judge}_vs_{reference}.json in dataset subdirectories
    Structure: judge_name, reference_name, proxies with data.lsp/data.ilsp containing objects with id field
    """
    data = []
    
    for dataset in datasets:
        dataset_dir = proxy_dir / dataset
        if not dataset_dir.exists():
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue
        
        proxy_files = list(dataset_dir.glob("judge_*.json"))
        
        for pf in proxy_files:
            with open(pf) as f:
                content = json.load(f)
            
            # Convert to common format - extract IDs from data objects
            proxies_converted = {}
            for proxy_name, proxy_info in content["proxies"].items():
                lsp_data = proxy_info.get("data", {}).get("lsp", [])
                ilsp_data = proxy_info.get("data", {}).get("ilsp", [])
                
                # Extract IDs from the data objects
                lsp_ids = [item["id"] for item in lsp_data if "id" in item]
                ilsp_ids = [item["id"] for item in ilsp_data if "id" in item]
                
                proxies_converted[proxy_name] = {
                    "lsp_ids": lsp_ids,
                    "ilsp_ids": ilsp_ids,
                }
            
            data.append({
                "judge": content["judge_name"],
                "reference": content["reference_name"],
                "proxies": proxies_converted,
                "dataset": dataset,
                "file": pf.name
            })
    
    return data


def load_proxy_files_verifiable(proxy_dir: Path, datasets: list) -> list:
    """
    Load proxy files from verifiable format.
    
    Structure: llm-sp-verif/{dataset}/{family}/proxies/{judge}.json
    Each file contains: {reference} -> {proxy} -> data.lsp/data.ilsp with J_v_R.example_id
    """
    data = []
    
    for dataset in datasets:
        dataset_dir = proxy_dir / dataset
        if not dataset_dir.exists():
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue
        
        # Find all proxies folders (can be in family subdirectories)
        proxy_dirs = list(dataset_dir.glob("*/proxies"))
        
        for pdir in proxy_dirs:
            proxy_files = list(pdir.glob("*.json"))
            
            for pf in proxy_files:
                # Judge name is the filename (e.g., llama-3.1-8b.json -> llama-3.1-8b)
                judge_name = pf.stem
                
                with open(pf) as f:
                    content = json.load(f)
                
                # Iterate over references
                for reference_name, ref_data in content.items():
                    # Iterate over proxies for this reference
                    proxies_converted = {}
                    
                    for proxy_name, proxy_info in ref_data.items():
                        lsp_data = proxy_info.get("data", {}).get("lsp", [])
                        ilsp_data = proxy_info.get("data", {}).get("ilsp", [])
                        
                        # Extract example IDs from J_v_R objects
                        # Use to_eval_id (the actual dataset ID) not example_id (internal sequence)
                        lsp_ids = []
                        for item in lsp_data:
                            if "J_v_R" in item and "to_eval_id" in item["J_v_R"]:
                                lsp_ids.append(item["J_v_R"]["to_eval_id"])
                        
                        ilsp_ids = []
                        for item in ilsp_data:
                            if "J_v_R" in item and "to_eval_id" in item["J_v_R"]:
                                ilsp_ids.append(item["J_v_R"]["to_eval_id"])
                        
                        proxies_converted[proxy_name] = {
                            "lsp_ids": lsp_ids,
                            "ilsp_ids": ilsp_ids,
                        }
                    
                    if proxies_converted:  # Only add if there are proxies
                        data.append({
                            "judge": judge_name,
                            "reference": reference_name,
                            "proxies": proxies_converted,
                            "dataset": dataset,
                            "file": pf.name
                        })
    
    return data


def load_proxy_files_panickserry(cache_base_dir: Path, datasets: list) -> list:
    """
    Load proxy information from panickserry cache files.
    
    For panickserry, there are no separate proxy files. Instead, the lsp/ilsp category
    is embedded directly in the cache files. We scan the cache structure to find
    all judge/reference pairs and extract proxy info.
    
    IMPORTANT: Capability matching is required per the paper (Section 4.2):
    XK = {x | G(x, oJ, oR) = G(x, oK, oR)}
    
    This means for ILSP (Judge worse than Ref), we only include proxies that are 
    ALSO worse than Ref on that example. This ensures fair comparison between
    Judge's wrong answer and Proxy's wrong answer.
    
    Structure: cache_base_dir/{dataset}_result/{dataset}/cache/{judge}/{reference}/
    """
    data = []
    
    for dataset in datasets:
        # Handle different path patterns
        cache_dir = cache_base_dir / f"{dataset}_result" / dataset / "cache"
        if not cache_dir.exists():
            # Try alternate path for cnn
            cache_dir = cache_base_dir / f"{dataset}_results" / dataset / "cache"
        if not cache_dir.exists():
            print(f"Warning: Cache directory not found: {cache_dir}")
            continue
        
        # Find all judge directories
        for judge_dir in cache_dir.iterdir():
            if not judge_dir.is_dir():
                continue
            judge_name = judge_dir.name
            
            # Find all reference directories under this judge
            for ref_dir in judge_dir.iterdir():
                if not ref_dir.is_dir():
                    continue
                reference_name = ref_dir.name
                
                # Check if required files exist
                j_vs_r_game1 = ref_dir / "J_vs_R_game1.jsonl"
                k_vs_r_game1 = ref_dir / "K_vs_R_game1.jsonl"
                if not j_vs_r_game1.exists() or not k_vs_r_game1.exists():
                    continue
                
                # Step 1: Get ILSP/LSP example IDs from J_vs_R (judge's correctness)
                j_ilsp_ids = set()
                j_lsp_ids = set()
                for game_file in ["J_vs_R_game1.jsonl", "J_vs_R_game2.jsonl"]:
                    filepath = ref_dir / game_file
                    if not filepath.exists():
                        continue
                    with open(filepath) as f:
                        for line in f:
                            try:
                                item = json.loads(line)
                                example_id = item.get("example_id", "")
                                category = item.get("category", "").lower()
                                if category == "ilsp":
                                    j_ilsp_ids.add(example_id)
                                elif category == "lsp":
                                    j_lsp_ids.add(example_id)
                            except json.JSONDecodeError:
                                continue
                
                # Step 2: Get proxy info from K_vs_R with CAPABILITY MATCHING
                # Only include (example, proxy) pairs where J and K have same outcome vs R
                proxies_converted = defaultdict(lambda: {"lsp_ids": [], "ilsp_ids": []})
                
                for game_file in ["K_vs_R_game1.jsonl", "K_vs_R_game2.jsonl"]:
                    filepath = ref_dir / game_file
                    if not filepath.exists():
                        continue
                    
                    with open(filepath) as f:
                        for line in f:
                            try:
                                item = json.loads(line)
                                proxy_name = item.get("proxy", "")
                                example_id = item.get("example_id", "")
                                k_category = item.get("category", "").lower()
                                
                                if not proxy_name or not example_id:
                                    continue
                                
                                # CAPABILITY MATCHING: J and K must have same outcome vs R
                                # For ILSP: J is ILSP AND K is ILSP (both worse than Ref)
                                # For LSP: J is LSP AND K is LSP (both better than Ref)
                                if example_id in j_ilsp_ids and k_category == "ilsp":
                                    if example_id not in proxies_converted[proxy_name]["ilsp_ids"]:
                                        proxies_converted[proxy_name]["ilsp_ids"].append(example_id)
                                elif example_id in j_lsp_ids and k_category == "lsp":
                                    if example_id not in proxies_converted[proxy_name]["lsp_ids"]:
                                        proxies_converted[proxy_name]["lsp_ids"].append(example_id)
                            except json.JSONDecodeError:
                                continue
                
                if proxies_converted:
                    data.append({
                        "judge": judge_name,
                        "reference": reference_name,
                        "proxies": dict(proxies_converted),
                        "dataset": dataset,
                        "file": f"{judge_name}/{reference_name}"
                    })
    
    return data


def load_proxy_files(source: str, proxy_dir: Path, datasets: list = None) -> list:
    """
    Load proxy files based on source type.
    """
    if source == "author_obfuscation":
        return load_proxy_files_author_obfuscation(proxy_dir)
    elif source == "dbg":
        if datasets is None:
            datasets = DBG_DATASETS
        return load_proxy_files_dbg(proxy_dir, datasets)
    elif source == "verifiable":
        if datasets is None:
            datasets = VERIFIABLE_DATASETS
        return load_proxy_files_verifiable(proxy_dir, datasets)
    elif source == "panickserry":
        if datasets is None:
            datasets = PANICKSERRY_DATASETS
        return load_proxy_files_panickserry(proxy_dir, datasets)
    else:
        raise ValueError(f"Unknown source: {source}")


def load_preference_scores(cache_dir: Path, judge: str, reference: str) -> dict:
    """
    Load preference scores from cache directory (dbg/verifiable format).
    
    Expected format: cache_dir/{judge}/{reference}/J_vs_R_game1.jsonl etc.
    
    Returns:
        Dict mapping example_id -> {
            'self_pref': float,  # sJ(x, oJ, oR) - average across game1 and game2
            'proxy_prefs': dict  # proxy_name -> float (sJ(x, oK, oR))
        }
    """
    results = {}
    
    cache_path = cache_dir / judge / reference
    if not cache_path.exists():
        return results
    
    # Load J vs R preferences (self)
    j_vs_r_game1_file = cache_path / "J_vs_R_game1.jsonl"
    j_vs_r_game2_file = cache_path / "J_vs_R_game2.jsonl"
    
    self_prefs = {}  # example_id -> [game1_pref, game2_pref]
    
    if j_vs_r_game1_file.exists():
        with open(j_vs_r_game1_file) as f:
            for line in f:
                item = json.loads(line)
                # Handle example_id=0 correctly (0 is falsy, so can't use 'or')
                example_id = item.get("example_id")
                if example_id is None:
                    example_id = item.get("id")
                # Preference is prob_response1 (J is first in game1)
                prob = item.get("prob_response1", 0.0)
                if example_id not in self_prefs:
                    self_prefs[example_id] = []
                self_prefs[example_id].append(prob)
    
    if j_vs_r_game2_file.exists():
        with open(j_vs_r_game2_file) as f:
            for line in f:
                item = json.loads(line)
                # Handle example_id=0 correctly (0 is falsy, so can't use 'or')
                example_id = item.get("example_id")
                if example_id is None:
                    example_id = item.get("id")
                # Preference is prob_response2 (J is second in game2)
                prob = item.get("prob_response2", 0.0)
                if example_id not in self_prefs:
                    self_prefs[example_id] = []
                self_prefs[example_id].append(prob)
    
    # Average across games for self preference
    for example_id, prefs in self_prefs.items():
        if example_id not in results:
            results[example_id] = {"self_pref": 0.0, "proxy_prefs": {}}
        results[example_id]["self_pref"] = np.mean(prefs) if prefs else 0.0
    
    # Load K vs R preferences (proxies)
    k_vs_r_game1_file = cache_path / "K_vs_R_game1.jsonl"
    k_vs_r_game2_file = cache_path / "K_vs_R_game2.jsonl"
    
    proxy_prefs = defaultdict(lambda: defaultdict(list))  # example_id -> proxy_name -> [prefs]
    
    for game_file in [k_vs_r_game1_file, k_vs_r_game2_file]:
        if game_file.exists():
            with open(game_file) as f:
                for line in f:
                    item = json.loads(line)
                    # Handle example_id=0 correctly (0 is falsy, so can't use 'or')
                    example_id = item.get("example_id")
                    if example_id is None:
                        example_id = item.get("id")
                    proxy_name = item.get("proxy", "")
                    if not proxy_name:
                        continue
                    
                    # Get preference based on game
                    if "game1" in str(game_file):
                        prob = item.get("prob_response1", 0.0)
                    else:
                        prob = item.get("prob_response2", 0.0)
                    
                    proxy_prefs[example_id][proxy_name].append(prob)
    
    # Average across games for each proxy
    for example_id, proxy_dict in proxy_prefs.items():
        if example_id not in results:
            results[example_id] = {"self_pref": 0.0, "proxy_prefs": {}}
        for proxy_name, prefs in proxy_dict.items():
            results[example_id]["proxy_prefs"][proxy_name] = np.mean(prefs) if prefs else 0.0
    
    return results


def load_preference_scores_author_obfuscation(pref_results_dir: Path, judge: str, reference: str) -> dict:
    """
    Load preference scores from author_obfuscation preference_results format.
    
    Expected format: pref_results_dir/clean_pref_quality_ben.json
    Each item has: evaluator, evaluatee, pid, forward_probability, backward_probability
    
    For J vs R: evaluator=judge, evaluatee=reference
    For K vs R: evaluator=proxy, evaluatee=reference (proxy != judge)
    
    Returns:
        Dict mapping example_id -> {
            'self_pref': float,  # sJ(x, oJ, oR) - average across forward and backward
            'proxy_prefs': dict  # proxy_name -> float (sJ(x, oK, oR))
        }
    """
    results = {}
    
    # Load preference results file
    # Use 2w_harmful for ILSP analysis (2-way position testing, harmful/ILSP examples)
    pref_file = pref_results_dir / "clean_pref_2w_quality_harmful.json"
    if not pref_file.exists():
        # Fallback to non-2w version
        pref_file = pref_results_dir / "clean_pref_quality_harmful.json"
    if not pref_file.exists():
        return results
    
    try:
        with open(pref_file) as f:
            pref_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return results
    
    # Process all preference items
    for item in pref_data:
        evaluator = item.get("evaluator", "")
        evaluatee = item.get("evaluatee", "")
        example_id = item.get("pid", "")
        forward_prob = item.get("forward_probability", 0.0)
        backward_prob = item.get("backward_probability", 0.0)
        
        if not example_id or not evaluatee:
            continue
        
        # Average forward and backward probabilities
        avg_prob = (forward_prob + backward_prob) / 2
        
        # Initialize result entry if needed
        if example_id not in results:
            results[example_id] = {"self_pref": 0.0, "proxy_prefs": {}}
        
        # Check if this is J vs R (self preference)
        if evaluator == judge and evaluatee == reference:
            results[example_id]["self_pref"] = avg_prob
        # Check if this is K vs R (proxy preference) - evaluator is a proxy, not the judge
        elif evaluator != judge and evaluatee == reference:
            proxy_name = evaluator
            results[example_id]["proxy_prefs"][proxy_name] = avg_prob
    
    return results


def compute_mean_differences_per_example(proxy_data: list, cache_dir: Path, ilsp_only: bool = True, source: str = None) -> dict:
    """
    Compute mean difference for each (example_id, judge, reference) triplet.
    
    Mean difference = sJ(x, oJ, oR) - mean_over_proxies(sJ(x, oK, oR))
    
    Args:
        proxy_data: List of dicts from load_proxy_files()
        cache_dir: Directory containing preference score caches (base dir for dbg)
        ilsp_only: If True, only compute for ILSP examples
        source: Data source type (for dbg, cache is at cache_dir/{dataset}/cache)
    
    Returns:
        Dict mapping (example_id, judge, reference) -> {
            'mean_diff': float,
            'proxy_count': int
        }
    """
    triplet_diffs = {}
    
    for entry in proxy_data:
        judge = entry["judge"]
        reference = entry["reference"]
        dataset = entry.get("dataset", "")
        
        # Determine actual cache directory and load function based on source
        if source == "dbg" and dataset:
            # For dbg: cache is at judge_swap_null_dbg_results/{dataset}/cache
            actual_cache_dir = cache_dir / dataset / "cache"
            prefs = load_preference_scores(actual_cache_dir, judge, reference)
        elif source == "verifiable" and dataset:
            # For verifiable: cache is at judge_swap_null_verif_smoke2/{dataset}/cache
            actual_cache_dir = cache_dir / dataset / "cache"
            prefs = load_preference_scores(actual_cache_dir, judge, reference)
        elif source == "panickserry" and dataset:
            # For panickserry: cache is at panickserry_results/{dataset}_result/{dataset}/cache
            actual_cache_dir = cache_dir / f"{dataset}_result" / dataset / "cache"
            if not actual_cache_dir.exists():
                actual_cache_dir = cache_dir / f"{dataset}_results" / dataset / "cache"
            prefs = load_preference_scores(actual_cache_dir, judge, reference)
        elif source == "author_obfuscation":
            # For author_obfuscation: preference results in cache_dir/clean_pref_quality_ben.json
            prefs = load_preference_scores_author_obfuscation(cache_dir, judge, reference)
        else:
            actual_cache_dir = cache_dir
            prefs = load_preference_scores(actual_cache_dir, judge, reference)
        
        # Get relevant example IDs based on ilsp_only flag
        relevant_ids = set()
        proxy_counts = defaultdict(int)  # (example_id, judge, reference) -> count
        
        for proxy_name, proxy_info in entry["proxies"].items():
            if ilsp_only:
                example_ids = set(proxy_info.get("ilsp_ids", []))
            else:
                lsp_ids = proxy_info.get("lsp_ids", [])
                ilsp_ids = proxy_info.get("ilsp_ids", [])
                example_ids = set(lsp_ids) | set(ilsp_ids)
            
            for example_id in example_ids:
                relevant_ids.add(example_id)
                proxy_counts[(example_id, judge, reference)] += 1
        
        # Compute mean differences for each example
        for example_id in relevant_ids:
            key = (example_id, judge, reference)
            
            if example_id not in prefs:
                continue
            
            self_pref = prefs[example_id]["self_pref"]
            proxy_prefs_dict = prefs[example_id]["proxy_prefs"]
            
            # Get proxies that match this example
            matching_proxies = []
            for proxy_name, proxy_info in entry["proxies"].items():
                if ilsp_only:
                    example_ids = set(proxy_info.get("ilsp_ids", []))
                else:
                    lsp_ids = proxy_info.get("lsp_ids", [])
                    ilsp_ids = proxy_info.get("ilsp_ids", [])
                    example_ids = set(lsp_ids) | set(ilsp_ids)
                
                if example_id in example_ids and proxy_name in proxy_prefs_dict:
                    matching_proxies.append(proxy_prefs_dict[proxy_name])
            
            if not matching_proxies:
                continue
            
            # Mean difference = self_pref - mean(proxy_prefs)
            mean_proxy_pref = np.mean(matching_proxies)
            mean_diff = self_pref - mean_proxy_pref
            
            triplet_diffs[key] = {
                "mean_diff": mean_diff,
                "proxy_count": proxy_counts[key]
            }
    
    return triplet_diffs


def compute_average_mean_diff_at_thresholds(triplet_diffs: dict, max_threshold: int = None) -> tuple:
    """
    Compute the average mean difference and standard error for examples with at least N proxies.
    
    Args:
        triplet_diffs: Dict mapping triplet -> {'mean_diff': float, 'proxy_count': int}
        max_threshold: Maximum threshold to consider (default: max count in data)
    
    Returns:
        Tuple of (thresholds, average_mean_diffs, standard_errors)
    """
    if not triplet_diffs:
        return [], [], []
    
    counts = [v["proxy_count"] for v in triplet_diffs.values()]
    
    if max_threshold is None:
        max_threshold = max(counts) if counts else 1
    
    thresholds = list(range(1, max_threshold + 1))
    avg_mean_diffs = []
    standard_errors = []
    
    for n in thresholds:
        # Filter to examples with at least N proxies
        matching_diffs = [
            v["mean_diff"] for v in triplet_diffs.values()
            if v["proxy_count"] >= n
        ]
        
        if matching_diffs:
            avg_mean_diffs.append(np.mean(matching_diffs))
            # Standard error = std / sqrt(n)
            std = np.std(matching_diffs, ddof=1) if len(matching_diffs) > 1 else 0.0
            se = std / np.sqrt(len(matching_diffs))
            standard_errors.append(se)
        else:
            avg_mean_diffs.append(0.0)
            standard_errors.append(0.0)
    
    return thresholds, avg_mean_diffs, standard_errors


def create_plot(thresholds: list, avg_mean_diffs: list, standard_errors: list, output_path: Path, title: str = None, main_title: str = None):
    """
    Create the proxy robustness plot for mean differences.
    
    Args:
        thresholds: List of N values (at least N proxies)
        avg_mean_diffs: Corresponding average mean differences
        standard_errors: Corresponding standard errors
        output_path: Where to save the plot
        title: Optional plot title
        main_title: Optional main title for the entire figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy arrays for arithmetic
    thresholds_arr = np.array(thresholds)
    avg_arr = np.array(avg_mean_diffs)
    se_arr = np.array(standard_errors)
    
    # Plot shaded confidence band (±1 SE)
    ax.fill_between(thresholds_arr, avg_arr - se_arr, avg_arr + se_arr, 
                    alpha=0.3, color='steelblue', label='±1 SE')
    
    # Plot mean line with markers
    ax.plot(thresholds, avg_mean_diffs, marker='o', linewidth=2, markersize=8, color='steelblue')
    
    ax.set_xlabel("Minimum Number of Proxies (N)", fontsize=12)
    ax.set_ylabel("Average Mean Difference", fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title("Average Mean Difference\nfor Examples With At Least N Valid Proxies", fontsize=14, fontweight='bold')
    
    ax.set_xlim(left=1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Set x-axis to integer ticks only
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # Add main title if provided
    if main_title:
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved plot to {output_path} and {output_path.with_suffix('.pdf')}")
    plt.close()


def create_subplots(dataset_results: dict, output_path: Path, main_title: str = None):
    """
    Create side-by-side subplots for multiple datasets.
    
    Args:
        dataset_results: Dict mapping dataset name -> (thresholds, avg_mean_diffs, standard_errors)
        output_path: Where to save the plot
        main_title: Optional main title for the entire figure
    """
    num_datasets = len(dataset_results)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 6))
    
    # Handle single subplot case
    if num_datasets == 1:
        axes = [axes]
    
    for idx, (dataset_name, (thresholds, avg_mean_diffs, standard_errors)) in enumerate(dataset_results.items()):
        ax = axes[idx]
        
        # Convert to numpy arrays for arithmetic
        thresholds_arr = np.array(thresholds)
        avg_arr = np.array(avg_mean_diffs)
        se_arr = np.array(standard_errors)
        
        # Plot shaded confidence band (±1 SE)
        ax.fill_between(thresholds_arr, avg_arr - se_arr, avg_arr + se_arr, 
                        alpha=0.3, color='steelblue')
        
        # Plot mean line with markers
        ax.plot(thresholds, avg_mean_diffs, marker='o', linewidth=2, markersize=8, color='steelblue')
        
        ax.set_xlabel("Minimum Number of Proxies (N)", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Average Mean Difference", fontsize=12)
        
        ax.set_title(dataset_name.upper(), fontsize=14, fontweight='bold')
        ax.set_xlim(left=1)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Set x-axis to integer ticks only
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # Add main title if provided
    if main_title:
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved plot to {output_path} and {output_path.with_suffix('.pdf')}")
    plt.close()


def print_summary(triplet_diffs: dict, thresholds: list, avg_mean_diffs: list, filter_label: str = "", source_label: str = ""):
    """Print summary statistics."""
    mean_diffs = [v["mean_diff"] for v in triplet_diffs.values()]
    proxy_counts = [v["proxy_count"] for v in triplet_diffs.values()]
    
    print("\n" + "=" * 60)
    print(f"MEAN DIFFERENCE SUMMARY")
    if source_label:
        print(f"Source: {source_label}")
    if filter_label:
        print(f"Filter: {filter_label}")
    print("=" * 60)
    print(f"Total (example, judge, reference) triplets: {len(triplet_diffs):,}")
    if mean_diffs:
        print(f"Min mean difference: {min(mean_diffs):.4f}")
        print(f"Max mean difference: {max(mean_diffs):.4f}")
        print(f"Mean mean difference: {np.mean(mean_diffs):.4f}")
        print(f"Median mean difference: {np.median(mean_diffs):.4f}")
    if proxy_counts:
        print(f"Min proxies per triplet: {min(proxy_counts)}")
        print(f"Max proxies per triplet: {max(proxy_counts)}")
        print(f"Mean proxies per triplet: {np.mean(proxy_counts):.2f}")
    print()
    print("Average mean difference for examples with at least N proxies:")
    print("-" * 40)
    for n, avg_diff in zip(thresholds, avg_mean_diffs):
        print(f"  N >= {n:2d}: {avg_diff:8.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Plot average mean difference across (example, judge, reference) triplets"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["author_obfuscation", "dbg", "verifiable", "panickserry"],
        default="author_obfuscation",
        help="Data source: author_obfuscation, dbg, verifiable, or panickserry"
    )
    parser.add_argument(
        "--proxy_dir",
        type=str,
        default=None,
        help="Directory containing proxy JSON files (default: auto based on source)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory containing preference score caches (default: auto based on source)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name or 'all'. For dbg: alpaca_eval, translation, truthfulness. For verifiable: math500, mbpp-plus, mmlu. For panickserry: cnn, xsum"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: auto based on source/dataset)"
    )
    parser.add_argument(
        "--max_threshold",
        type=int,
        default=None,
        help="Maximum threshold N to show (default: auto from data)"
    )
    parser.add_argument(
        "--include_lsp",
        action="store_true",
        help="Include both LSP and ILSP examples (default: ILSP only)"
    )
    args = parser.parse_args()
    
    # Set default proxy directory based on source
    if args.proxy_dir is None:
        args.proxy_dir = DEFAULT_PATHS[args.source]
    
    proxy_dir = Path(args.proxy_dir)
    
    # Set default cache directory based on source
    if args.cache_dir is None:
        args.cache_dir = DEFAULT_CACHE_PATHS.get(args.source, "cache")
    
    cache_dir = Path(args.cache_dir)
    
    # Set default output path
    if args.output is None:
        if args.source == "dbg":
            args.output = f"proxy_robustness_plot2_dbg_{args.dataset}.png"
        elif args.source == "verifiable":
            args.output = f"proxy_robustness_plot2_verifiable_{args.dataset}.png"
        elif args.source == "panickserry":
            args.output = f"proxy_robustness_plot2_panickserry_{args.dataset}.png"
        else:
            args.output = "proxy_robustness_plot2_author_obfuscation.png"
    
    output_path = Path(args.output)
    
    if not proxy_dir.exists():
        print(f"ERROR: Proxy directory not found: {proxy_dir}")
        return
    
    if not cache_dir.exists():
        print(f"WARNING: Cache directory not found: {cache_dir}")
        print("Mean differences will be computed only for examples with cached preferences.")
    
    # Determine datasets for dbg/verifiable/panickserry source
    datasets = None
    use_all_datasets = False
    if args.source == "dbg":
        if args.dataset == "all":
            datasets = DBG_DATASETS
            use_all_datasets = True
        else:
            datasets = [args.dataset]
    elif args.source == "verifiable":
        if args.dataset == "all":
            datasets = VERIFIABLE_DATASETS
            use_all_datasets = True
        else:
            datasets = [args.dataset]
    elif args.source == "panickserry":
        if args.dataset == "all":
            datasets = PANICKSERRY_DATASETS
            use_all_datasets = True
        else:
            datasets = [args.dataset]
    
    print(f"Loading proxy files from: {proxy_dir}")
    if args.source in ["dbg", "verifiable", "panickserry"]:
        print(f"Datasets: {datasets}")
    print(f"Loading preference scores from: {cache_dir}")
    
    proxy_data = load_proxy_files(args.source, proxy_dir, datasets)
    print(f"Loaded {len(proxy_data)} (judge, reference) pair files")
    
    # Count unique datasets
    unique_datasets = set(d["dataset"] for d in proxy_data)
    print(f"Datasets found: {unique_datasets}")
    
    ilsp_only = not args.include_lsp
    filter_label = "ILSP only" if ilsp_only else "LSP + ILSP"
    
    # Map source names to display names
    source_display_names = {
        "dbg": "dbg_score_paper",
        "author_obfuscation": "author_obfuscation",
        "verifiable": "llm-sp-verif",
        "panickserry": "panickserry"
    }
    domain_name = source_display_names.get(args.source, args.source)
    main_title = f"{domain_name}: Average Mean Difference for Examples With At Least N Valid Proxies"
    
    # If using all datasets, create subplots for each dataset
    if use_all_datasets and len(unique_datasets) > 1:
        print(f"Computing mean differences per (example, judge, reference) triplet ({filter_label})...")
        dataset_results = {}
        
        for dataset in sorted(unique_datasets):
            # Filter proxy_data for this dataset
            dataset_data = [d for d in proxy_data if d["dataset"] == dataset]
            triplet_diffs = compute_mean_differences_per_example(dataset_data, cache_dir, ilsp_only=ilsp_only, source=args.source)
            
            if not triplet_diffs:
                print(f"Warning: No triplets found for dataset {dataset}")
                continue
            
            print(f"Computing average mean differences for {dataset}...")
            thresholds, avg_mean_diffs, standard_errors = compute_average_mean_diff_at_thresholds(
                triplet_diffs, max_threshold=args.max_threshold
            )
            dataset_results[dataset] = (thresholds, avg_mean_diffs, standard_errors)
            
            print_summary(triplet_diffs, thresholds, avg_mean_diffs, filter_label, dataset)
        
        if not dataset_results:
            print("ERROR: No triplets found for any dataset!")
            return
        
        print(f"\nCreating subplots...")
        create_subplots(dataset_results, output_path, main_title=main_title)
    else:
        # Single dataset or single plot
        print(f"Computing mean differences per (example, judge, reference) triplet ({filter_label})...")
        triplet_diffs = compute_mean_differences_per_example(proxy_data, cache_dir, ilsp_only=ilsp_only, source=args.source)
        
        if not triplet_diffs:
            print("ERROR: No triplets found!")
            return
        
        print("Computing average mean differences at each threshold...")
        thresholds, avg_mean_diffs, standard_errors = compute_average_mean_diff_at_thresholds(
            triplet_diffs, max_threshold=args.max_threshold
        )
        
        # Use dataset name as title if available
        if unique_datasets:
            dataset_name = list(unique_datasets)[0]
            title = dataset_name.upper()
        else:
            source_label = f"{args.source}"
            if args.source in ["dbg", "verifiable", "panickserry"]:
                source_label += f" ({args.dataset})"
            title = source_label.upper()
        
        print_summary(triplet_diffs, thresholds, avg_mean_diffs, filter_label, title)
        
        print(f"\nCreating plot...")
        create_plot(thresholds, avg_mean_diffs, standard_errors, output_path, title=title, main_title=main_title)


if __name__ == "__main__":
    main()
