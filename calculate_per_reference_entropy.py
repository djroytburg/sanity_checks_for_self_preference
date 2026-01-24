# calculate_per_reference_entropy.py: Calculate entropy for each (judge, reference) pair
# Created: January 18, 2026, 03:30 AM EST
# Last Modified: January 18, 2026, 03:30 AM EST

import json
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
from glob import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_logs/calculate_per_reference_entropy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_entropy(probs):
    """
    Calculate binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).
    
    Args:
        probs (list): List of probabilities
        
    Returns:
        float: Average binary entropy
    """
    probs = np.array(probs)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    entropy_values = -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)
    return np.mean(entropy_values)


def load_jvsr_cache_verif(dataset_dir, dataset, judge, reference):
    """Load J_vs_R cache for verif experiments."""
    lsp_examples = []
    ilsp_examples = []
    
    # Try game1
    game1_path = Path(dataset_dir) / dataset / 'cache' / judge / reference / 'J_vs_R_game1.jsonl'
    game2_path = Path(dataset_dir) / dataset / 'cache' / judge / reference / 'J_vs_R_game2.jsonl'
    
    if not game1_path.exists() or not game2_path.exists():
        return None, None
    
    # Load game1
    game1_data = {}
    with open(game1_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            game1_data[entry['example_id']] = entry
    
    # Load game2  
    with open(game2_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            example_id = entry['example_id']
            
            if example_id not in game1_data:
                continue
            
            # Calculate p_self
            prob1 = game1_data[example_id].get('prob_response1', 0.5)
            prob2 = entry.get('prob_response2', 0.5)
            p_self = (prob1 + prob2) / 2
            
            # Classify as LSP or ILSP
            if game1_data[example_id].get('category') == 'lsp':
                lsp_examples.append(p_self)
            else:
                ilsp_examples.append(p_self)
    
    return lsp_examples, ilsp_examples


def load_jvsr_cache_dbg(dataset_dir, dataset, judge, reference):
    """Load J_vs_R cache for DBG experiments."""
    lsp_examples = []
    ilsp_examples = []
    
    # Try game1
    game1_path = Path(dataset_dir) / dataset / 'cache' / judge / reference / 'J_vs_R_game1.jsonl'
    game2_path = Path(dataset_dir) / dataset / 'cache' / judge / reference / 'J_vs_R_game2.jsonl'
    
    if not game1_path.exists() or not game2_path.exists():
        return None, None
    
    # Load game1
    game1_data = {}
    with open(game1_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            game1_data[entry['example_id']] = entry
    
    # Load game2
    with open(game2_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            example_id = entry['example_id']
            
            if example_id not in game1_data:
                continue
            
            # Calculate p_self
            prob1 = game1_data[example_id].get('prob_response1', 0.5)
            prob2 = entry.get('prob_response2', 0.5)
            p_self = (prob1 + prob2) / 2
            
            # Classify as LSP or ILSP
            if game1_data[example_id].get('category') == 'lsp':
                lsp_examples.append(p_self)
            else:
                ilsp_examples.append(p_self)
    
    return lsp_examples, ilsp_examples


def load_jvsr_cache_author_obf(dataset_dir, dataset, judge, reference):
    """Load J_vs_R cache for author obfuscation experiments."""
    lsp_examples = []
    ilsp_examples = []

    # Files are in quality/{judge}/judge_swap_{judge}_{ref}_{proxy}.json
    # We need to find all proxy files for this (judge, ref) pair
    pattern = Path(dataset_dir) / dataset / judge / f'judge_swap_{judge}_{reference}_*.json'
    proxy_files = glob(str(pattern))

    if not proxy_files:
        return None, None

    # Aggregate across all proxies
    all_game1 = {}
    all_game2 = {}

    for proxy_file in proxy_files:
        with open(proxy_file, 'r') as f:
            data = json.load(f)

        # Game 1
        for entry in data.get('game_1', []):
            to_eval_id = entry['to_eval_id']
            all_game1[to_eval_id] = entry

        # Game 2
        for entry in data.get('game_2', []):
            to_eval_id = entry['to_eval_id']
            all_game2[to_eval_id] = entry

    # Combine games
    for to_eval_id, entry1 in all_game1.items():
        if to_eval_id not in all_game2:
            continue

        entry2 = all_game2[to_eval_id]

        # Calculate p_self (using prob_A and prob_B)
        prob1 = entry1.get('prob_A', 0.5)
        prob2 = entry2.get('prob_B', 0.5)
        p_self = (prob1 + prob2) / 2

        # Classify as LSP or ILSP
        if entry1.get('category') == 'lsp':
            lsp_examples.append(p_self)
        else:
            ilsp_examples.append(p_self)

    return lsp_examples, ilsp_examples


def load_jvsr_cache_panickserry(cache_dir, winrate_dir, dataset, judge, reference):
    """Load J_vs_R cache for panickserry (CNN/XSUM) experiments."""
    lsp_examples = []
    ilsp_examples = []

    # Normalize judge name (gpt-3.5-turbo -> GPT-3.5, gpt-4 -> GPT-4)
    judge_norm = judge.replace("gpt-3.5-turbo", "GPT-3.5").replace("gpt-4", "GPT-4")
    ref_norm = reference

    # Load LSP/ILSP IDs from winrate file
    winrate_file = Path(winrate_dir) / f"{judge_norm}_vs_{ref_norm}.json"

    if not winrate_file.exists():
        return None, None

    with open(winrate_file, 'r') as f:
        winrate_data = json.load(f)

    # Collect LSP/ILSP IDs from all proxies
    all_lsp_ids = set()
    all_ilsp_ids = set()

    for proxy_name, proxy_data in winrate_data.get('proxies', {}).items():
        if proxy_name == 'oracle_gpt5':
            continue

        lsp_ids = proxy_data.get('lsp_ids', [])
        ilsp_ids = proxy_data.get('ilsp_ids', [])

        all_lsp_ids.update(lsp_ids)
        all_ilsp_ids.update(ilsp_ids)

    if not all_lsp_ids and not all_ilsp_ids:
        return None, None

    # Normalize reference name for cache directory lookup
    # CNN uses: gpt3.5, human, llama2
    # XSUM uses: gpt-3.5 (for gpt-4), llama2 (for gpt-3.5-turbo), llama (for gpt-4)
    # Try multiple variations to handle inconsistencies
    ref_variations = [reference]

    if 'GPT-3.5' in reference or 'gpt-3.5' in reference:
        if dataset == 'cnn':
            ref_variations = ['gpt3.5']
        else:
            ref_variations = ['gpt-3.5', 'gpt3.5']
    elif 'llama' in reference.lower():
        # Try both llama and llama2
        ref_variations = ['llama2', 'llama']

    # Try to find cache files with different reference name variations
    game1_path = None
    game2_path = None

    for ref_cache in ref_variations:
        g1 = Path(cache_dir) / dataset / 'cache' / judge / ref_cache / 'J_vs_R_game1.jsonl'
        g2 = Path(cache_dir) / dataset / 'cache' / judge / ref_cache / 'J_vs_R_game2.jsonl'

        if g1.exists() and g2.exists():
            game1_path = g1
            game2_path = g2
            break

    if not game1_path or not game2_path:
        return None, None

    # Load game1
    game1_data = {}
    with open(game1_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            game1_data[entry['example_id']] = entry

    # Load game2 and calculate p_self
    with open(game2_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            example_id = entry['example_id']

            if example_id not in game1_data:
                continue

            # Calculate p_self
            prob1 = game1_data[example_id].get('prob_response1', 0.5)
            prob2 = entry.get('prob_response2', 0.5)
            p_self = (prob1 + prob2) / 2

            # Classify as LSP or ILSP based on proxy categorizations
            if example_id in all_lsp_ids:
                lsp_examples.append(p_self)
            elif example_id in all_ilsp_ids:
                ilsp_examples.append(p_self)

    return lsp_examples, ilsp_examples


def calculate_per_reference_entropy_all():
    """
    Calculate entropy for all (dataset, judge, reference) combinations.

    Returns:
        dict: {(dataset, judge, reference): {entropy_lsp, entropy_ilsp, entropy_gap}}
    """
    experiments = [
        ('judge_swap_null_verif_smoke2', ['math500', 'mmlu', 'mbpp-plus', 'alpaca_eval'], load_jvsr_cache_verif),
        ('judge_swap_null_dbg_results', ['translation', 'truthfulness', 'alpaca_eval'], load_jvsr_cache_dbg),
        ('judge_swap_null_author_obfuscation', ['quality'], load_jvsr_cache_author_obf),
    ]

    results = {}

    for exp_dir, datasets, load_func in experiments:
        for dataset in datasets:
            logger.info(f"Processing {exp_dir}/{dataset}...")

            # Get list of (judge, reference) pairs from aggregated file
            agg_file = Path(exp_dir) / dataset / 'analysis' / 'aggregated_by_judge_reference.json'

            if not agg_file.exists():
                logger.warning(f"Aggregated file not found: {agg_file}")
                continue

            with open(agg_file, 'r') as f:
                agg_data = json.load(f)

            for entry in agg_data:
                judge = entry['judge']
                reference = entry['reference']

                # Load J_vs_R cache
                lsp_probs, ilsp_probs = load_func(exp_dir, dataset, judge, reference)

                if lsp_probs is None or ilsp_probs is None:
                    logger.warning(f"No cache data for ({dataset}, {judge}, {reference})")
                    continue

                if len(lsp_probs) == 0 or len(ilsp_probs) == 0:
                    logger.warning(f"Empty distributions for ({dataset}, {judge}, {reference})")
                    continue

                # Calculate entropies
                entropy_lsp = calculate_entropy(lsp_probs)
                entropy_ilsp = calculate_entropy(ilsp_probs)
                entropy_gap = entropy_ilsp - entropy_lsp

                key = (dataset, judge, reference)
                results[key] = {
                    'entropy_lsp': entropy_lsp,
                    'entropy_ilsp': entropy_ilsp,
                    'entropy_gap': entropy_gap,
                    'n_lsp': len(lsp_probs),
                    'n_ilsp': len(ilsp_probs)
                }

                logger.debug(f"  ({judge}, {reference}): H_LSP={entropy_lsp:.3f}, H_ILSP={entropy_ilsp:.3f}, gap={entropy_gap:.3f}")

    # Process panickserry datasets (CNN and XSUM)
    panickserry_experiments = [
        ('panickserry_results/cnn_results', 'panickserry_results/cnn_winrates', 'cnn'),
        ('panickserry_results/xsum_result', 'panickserry_results/xsum_winrates', 'xsum'),
    ]

    for cache_base, winrate_dir, dataset in panickserry_experiments:
        logger.info(f"Processing panickserry/{dataset}...")

        winrate_path = Path(winrate_dir)
        if not winrate_path.exists():
            logger.warning(f"Winrate directory not found: {winrate_path}")
            continue

        # Get list of (judge, reference) pairs from winrate files
        for winrate_file in winrate_path.glob("*.json"):
            # Parse filename: GPT-4_vs_human.json -> judge=GPT-4, reference=human
            filename = winrate_file.stem
            parts = filename.split('_vs_')

            if len(parts) != 2:
                continue

            judge_norm = parts[0]
            reference = parts[1]

            # Denormalize judge name (GPT-3.5 -> gpt-3.5-turbo, GPT-4 -> gpt-4)
            judge = judge_norm.replace("GPT-3.5", "gpt-3.5-turbo").replace("GPT-4", "gpt-4")

            # Load J_vs_R cache
            lsp_probs, ilsp_probs = load_jvsr_cache_panickserry(cache_base, winrate_dir, dataset, judge, reference)

            if lsp_probs is None or ilsp_probs is None:
                logger.warning(f"No cache data for ({dataset}, {judge}, {reference})")
                continue

            if len(lsp_probs) == 0 or len(ilsp_probs) == 0:
                logger.warning(f"Empty distributions for ({dataset}, {judge}, {reference})")
                continue

            # Calculate entropies
            entropy_lsp = calculate_entropy(lsp_probs)
            entropy_ilsp = calculate_entropy(ilsp_probs)
            entropy_gap = entropy_ilsp - entropy_lsp

            key = (dataset, judge, reference)
            results[key] = {
                'entropy_lsp': entropy_lsp,
                'entropy_ilsp': entropy_ilsp,
                'entropy_gap': entropy_gap,
                'n_lsp': len(lsp_probs),
                'n_ilsp': len(ilsp_probs)
            }

            logger.info(f"  ({judge}, {reference}): H_LSP={entropy_lsp:.3f}, H_ILSP={entropy_ilsp:.3f}, gap={entropy_gap:.3f}")

    logger.info(f"Calculated per-reference entropy for {len(results)} combinations")
    return results


def main():
    """Main function."""
    logger.info("Starting per-reference entropy calculation")
    
    # Calculate entropies
    results = calculate_per_reference_entropy_all()
    
    # Save to JSON
    output_file = Path('per_reference_entropy.json')
    
    # Convert tuple keys to strings for JSON
    json_results = {}
    for (dataset, judge, reference), metrics in results.items():
        key = f"{dataset}||{judge}||{reference}"
        json_results[key] = metrics
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Saved per-reference entropy to {output_file}")
    logger.info(f"Total: {len(results)} combinations")


if __name__ == '__main__':
    main()
