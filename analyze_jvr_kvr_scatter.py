# analyze_jvr_kvr_scatter.py: J vs R / K vs R scatter analysis with LOBF
# Generates scatter plots comparing P(J|JvR) vs P(K|KvR) at multiple aggregation levels
# Created: January 23, 2026, 10:00 AM EST
# Last Modified: January 23, 2026, 10:00 AM EST

import argparse
import json
import logging
import os
import sys
import getpass
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress
from glob import glob


# -------------------------
# --- CONSTANTS ---
# -------------------------

VERIF_DATASETS = ["math500", "mmlu", "mbpp-plus"]
DBG_DATASETS = ["alpaca_eval", "translation", "truthfulness"]
AUTHOR_OBF_DATASETS = ["quality"]
PANICKSERRY_DATASETS = ["cnn", "xsum"]

PAPER_GROUPS = {
    "llm-sp-verif": VERIF_DATASETS,
    "dbg-score-paper": DBG_DATASETS,
    "author_obfuscation": AUTHOR_OBF_DATASETS,
    "panickserry": PANICKSERRY_DATASETS,
}

FAMILY_COLORS = {
    'llama': '#4A90E2',
    'gemma': '#2ECC71',
    'qwen': '#E74C3C',
    'gpt': '#F39C12',
    'mistral': '#9B59B6',
    'phi': '#1ABC9C',
    'deepseek': '#E67E22',
    'other': '#95A5A6'
}


# -------------------------
# --- LOGGING SETUP ---
# -------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging with file and console handlers."""
    log_dir = output_dir / "file_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analyze_jvr_kvr_scatter_{timestamp}.log"

    logger = logging.getLogger("analyze_jvr_kvr_scatter")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("J vs R / K vs R SCATTER ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"User: {getpass.getuser()}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    return logger


# -------------------------
# --- UTILITIES ---
# -------------------------

def extract_model_family(model_name: str) -> str:
    """Extract model family from model name."""
    model_lower = model_name.lower()
    # Check GPT variants first (more specific before general)
    if 'gpt-4' in model_lower or 'gpt4' in model_lower:
        return 'gpt'
    if 'gpt-3.5' in model_lower or 'gpt3.5' in model_lower:
        return 'gpt'

    families = ['llama', 'gemma', 'qwen', 'mistral', 'phi', 'deepseek', 'gpt', 'hermes']
    for family in families:
        if family in model_lower:
            return family
    return 'other'


def extract_model_size(model_name: str) -> int:
    """Extract model size in billions of parameters."""
    import re
    pattern = r'(\d+)[bB](?:[^a-zA-Z]|$)'
    matches = re.findall(pattern, model_name)
    return int(matches[-1]) if matches else 16


def size_to_marker_size(size: int) -> int:
    """Convert model size to marker size."""
    if size <= 3:
        return 80
    elif size <= 9:
        return 200
    elif size <= 32:
        return 400
    else:
        return 700


# -------------------------
# --- DATA LOADING ---
# -------------------------

def load_verif_dbg_cache(
    cache_dir: Path,
    logger: logging.Logger
) -> Dict[Tuple[str, str, str, str], Dict]:
    """
    Load J vs R and K vs R data from verif/dbg cache structure.

    Returns:
        Dict mapping (dataset, judge, reference, example_id) -> {
            'j_prob': float,  # averaged across games
            'k_probs': List[float],  # one per proxy, each averaged across games
            'category': str  # 'lsp' or 'ilsp'
        }
    """
    results = {}

    dataset = cache_dir.parent.name

    for judge_dir in cache_dir.iterdir():
        if not judge_dir.is_dir():
            continue
        judge = judge_dir.name

        for ref_dir in judge_dir.iterdir():
            if not ref_dir.is_dir():
                continue
            reference = ref_dir.name

            # Load J vs R games
            j_game1_file = ref_dir / "J_vs_R_game1.jsonl"
            j_game2_file = ref_dir / "J_vs_R_game2.jsonl"

            if not j_game1_file.exists() or not j_game2_file.exists():
                logger.debug(f"Missing J_vs_R files for {judge}/{reference}")
                continue

            # Load J game1
            j_game1_by_id = {}
            with open(j_game1_file) as f:
                for line in f:
                    entry = json.loads(line)
                    j_game1_by_id[entry['example_id']] = entry

            # Load J game2
            j_game2_by_id = {}
            with open(j_game2_file) as f:
                for line in f:
                    entry = json.loads(line)
                    j_game2_by_id[entry['example_id']] = entry

            # Load K vs R games (multiple proxies per file)
            k_game1_file = ref_dir / "K_vs_R_game1.jsonl"
            k_game2_file = ref_dir / "K_vs_R_game2.jsonl"

            if not k_game1_file.exists() or not k_game2_file.exists():
                logger.debug(f"Missing K_vs_R files for {judge}/{reference}")
                continue

            # Group K by (example_id, proxy)
            k_game1_by_id_proxy = defaultdict(dict)
            with open(k_game1_file) as f:
                for line in f:
                    entry = json.loads(line)
                    ex_id = entry['example_id']
                    proxy = entry.get('proxy', 'unknown')
                    k_game1_by_id_proxy[ex_id][proxy] = entry

            k_game2_by_id_proxy = defaultdict(dict)
            with open(k_game2_file) as f:
                for line in f:
                    entry = json.loads(line)
                    ex_id = entry['example_id']
                    proxy = entry.get('proxy', 'unknown')
                    k_game2_by_id_proxy[ex_id][proxy] = entry

            # Compute averaged probabilities for each example
            common_j_ids = set(j_game1_by_id.keys()) & set(j_game2_by_id.keys())

            for ex_id in common_j_ids:
                j_g1 = j_game1_by_id[ex_id]
                j_g2 = j_game2_by_id[ex_id]

                # J probability: avg of game1 prob_response1 and game2 prob_response2
                j_prob = (j_g1['prob_response1'] + j_g2['prob_response2']) / 2.0
                category = j_g1.get('category', 'unknown')

                # K probabilities: avg across games for each proxy
                k_probs = []
                proxies_g1 = k_game1_by_id_proxy.get(ex_id, {})
                proxies_g2 = k_game2_by_id_proxy.get(ex_id, {})

                common_proxies = set(proxies_g1.keys()) & set(proxies_g2.keys())
                for proxy in common_proxies:
                    k_g1 = proxies_g1[proxy]
                    k_g2 = proxies_g2[proxy]
                    k_prob = (k_g1['prob_response1'] + k_g2['prob_response2']) / 2.0
                    k_probs.append(k_prob)

                if k_probs:
                    key = (dataset, judge, reference, ex_id)
                    results[key] = {
                        'j_prob': j_prob,
                        'k_probs': k_probs,
                        'category': category
                    }

    logger.info(f"Loaded {len(results)} examples from {cache_dir}")
    return results


def load_author_obf_cache(
    quality_dir: Path,
    logger: logging.Logger
) -> Dict[Tuple[str, str, str, str], Dict]:
    """
    Load J vs R and K vs R data from author_obfuscation format.

    Returns:
        Dict mapping (dataset, judge, reference, example_id) -> {
            'j_prob': float,
            'k_probs': List[float],
            'category': str
        }
    """
    results = {}
    dataset = "quality"

    # Load LSP/ILSP labels from proxy directory
    proxy_dir = Path("author_obfuscation/data/quality/proxies")
    label_cache = {}

    def get_labels(judge: str, reference: str) -> Tuple[set, set]:
        """Get (lsp_ids, ilsp_ids) for a judge/reference pair."""
        cache_key = (judge, reference)
        if cache_key in label_cache:
            return label_cache[cache_key]

        proxy_file = proxy_dir / f"evaluator_{judge}_vs_{reference}.json"
        if not proxy_file.exists():
            label_cache[cache_key] = (set(), set())
            return label_cache[cache_key]

        try:
            with open(proxy_file) as f:
                data = json.load(f)

            # Aggregate LSP/ILSP IDs across all proxies
            lsp_ids = set()
            ilsp_ids = set()
            for proxy_name, proxy_info in data.get("proxies", {}).items():
                lsp_ids.update(proxy_info.get("lsp_ids", []))
                ilsp_ids.update(proxy_info.get("ilsp_ids", []))

            label_cache[cache_key] = (lsp_ids, ilsp_ids)
        except Exception as e:
            logger.warning(f"Failed to load labels from {proxy_file}: {e}")
            label_cache[cache_key] = (set(), set())

        return label_cache[cache_key]

    # Walk quality_dir / judge / *.json
    for judge_dir in quality_dir.iterdir():
        if not judge_dir.is_dir() or judge_dir.name == "analysis":
            continue

        judge = judge_dir.name

        for json_file in judge_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                if not isinstance(data, dict) or 'metadata' not in data:
                    continue

                metadata = data['metadata']
                file_judge = metadata['judge']
                proxy = metadata['proxy']
                reference = metadata['reference']

                assert file_judge == judge, f"Judge mismatch: {file_judge} vs {judge}"

                lsp_ids, ilsp_ids = get_labels(judge, reference)

                # Group by example_id
                j_game1_by_id = {}
                j_game2_by_id = {}
                k_game1_by_id = defaultdict(list)
                k_game2_by_id = defaultdict(list)

                for item in data.get('J_vs_R_game1', []):
                    j_game1_by_id[item['example_id']] = item

                for item in data.get('J_vs_R_game2', []):
                    j_game2_by_id[item['example_id']] = item

                for item in data.get('K_vs_R_game1', []):
                    k_game1_by_id[item['example_id']].append(item)

                for item in data.get('K_vs_R_game2', []):
                    k_game2_by_id[item['example_id']].append(item)

                # Process each example
                common_j_ids = set(j_game1_by_id.keys()) & set(j_game2_by_id.keys())

                for ex_id in common_j_ids:
                    j_g1 = j_game1_by_id[ex_id]
                    j_g2 = j_game2_by_id[ex_id]

                    # J probability: avg of game1 prob_A and game2 prob_B
                    j_prob = (j_g1['prob_A'] + j_g2['prob_B']) / 2.0

                    # Determine category
                    if ex_id in lsp_ids:
                        category = 'lsp'
                    elif ex_id in ilsp_ids:
                        category = 'ilsp'
                    else:
                        category = j_g1.get('category', 'unknown')

                    # K probabilities
                    k_g1_list = k_game1_by_id.get(ex_id, [])
                    k_g2_list = k_game2_by_id.get(ex_id, [])

                    k_probs = []
                    # Match by position (assuming same order)
                    for k_g1, k_g2 in zip(k_g1_list, k_g2_list):
                        k_prob = (k_g1['prob_A'] + k_g2['prob_B']) / 2.0
                        k_probs.append(k_prob)

                    if k_probs:
                        key = (dataset, judge, reference, ex_id)
                        if key in results:
                            # Aggregate K probs from multiple proxy files
                            results[key]['k_probs'].extend(k_probs)
                        else:
                            results[key] = {
                                'j_prob': j_prob,
                                'k_probs': k_probs,
                                'category': category
                            }

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

    logger.info(f"Loaded {len(results)} examples from author_obfuscation/quality")
    return results


def load_panickserry_cache(
    cache_dir: Path,
    winrate_dir: Path,
    dataset: str,
    logger: logging.Logger
) -> Dict[Tuple[str, str, str, str], Dict]:
    """
    Load J vs R and K vs R data from panickserry (CNN/XSUM) cache structure.

    Returns:
        Dict mapping (dataset, judge, reference, example_id) -> {
            'j_prob': float,
            'k_probs': List[float],
            'category': str
        }
    """
    results = {}

    # Load LSP/ILSP labels from winrate directory
    label_cache = {}

    def get_labels(judge: str, reference: str) -> Tuple[set, set]:
        """Get (lsp_ids, ilsp_ids) for a judge/reference pair."""
        cache_key = (judge, reference)
        if cache_key in label_cache:
            return label_cache[cache_key]

        # Normalize judge name for winrate file
        judge_norm = judge.replace("gpt-3.5-turbo", "GPT-3.5").replace("gpt-4", "GPT-4")

        winrate_file = winrate_dir / f"{judge_norm}_vs_{reference}.json"
        if not winrate_file.exists():
            label_cache[cache_key] = (set(), set())
            return label_cache[cache_key]

        try:
            with open(winrate_file) as f:
                data = json.load(f)

            # Aggregate LSP/ILSP IDs across all proxies
            lsp_ids = set()
            ilsp_ids = set()
            for proxy_name, proxy_info in data.get("proxies", {}).items():
                if proxy_name == 'oracle_gpt5':
                    continue
                lsp_ids.update(proxy_info.get("lsp_ids", []))
                ilsp_ids.update(proxy_info.get("ilsp_ids", []))

            label_cache[cache_key] = (lsp_ids, ilsp_ids)
        except Exception as e:
            logger.warning(f"Failed to load labels from {winrate_file}: {e}")
            label_cache[cache_key] = (set(), set())

        return label_cache[cache_key]

    # Walk cache_dir / dataset / cache / judge / reference
    cache_path = cache_dir / dataset / 'cache'
    if not cache_path.exists():
        logger.warning(f"Cache path not found: {cache_path}")
        return results

    for judge_dir in cache_path.iterdir():
        if not judge_dir.is_dir():
            continue
        judge = judge_dir.name

        for ref_dir in judge_dir.iterdir():
            if not ref_dir.is_dir():
                continue
            reference = ref_dir.name

            # Load J vs R games
            j_game1_file = ref_dir / "J_vs_R_game1.jsonl"
            j_game2_file = ref_dir / "J_vs_R_game2.jsonl"

            if not j_game1_file.exists() or not j_game2_file.exists():
                logger.debug(f"Missing J_vs_R files for {judge}/{reference}")
                continue

            # Load K vs R games
            k_game1_file = ref_dir / "K_vs_R_game1.jsonl"
            k_game2_file = ref_dir / "K_vs_R_game2.jsonl"

            if not k_game1_file.exists() or not k_game2_file.exists():
                logger.debug(f"Missing K_vs_R files for {judge}/{reference}")
                continue

            # Get LSP/ILSP labels - try different reference name variations
            # CNN uses: gpt3.5, human, llama2
            # XSUM uses: gpt-3.5/llama2 for gpt-3.5-turbo, gpt-3.5/llama for gpt-4
            ref_variations = [reference]
            if reference == 'gpt3.5':
                ref_variations = ['GPT-3.5', 'gpt-3.5']
            elif reference == 'gpt-3.5':
                ref_variations = ['GPT-3.5', 'gpt3.5']
            elif reference in ['llama', 'llama2']:
                ref_variations = ['llama2', 'llama']

            lsp_ids = set()
            ilsp_ids = set()
            for ref_var in ref_variations:
                lsp, ilsp = get_labels(judge, ref_var)
                if lsp or ilsp:
                    lsp_ids = lsp
                    ilsp_ids = ilsp
                    break

            if not lsp_ids and not ilsp_ids:
                logger.debug(f"No LSP/ILSP labels for {judge}/{reference}")
                continue

            # Load J game1
            j_game1_by_id = {}
            with open(j_game1_file) as f:
                for line in f:
                    entry = json.loads(line)
                    j_game1_by_id[entry['example_id']] = entry

            # Load J game2
            j_game2_by_id = {}
            with open(j_game2_file) as f:
                for line in f:
                    entry = json.loads(line)
                    j_game2_by_id[entry['example_id']] = entry

            # Load K games - group by (example_id, proxy)
            k_game1_by_id_proxy = defaultdict(dict)
            with open(k_game1_file) as f:
                for line in f:
                    entry = json.loads(line)
                    ex_id = entry['example_id']
                    proxy = entry.get('proxy', 'unknown')
                    k_game1_by_id_proxy[ex_id][proxy] = entry

            k_game2_by_id_proxy = defaultdict(dict)
            with open(k_game2_file) as f:
                for line in f:
                    entry = json.loads(line)
                    ex_id = entry['example_id']
                    proxy = entry.get('proxy', 'unknown')
                    k_game2_by_id_proxy[ex_id][proxy] = entry

            # Compute averaged probabilities for each example
            common_j_ids = set(j_game1_by_id.keys()) & set(j_game2_by_id.keys())

            for ex_id in common_j_ids:
                j_g1 = j_game1_by_id[ex_id]
                j_g2 = j_game2_by_id[ex_id]

                # J probability: avg of game1 prob_response1 and game2 prob_response2
                j_prob = (j_g1['prob_response1'] + j_g2['prob_response2']) / 2.0

                # Determine category based on LSP/ILSP labels
                if ex_id in lsp_ids:
                    category = 'lsp'
                elif ex_id in ilsp_ids:
                    category = 'ilsp'
                else:
                    category = j_g1.get('category', 'unknown')

                # K probabilities: avg across games for each proxy
                k_probs = []
                proxies_g1 = k_game1_by_id_proxy.get(ex_id, {})
                proxies_g2 = k_game2_by_id_proxy.get(ex_id, {})

                common_proxies = set(proxies_g1.keys()) & set(proxies_g2.keys())
                for proxy in common_proxies:
                    k_g1 = proxies_g1[proxy]
                    k_g2 = proxies_g2[proxy]
                    k_prob = (k_g1['prob_response1'] + k_g2['prob_response2']) / 2.0
                    k_probs.append(k_prob)

                if k_probs:
                    key = (dataset, judge, reference, ex_id)
                    results[key] = {
                        'j_prob': j_prob,
                        'k_probs': k_probs,
                        'category': category
                    }

    logger.info(f"Loaded {len(results)} examples from panickserry/{dataset}")
    return results


def load_all_data(logger: logging.Logger) -> Dict[Tuple[str, str, str, str], Dict]:
    """Load all data from all cache sources."""
    all_data = {}

    # Load verif datasets
    verif_base = Path("judge_swap_null_verif_smoke2")
    for dataset in VERIF_DATASETS:
        cache_dir = verif_base / dataset / "cache"
        if cache_dir.exists():
            data = load_verif_dbg_cache(cache_dir, logger)
            all_data.update(data)
        else:
            logger.warning(f"Cache not found: {cache_dir}")

    # Load DBG datasets
    dbg_base = Path("judge_swap_null_dbg_results")
    for dataset in DBG_DATASETS:
        cache_dir = dbg_base / dataset / "cache"
        if cache_dir.exists():
            data = load_verif_dbg_cache(cache_dir, logger)
            all_data.update(data)
        else:
            logger.warning(f"Cache not found: {cache_dir}")

    # Load author_obfuscation
    author_obf_dir = Path("judge_swap_null_author_obfuscation/quality")
    if author_obf_dir.exists():
        data = load_author_obf_cache(author_obf_dir, logger)
        all_data.update(data)
    else:
        logger.warning(f"Author obfuscation dir not found: {author_obf_dir}")

    # Load panickserry datasets (CNN and XSUM)
    panickserry_configs = [
        ("cnn", Path("CNN_and_XSUM results/cnn_results"), Path("CNN_and_XSUM results/cnn_winrates")),
        ("xsum", Path("CNN_and_XSUM results/xsum_result"), Path("CNN_and_XSUM results/xsum_winrates")),
    ]

    for dataset, cache_base, winrate_dir in panickserry_configs:
        if cache_base.exists() and winrate_dir.exists():
            data = load_panickserry_cache(cache_base, winrate_dir, dataset, logger)
            all_data.update(data)
        else:
            if not cache_base.exists():
                logger.warning(f"Panickserry cache not found: {cache_base}")
            if not winrate_dir.exists():
                logger.warning(f"Panickserry winrate dir not found: {winrate_dir}")

    logger.info(f"Total examples loaded: {len(all_data)}")
    return all_data


# -------------------------
# --- STATISTICS ---
# -------------------------

def compute_lobf_stats(
    x: np.ndarray,
    y: np.ndarray,
    force_origin: bool = False
) -> Dict:
    """
    Compute line of best fit statistics.

    Args:
        x: X values
        y: Y values
        force_origin: If True, force intercept = 0

    Returns:
        Dict with slope, intercept, r_squared, r_value, p_value, std_err
    """
    if len(x) < 2:
        return {
            'slope': float('nan'),
            'intercept': float('nan'),
            'r_squared': float('nan'),
            'r_value': float('nan'),
            'p_value': float('nan'),
            'std_err': float('nan'),
            'n': len(x)
        }

    if force_origin:
        # Fit y = slope * x (no intercept)
        slope = np.sum(x * y) / np.sum(x * x)
        intercept = 0.0
        y_pred = slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')

        # Compute correlation manually
        if np.std(x) > 0 and np.std(y) > 0:
            r_value = np.corrcoef(x, y)[0, 1]
        else:
            r_value = float('nan')

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'r_value': r_value,
            'p_value': float('nan'),  # Not computed for forced origin
            'std_err': float('nan'),
            'n': len(x)
        }
    else:
        result = linregress(x, y)
        return {
            'slope': result.slope,
            'intercept': result.intercept,
            'r_squared': result.rvalue ** 2,
            'r_value': result.rvalue,
            'p_value': result.pvalue,
            'std_err': result.stderr,
            'n': len(x)
        }


def compute_example_level_stats(
    data: Dict[Tuple[str, str, str, str], Dict],
    category_filter: str = 'ilsp'
) -> pd.DataFrame:
    """
    Compute per-example statistics.

    Returns DataFrame with columns:
        dataset, judge, reference, example_id, j_prob, k_prob_mean, diff, category
    """
    rows = []

    for (dataset, judge, reference, ex_id), entry in data.items():
        if category_filter and entry['category'] != category_filter:
            continue

        j_prob = entry['j_prob']
        k_prob_mean = np.mean(entry['k_probs'])
        diff = j_prob - k_prob_mean

        rows.append({
            'dataset': dataset,
            'judge': judge,
            'reference': reference,
            'example_id': ex_id,
            'j_prob': j_prob,
            'k_prob_mean': k_prob_mean,
            'diff': diff,
            'n_proxies': len(entry['k_probs']),
            'category': entry['category']
        })

    return pd.DataFrame(rows)


def compute_reference_level_stats(
    example_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute per-(dataset, judge, reference) statistics.

    Aggregates diffs per example within each (dataset, judge, reference).
    """
    rows = []

    grouped = example_df.groupby(['dataset', 'judge', 'reference'])

    for (dataset, judge, reference), group in grouped:
        j_probs = group['j_prob'].values
        k_probs = group['k_prob_mean'].values
        diffs = group['diff'].values

        n = len(diffs)
        mean_j = np.mean(j_probs)
        mean_k = np.mean(k_probs)
        mean_diff = np.mean(diffs)
        var_diff = np.var(diffs, ddof=1) if n > 1 else 0.0
        std_diff = np.std(diffs, ddof=1) if n > 1 else 0.0
        se_diff = std_diff / np.sqrt(n) if n > 0 else 0.0

        # t-test: H0: mean_diff = 0
        if n >= 2:
            t_stat, p_value = stats.ttest_1samp(diffs, 0.0)
        else:
            t_stat, p_value = float('nan'), float('nan')

        # LOBF stats (free intercept)
        lobf_free = compute_lobf_stats(k_probs, j_probs, force_origin=False)
        # LOBF stats (forced origin)
        lobf_origin = compute_lobf_stats(k_probs, j_probs, force_origin=True)

        rows.append({
            'dataset': dataset,
            'judge': judge,
            'reference': reference,
            'n_examples': n,
            'mean_j_vs_r': mean_j,
            'mean_k_vs_r': mean_k,
            'mean_diff': mean_diff,
            'var_diff': var_diff,
            'std_diff': std_diff,
            'se_diff': se_diff,
            't_stat': t_stat,
            'ttest_p': p_value,
            'slope_free': lobf_free['slope'],
            'intercept_free': lobf_free['intercept'],
            'r_squared_free': lobf_free['r_squared'],
            'slope_origin': lobf_origin['slope'],
            'r_squared_origin': lobf_origin['r_squared'],
            'family': extract_model_family(judge),
            'size': extract_model_size(judge)
        })

    return pd.DataFrame(rows)


def compute_judge_level_stats(
    example_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute per-(dataset, judge) statistics.

    Aggregates diffs across all references for each (dataset, judge).
    """
    rows = []

    grouped = example_df.groupby(['dataset', 'judge'])

    for (dataset, judge), group in grouped:
        j_probs = group['j_prob'].values
        k_probs = group['k_prob_mean'].values
        diffs = group['diff'].values

        n = len(diffs)
        n_refs = group['reference'].nunique()
        mean_j = np.mean(j_probs)
        mean_k = np.mean(k_probs)
        mean_diff = np.mean(diffs)
        var_diff = np.var(diffs, ddof=1) if n > 1 else 0.0
        std_diff = np.std(diffs, ddof=1) if n > 1 else 0.0
        se_diff = std_diff / np.sqrt(n) if n > 0 else 0.0

        # t-test: H0: mean_diff = 0
        if n >= 2:
            t_stat, p_value = stats.ttest_1samp(diffs, 0.0)
        else:
            t_stat, p_value = float('nan'), float('nan')

        # LOBF stats
        lobf_free = compute_lobf_stats(k_probs, j_probs, force_origin=False)
        lobf_origin = compute_lobf_stats(k_probs, j_probs, force_origin=True)

        rows.append({
            'dataset': dataset,
            'judge': judge,
            'n_examples': n,
            'n_references': n_refs,
            'mean_j_vs_r': mean_j,
            'mean_k_vs_r': mean_k,
            'mean_diff': mean_diff,
            'var_diff': var_diff,
            'std_diff': std_diff,
            'se_diff': se_diff,
            't_stat': t_stat,
            'ttest_p': p_value,
            'slope_free': lobf_free['slope'],
            'intercept_free': lobf_free['intercept'],
            'r_squared_free': lobf_free['r_squared'],
            'slope_origin': lobf_origin['slope'],
            'r_squared_origin': lobf_origin['r_squared'],
            'family': extract_model_family(judge),
            'size': extract_model_size(judge)
        })

    return pd.DataFrame(rows)


# -------------------------
# --- VISUALIZATION ---
# -------------------------

def create_per_example_scatter(
    example_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
    subsample_n: int = 2000
) -> Dict:
    """
    Create scatter plot with one point per (dataset, example_id, judge, reference).

    X-axis: mean P(K) across proxies
    Y-axis: P(J)

    Returns correlation report dict.
    """
    logger.info(f"Creating per-example scatter plot (n={len(example_df)}, subsample={subsample_n})")

    # Get all data for computing stats
    x_all = example_df['k_prob_mean'].values
    y_all = example_df['j_prob'].values

    # Compute global LOBF (both versions)
    lobf_free = compute_lobf_stats(x_all, y_all, force_origin=False)
    lobf_origin = compute_lobf_stats(x_all, y_all, force_origin=True)

    # Decide which to use: prefer higher R²
    use_origin = lobf_origin['r_squared'] > lobf_free['r_squared']
    lobf_main = lobf_origin if use_origin else lobf_free

    logger.info(f"Global LOBF (free): R²={lobf_free['r_squared']:.4f}, slope={lobf_free['slope']:.4f}, intercept={lobf_free['intercept']:.4f}")
    logger.info(f"Global LOBF (origin): R²={lobf_origin['r_squared']:.4f}, slope={lobf_origin['slope']:.4f}")
    logger.info(f"Using {'origin' if use_origin else 'free'} intercept for plot")

    # Subsample for visualization
    if len(example_df) > subsample_n:
        plot_df = example_df.sample(n=subsample_n, random_state=42)
    else:
        plot_df = example_df

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot points with family colors, lower alpha
    for family in plot_df['family'].unique():
        mask = plot_df['family'] == family
        color = FAMILY_COLORS.get(family, FAMILY_COLORS['other'])
        ax.scatter(
            plot_df.loc[mask, 'k_prob_mean'],
            plot_df.loc[mask, 'j_prob'],
            c=color,
            alpha=0.25,
            s=20,
            label=family.capitalize(),
            edgecolors='none'
        )

    # Plot LOBF
    x_line = np.linspace(0, 1, 100)
    if use_origin:
        y_line = lobf_main['slope'] * x_line
    else:
        y_line = lobf_main['slope'] * x_line + lobf_main['intercept']
    ax.plot(x_line, y_line, 'k-', linewidth=2, label='LOBF')

    # Plot y=x reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='y=x')

    ax.set_xlabel('Mean P(K) across proxies', fontsize=12)
    ax.set_ylabel('P(J)', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # Title with stats
    ax.set_title(
        f'Per-Example: P(J|JvR) vs Mean P(K|KvR)\n'
        f'n={len(example_df):,} (showing {len(plot_df):,}), '
        f'R²={lobf_main["r_squared"]:.4f}',
        fontsize=11
    )

    plt.tight_layout()

    # Save
    for fmt in ['png', 'pdf', 'svg']:
        out_path = output_dir / 'plots' / f'per_example_scatter_global.{fmt}'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
    logger.info(f"Saved per-example scatter to {output_dir / 'plots'}")

    # Compute per-family, per-dataset, per-judge LOBFs
    correlation_report = {
        'global': {
            'free': lobf_free,
            'origin': lobf_origin,
            'used': 'origin' if use_origin else 'free'
        },
        'by_family': {},
        'by_dataset': {},
        'by_judge': {}
    }

    # Per-family
    for family in example_df['family'].unique():
        mask = example_df['family'] == family
        x_fam = example_df.loc[mask, 'k_prob_mean'].values
        y_fam = example_df.loc[mask, 'j_prob'].values
        lobf_fam_free = compute_lobf_stats(x_fam, y_fam, force_origin=False)
        lobf_fam_origin = compute_lobf_stats(x_fam, y_fam, force_origin=True)
        correlation_report['by_family'][family] = {
            'free': lobf_fam_free,
            'origin': lobf_fam_origin
        }

    # Per-dataset
    for dataset in example_df['dataset'].unique():
        mask = example_df['dataset'] == dataset
        x_ds = example_df.loc[mask, 'k_prob_mean'].values
        y_ds = example_df.loc[mask, 'j_prob'].values
        lobf_ds_free = compute_lobf_stats(x_ds, y_ds, force_origin=False)
        lobf_ds_origin = compute_lobf_stats(x_ds, y_ds, force_origin=True)
        correlation_report['by_dataset'][dataset] = {
            'free': lobf_ds_free,
            'origin': lobf_ds_origin
        }

    # Per-judge (top 10 by sample count)
    judge_counts = example_df['judge'].value_counts()
    top_judges = judge_counts.head(10).index.tolist()
    for judge in top_judges:
        mask = example_df['judge'] == judge
        x_j = example_df.loc[mask, 'k_prob_mean'].values
        y_j = example_df.loc[mask, 'j_prob'].values
        lobf_j_free = compute_lobf_stats(x_j, y_j, force_origin=False)
        lobf_j_origin = compute_lobf_stats(x_j, y_j, force_origin=True)
        correlation_report['by_judge'][judge] = {
            'free': lobf_j_free,
            'origin': lobf_j_origin
        }

    # Find best sub-LOBFs
    best_family = max(correlation_report['by_family'].items(),
                      key=lambda x: max(x[1]['free']['r_squared'], x[1]['origin']['r_squared']))
    best_dataset = max(correlation_report['by_dataset'].items(),
                       key=lambda x: max(x[1]['free']['r_squared'], x[1]['origin']['r_squared']))

    logger.info(f"Best family LOBF: {best_family[0]} (R²={max(best_family[1]['free']['r_squared'], best_family[1]['origin']['r_squared']):.4f})")
    logger.info(f"Best dataset LOBF: {best_dataset[0]} (R²={max(best_dataset[1]['free']['r_squared'], best_dataset[1]['origin']['r_squared']):.4f})")

    return correlation_report


def create_per_judge_scatter(
    judge_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> Dict:
    """
    Create scatter plot with one point per (dataset, judge).

    X-axis: mean P(K)
    Y-axis: mean P(J)
    """
    logger.info(f"Creating per-judge scatter plot (n={len(judge_df)})")

    # Get all data
    x_all = judge_df['mean_k_vs_r'].values
    y_all = judge_df['mean_j_vs_r'].values

    # Compute LOBF
    lobf_free = compute_lobf_stats(x_all, y_all, force_origin=False)
    lobf_origin = compute_lobf_stats(x_all, y_all, force_origin=True)
    use_origin = lobf_origin['r_squared'] > lobf_free['r_squared']
    lobf_main = lobf_origin if use_origin else lobf_free

    logger.info(f"Judge-level LOBF (free): R²={lobf_free['r_squared']:.4f}")
    logger.info(f"Judge-level LOBF (origin): R²={lobf_origin['r_squared']:.4f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by size (largest first) for layering
    judge_df_sorted = judge_df.sort_values('size', ascending=False)

    # Plot each point
    for _, row in judge_df_sorted.iterrows():
        family = row['family']
        size = row['size']
        color = FAMILY_COLORS.get(family, FAMILY_COLORS['other'])
        marker_size = size_to_marker_size(size)

        ax.scatter(
            row['mean_k_vs_r'],
            row['mean_j_vs_r'],
            c=color,
            s=marker_size,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )

    # Plot LOBF
    x_line = np.linspace(0, 1, 100)
    if use_origin:
        y_line = lobf_main['slope'] * x_line
    else:
        y_line = lobf_main['slope'] * x_line + lobf_main['intercept']
    ax.plot(x_line, y_line, 'k-', linewidth=2)

    # Plot y=x reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Mean P(K) (aggregated across examples)', fontsize=12)
    ax.set_ylabel('Mean P(J) (aggregated across examples)', fontsize=12)
    ax.set_xlim(0.3, 0.9)
    ax.set_ylim(0.3, 0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Legend for families
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    families_present = judge_df['family'].unique()
    family_handles = [
        mpatches.Patch(color=FAMILY_COLORS.get(f, FAMILY_COLORS['other']), label=f.capitalize())
        for f in sorted(families_present)
    ]

    # Size legend
    sizes_present = judge_df['size'].unique()
    size_handles = []
    size_ranges = [(3, '≤3B', 80), (7, '4-9B', 200), (14, '10-32B', 400), (70, '≥70B', 700)]
    for ref_size, label, marker_size in size_ranges:
        has_size = False
        if ref_size == 3 and any(s <= 3 for s in sizes_present):
            has_size = True
        elif ref_size == 7 and any(4 <= s <= 9 for s in sizes_present):
            has_size = True
        elif ref_size == 14 and any(10 <= s <= 32 for s in sizes_present):
            has_size = True
        elif ref_size == 70 and any(s >= 70 for s in sizes_present):
            has_size = True

        if has_size:
            size_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=np.sqrt(marker_size / np.pi),
                       label=label, markeredgecolor='black', markeredgewidth=0.5)
            )

    # Combine legends
    all_handles = family_handles + size_handles
    ax.legend(handles=all_handles, loc='upper left', fontsize=9, ncol=2)

    ax.set_title(
        f'Per-Judge: Mean P(J) vs Mean P(K)\n'
        f'n={len(judge_df)}, R²={lobf_main["r_squared"]:.4f}',
        fontsize=11
    )

    plt.tight_layout()

    # Save
    for fmt in ['png', 'pdf', 'svg']:
        out_path = output_dir / 'plots' / f'per_judge_scatter.{fmt}'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
    logger.info(f"Saved per-judge scatter")

    return {
        'free': lobf_free,
        'origin': lobf_origin,
        'used': 'origin' if use_origin else 'free'
    }


def create_per_reference_scatter_by_paper(
    ref_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Create scatter plots by paper group.

    One row per paper, one subplot per dataset.
    """
    logger.info("Creating per-reference scatter plots by paper")

    # Determine layout
    max_datasets = max(len(ds) for ds in PAPER_GROUPS.values())
    n_papers = len(PAPER_GROUPS)

    fig, axes = plt.subplots(n_papers, max_datasets, figsize=(5 * max_datasets, 5 * n_papers))
    if n_papers == 1:
        axes = [axes]

    for row_idx, (paper_name, datasets) in enumerate(PAPER_GROUPS.items()):
        for col_idx in range(max_datasets):
            ax = axes[row_idx][col_idx] if max_datasets > 1 else axes[row_idx]

            if col_idx >= len(datasets):
                ax.set_visible(False)
                continue

            dataset = datasets[col_idx]
            df_ds = ref_df[ref_df['dataset'] == dataset]

            if len(df_ds) == 0:
                ax.text(0.5, 0.5, f'No data for {dataset}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(dataset)
                continue

            # Get data
            x = df_ds['mean_k_vs_r'].values
            y = df_ds['mean_j_vs_r'].values

            # Compute LOBF
            lobf_free = compute_lobf_stats(x, y, force_origin=False)
            lobf_origin = compute_lobf_stats(x, y, force_origin=True)
            use_origin = lobf_origin['r_squared'] > lobf_free['r_squared']
            lobf_main = lobf_origin if use_origin else lobf_free

            # Sort by size for layering
            df_ds_sorted = df_ds.sort_values('size', ascending=False)

            # Plot points
            for _, row in df_ds_sorted.iterrows():
                family = row['family']
                size = row['size']
                color = FAMILY_COLORS.get(family, FAMILY_COLORS['other'])
                marker_size = size_to_marker_size(size)

                ax.scatter(
                    row['mean_k_vs_r'],
                    row['mean_j_vs_r'],
                    c=color,
                    s=marker_size,
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=0.5
                )

            # Plot LOBF
            x_line = np.linspace(0, 1, 100)
            if use_origin:
                y_line = lobf_main['slope'] * x_line
            else:
                y_line = lobf_main['slope'] * x_line + lobf_main['intercept']
            ax.plot(x_line, y_line, 'k-', linewidth=2)

            # y=x reference
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

            ax.set_xlim(0.3, 0.9)
            ax.set_ylim(0.3, 0.9)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            ax.set_title(f'{dataset}\n(n={len(df_ds)}, R²={lobf_main["r_squared"]:.3f})', fontsize=10)

            if col_idx == 0:
                ax.set_ylabel(f'{paper_name}\nMean P(J)', fontsize=10)
            if row_idx == n_papers - 1:
                ax.set_xlabel('Mean P(K)', fontsize=10)

    # Add legend
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    families_present = ref_df['family'].unique()
    family_handles = [
        mpatches.Patch(color=FAMILY_COLORS.get(f, FAMILY_COLORS['other']), label=f.capitalize())
        for f in sorted(families_present)
    ]

    sizes_present = ref_df['size'].unique()
    size_handles = []
    size_ranges = [(3, '≤3B', 80), (7, '4-9B', 200), (14, '10-32B', 400), (70, '≥70B', 700)]
    for ref_size, label, marker_size in size_ranges:
        has_size = False
        if ref_size == 3 and any(s <= 3 for s in sizes_present):
            has_size = True
        elif ref_size == 7 and any(4 <= s <= 9 for s in sizes_present):
            has_size = True
        elif ref_size == 14 and any(10 <= s <= 32 for s in sizes_present):
            has_size = True
        elif ref_size == 70 and any(s >= 70 for s in sizes_present):
            has_size = True

        if has_size:
            size_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=np.sqrt(marker_size / np.pi),
                       label=label, markeredgecolor='black', markeredgewidth=0.5)
            )

    fig.legend(handles=family_handles + size_handles,
               loc='lower center', ncol=len(family_handles) + len(size_handles),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Per-Reference: Mean P(J) vs Mean P(K) by Paper Group', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    # Save
    for fmt in ['png', 'pdf', 'svg']:
        out_path = output_dir / 'plots' / f'per_reference_scatter_by_paper.{fmt}'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
    logger.info("Saved per-reference scatter by paper")


# -------------------------
# --- MAIN ---
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="J vs R / K vs R scatter analysis")
    parser.add_argument("--output_dir", type=str, default="jvr_kvr_analysis",
                       help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Load all data
    logger.info("Loading all data...")
    all_data = load_all_data(logger)

    if not all_data:
        logger.error("No data loaded. Exiting.")
        return

    # Compute example-level stats for ILSP
    logger.info("\n" + "=" * 80)
    logger.info("Computing ILSP statistics...")
    logger.info("=" * 80)

    example_df_ilsp = compute_example_level_stats(all_data, category_filter='ilsp')
    logger.info(f"ILSP examples: {len(example_df_ilsp)}")

    # Add family column
    example_df_ilsp['family'] = example_df_ilsp['judge'].apply(extract_model_family)

    # Compute reference-level stats
    ref_df_ilsp = compute_reference_level_stats(example_df_ilsp, logger)
    logger.info(f"ILSP (dataset, judge, ref) combinations: {len(ref_df_ilsp)}")

    # Compute judge-level stats
    judge_df_ilsp = compute_judge_level_stats(example_df_ilsp, logger)
    logger.info(f"ILSP (dataset, judge) combinations: {len(judge_df_ilsp)}")

    # Also compute LSP stats for completeness
    logger.info("\n" + "=" * 80)
    logger.info("Computing LSP statistics...")
    logger.info("=" * 80)

    example_df_lsp = compute_example_level_stats(all_data, category_filter='lsp')
    example_df_lsp['family'] = example_df_lsp['judge'].apply(extract_model_family)
    logger.info(f"LSP examples: {len(example_df_lsp)}")

    ref_df_lsp = compute_reference_level_stats(example_df_lsp, logger)
    judge_df_lsp = compute_judge_level_stats(example_df_lsp, logger)

    # Save CSVs
    logger.info("\n" + "=" * 80)
    logger.info("Saving CSV files...")
    logger.info("=" * 80)

    csv_dir = output_dir / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Level A: (dataset, judge)
    judge_df_ilsp.to_csv(csv_dir / 'statistics_by_dataset_judge.csv', index=False)
    judge_df_lsp.to_csv(csv_dir / 'statistics_by_dataset_judge_lsp.csv', index=False)
    logger.info(f"Saved judge-level CSVs")

    # Level B: (dataset, judge, reference)
    ref_df_ilsp.to_csv(csv_dir / 'statistics_by_dataset_judge_reference.csv', index=False)
    ref_df_lsp.to_csv(csv_dir / 'statistics_by_dataset_judge_reference_lsp.csv', index=False)
    logger.info(f"Saved reference-level CSVs")

    # Create plots
    logger.info("\n" + "=" * 80)
    logger.info("Creating plots...")
    logger.info("=" * 80)

    # Per-example scatter
    corr_report_example = create_per_example_scatter(example_df_ilsp, output_dir, logger)

    # Per-judge scatter
    corr_report_judge = create_per_judge_scatter(judge_df_ilsp, output_dir, logger)

    # Per-reference scatter by paper
    create_per_reference_scatter_by_paper(ref_df_ilsp, output_dir, logger)

    # Save correlation report
    correlation_report = {
        'per_example': corr_report_example,
        'per_judge': corr_report_judge
    }

    with open(output_dir / 'correlation_report.json', 'w') as f:
        json.dump(correlation_report, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    logger.info(f"\nSaved correlation report to {output_dir / 'correlation_report.json'}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total ILSP examples: {len(example_df_ilsp)}")
    logger.info(f"Total LSP examples: {len(example_df_lsp)}")
    logger.info(f"ILSP (dataset, judge) pairs: {len(judge_df_ilsp)}")
    logger.info(f"ILSP (dataset, judge, ref) triplets: {len(ref_df_ilsp)}")
    logger.info(f"Global ILSP R² (per-example): {corr_report_example['global']['free']['r_squared']:.4f} (free), {corr_report_example['global']['origin']['r_squared']:.4f} (origin)")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
