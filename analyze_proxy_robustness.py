# analyze_proxy_robustness.py: Comprehensive proxy robustness analysis
# Part 1: Mean difference stability across varying proxy counts
# Part 2: Gold winrate balance verification via scatter plots
# Part 3: Judge-proxy composition tables
# Part 4: Sensitivity analysis (out-of-family proxies)
# Created: January 23, 2026, 11:00 AM EST
# Last Modified: January 23, 2026, 02:00 PM EST

import argparse
import json
import logging
import os
import sys
import getpass
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from glob import glob
from scipy.stats import linregress, pearsonr


# -------------------------
# --- CONSTANTS ---
# -------------------------

VERIF_DATASETS = ["math500", "mmlu", "mbpp-plus"]
DBG_DATASETS = ["alpaca_eval", "translation", "truthfulness"]
AUTHOR_OBF_DATASETS = ["quality"]
PANICKSERRY_DATASETS = ["cnn", "xsum"]

ALL_DATASETS = VERIF_DATASETS + DBG_DATASETS + AUTHOR_OBF_DATASETS + PANICKSERRY_DATASETS

PAPER_GROUPS = {
    "llm-sp-verif": VERIF_DATASETS,
    "dbg-score-paper": DBG_DATASETS,
    "author_obfuscation": AUTHOR_OBF_DATASETS,
    "panickserry": PANICKSERRY_DATASETS,
}

DATASET_TO_PAPER = {}
for paper, datasets in PAPER_GROUPS.items():
    for ds in datasets:
        DATASET_TO_PAPER[ds] = paper

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

# Category colors for Part 1
CATEGORY_COLORS = {
    'ilsp': '#E74C3C',   # Red
    'lsp': '#2ECC71',    # Green
    'joint': '#4A90E2'   # Blue
}

RANDOM_SEED = 42

# Font sizes for larger text
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12


# -------------------------
# --- LOGGING SETUP ---
# -------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging with file and console handlers."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analyze_proxy_robustness_{timestamp}.log"

    logger = logging.getLogger("analyze_proxy_robustness")
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
    logger.info("PROXY ROBUSTNESS ANALYSIS")
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
        return 100
    elif size <= 9:
        return 250
    elif size <= 32:
        return 450
    else:
        return 750


def get_paper_marker(paper: str) -> str:
    """Get marker shape for paper."""
    markers = {
        'llm-sp-verif': 'o',      # circle
        'dbg-score-paper': 's',   # square
        'author_obfuscation': '^' # triangle
    }
    return markers.get(paper, 'o')


# -------------------------
# --- DATA LOADING: CACHES ---
# -------------------------

def load_verif_dbg_cache(
    cache_dir: Path,
    logger: logging.Logger
) -> Dict[Tuple[str, str, str, str], Dict]:
    """
    Load J vs R and K vs R data from verif/dbg cache structure.

    Returns:
        Dict mapping (dataset, judge, reference, example_id) -> {
            'j_prob': float,
            'k_probs': Dict[str, float],  # proxy_name -> averaged prob
            'category': str
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

            # Load K vs R games
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

            # Compute averaged probabilities
            common_j_ids = set(j_game1_by_id.keys()) & set(j_game2_by_id.keys())

            for ex_id in common_j_ids:
                j_g1 = j_game1_by_id[ex_id]
                j_g2 = j_game2_by_id[ex_id]

                j_prob = (j_g1['prob_response1'] + j_g2['prob_response2']) / 2.0
                category = j_g1.get('category', 'unknown')

                # K probabilities per proxy
                k_probs = {}
                proxies_g1 = k_game1_by_id_proxy.get(ex_id, {})
                proxies_g2 = k_game2_by_id_proxy.get(ex_id, {})

                common_proxies = set(proxies_g1.keys()) & set(proxies_g2.keys())
                for proxy in common_proxies:
                    k_g1 = proxies_g1[proxy]
                    k_g2 = proxies_g2[proxy]
                    k_prob = (k_g1['prob_response1'] + k_g2['prob_response2']) / 2.0
                    k_probs[proxy] = k_prob

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
    """
    results = {}
    dataset = "quality"

    # Load LSP/ILSP labels from proxy directory
    proxy_dir = Path("author_obfuscation/data/quality/proxies")
    label_cache = {}

    def get_labels(judge: str, reference: str) -> Tuple[set, set]:
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
                    k_game1_by_id[item['example_id']].append((proxy, item))

                for item in data.get('K_vs_R_game2', []):
                    k_game2_by_id[item['example_id']].append((proxy, item))

                common_j_ids = set(j_game1_by_id.keys()) & set(j_game2_by_id.keys())

                for ex_id in common_j_ids:
                    j_g1 = j_game1_by_id[ex_id]
                    j_g2 = j_game2_by_id[ex_id]

                    j_prob = (j_g1['prob_A'] + j_g2['prob_B']) / 2.0

                    if ex_id in lsp_ids:
                        category = 'lsp'
                    elif ex_id in ilsp_ids:
                        category = 'ilsp'
                    else:
                        category = j_g1.get('category', 'unknown')

                    # K probabilities
                    k_g1_list = k_game1_by_id.get(ex_id, [])
                    k_g2_list = k_game2_by_id.get(ex_id, [])

                    key = (dataset, judge, reference, ex_id)

                    if key not in results:
                        results[key] = {
                            'j_prob': j_prob,
                            'k_probs': {},
                            'category': category
                        }

                    # Add K probs from this file
                    for (p, k_g1), (_, k_g2) in zip(k_g1_list, k_g2_list):
                        k_prob = (k_g1['prob_A'] + k_g2['prob_B']) / 2.0
                        results[key]['k_probs'][p] = k_prob

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue

    logger.info(f"Loaded {len(results)} examples from author_obfuscation/quality")
    return results


def load_cnn_cache(
    cache_dir: Path,
    proxy_dir: Path,
    logger: logging.Logger
) -> Dict[Tuple[str, str, str, str], Dict]:
    """
    Load J vs R and K vs R data from CNN cache structure.

    Args:
        cache_dir: panickserry_results/cnn_results/cnn/cache
        proxy_dir: panickserry_results/cnn_winrates
    """
    results = {}
    dataset = "cnn"

    # Load LSP/ILSP labels from proxy directory
    label_cache = {}

    def get_labels(judge: str, reference: str) -> Tuple[set, set]:
        cache_key = (judge, reference)
        if cache_key in label_cache:
            return label_cache[cache_key]

        # Normalize judge name for file matching
        judge_norm = judge.replace("gpt-3.5-turbo", "GPT-3.5").replace("gpt-4", "GPT-4")
        ref_norm = reference.replace("llama2", "llama2").replace("human", "human")

        proxy_file = proxy_dir / f"{judge_norm}_vs_{ref_norm}.json"
        if not proxy_file.exists():
            label_cache[cache_key] = (set(), set())
            return label_cache[cache_key]

        try:
            with open(proxy_file) as f:
                data = json.load(f)

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

            # Load K vs R games
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

            # Get LSP/ILSP labels
            lsp_ids, ilsp_ids = get_labels(judge, reference)

            # Compute averaged probabilities
            common_j_ids = set(j_game1_by_id.keys()) & set(j_game2_by_id.keys())

            for ex_id in common_j_ids:
                j_g1 = j_game1_by_id[ex_id]
                j_g2 = j_game2_by_id[ex_id]

                j_prob = (j_g1['prob_response1'] + j_g2['prob_response2']) / 2.0

                # Determine category from labels
                if ex_id in lsp_ids:
                    category = 'lsp'
                elif ex_id in ilsp_ids:
                    category = 'ilsp'
                else:
                    category = j_g1.get('category', 'unknown')

                # K probabilities per proxy
                k_probs = {}
                proxies_g1 = k_game1_by_id_proxy.get(ex_id, {})
                proxies_g2 = k_game2_by_id_proxy.get(ex_id, {})

                common_proxies = set(proxies_g1.keys()) & set(proxies_g2.keys())
                for proxy in common_proxies:
                    k_g1 = proxies_g1[proxy]
                    k_g2 = proxies_g2[proxy]
                    k_prob = (k_g1['prob_response1'] + k_g2['prob_response2']) / 2.0
                    k_probs[proxy] = k_prob

                if k_probs:
                    key = (dataset, judge, reference, ex_id)
                    results[key] = {
                        'j_prob': j_prob,
                        'k_probs': k_probs,
                        'category': category
                    }

    logger.info(f"Loaded {len(results)} examples from CNN cache")
    return results


def load_xsum_cache(
    cache_dir: Path,
    proxy_dir: Path,
    logger: logging.Logger
) -> Dict[Tuple[str, str, str, str], Dict]:
    """
    Load J vs R and K vs R data from XSUM cache structure.

    Args:
        cache_dir: panickserry_results/xsum_result/xsum/cache
        proxy_dir: panickserry_results/xsum_winrates
    """
    # XSUM has the same structure as CNN
    results = {}
    dataset = "xsum"

    # Load LSP/ILSP labels from proxy directory
    label_cache = {}

    def get_labels(judge: str, reference: str) -> Tuple[set, set]:
        cache_key = (judge, reference)
        if cache_key in label_cache:
            return label_cache[cache_key]

        # Normalize judge name for file matching
        judge_norm = judge.replace("gpt-3.5-turbo", "GPT-3.5").replace("gpt-4", "GPT-4")
        ref_norm = reference

        proxy_file = proxy_dir / f"{judge_norm}_vs_{ref_norm}.json"
        if not proxy_file.exists():
            label_cache[cache_key] = (set(), set())
            return label_cache[cache_key]

        try:
            with open(proxy_file) as f:
                data = json.load(f)

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

            # Load K vs R games
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

            # Get LSP/ILSP labels
            lsp_ids, ilsp_ids = get_labels(judge, reference)

            # Compute averaged probabilities
            common_j_ids = set(j_game1_by_id.keys()) & set(j_game2_by_id.keys())

            for ex_id in common_j_ids:
                j_g1 = j_game1_by_id[ex_id]
                j_g2 = j_game2_by_id[ex_id]

                j_prob = (j_g1['prob_response1'] + j_g2['prob_response2']) / 2.0

                # Determine category from labels
                if ex_id in lsp_ids:
                    category = 'lsp'
                elif ex_id in ilsp_ids:
                    category = 'ilsp'
                else:
                    category = j_g1.get('category', 'unknown')

                # K probabilities per proxy
                k_probs = {}
                proxies_g1 = k_game1_by_id_proxy.get(ex_id, {})
                proxies_g2 = k_game2_by_id_proxy.get(ex_id, {})

                common_proxies = set(proxies_g1.keys()) & set(proxies_g2.keys())
                for proxy in common_proxies:
                    k_g1 = proxies_g1[proxy]
                    k_g2 = proxies_g2[proxy]
                    k_prob = (k_g1['prob_response1'] + k_g2['prob_response2']) / 2.0
                    k_probs[proxy] = k_prob

                if k_probs:
                    key = (dataset, judge, reference, ex_id)
                    results[key] = {
                        'j_prob': j_prob,
                        'k_probs': k_probs,
                        'category': category
                    }

    logger.info(f"Loaded {len(results)} examples from XSUM cache")
    return results


def load_all_cache_data(logger: logging.Logger) -> Dict[Tuple[str, str, str, str], Dict]:
    """Load all cache data from all sources."""
    all_data = {}

    # Verif datasets
    verif_base = Path("judge_swap_null_verif_smoke2")
    for dataset in VERIF_DATASETS:
        cache_dir = verif_base / dataset / "cache"
        if cache_dir.exists():
            data = load_verif_dbg_cache(cache_dir, logger)
            all_data.update(data)
        else:
            logger.warning(f"Cache not found: {cache_dir}")

    # DBG datasets
    dbg_base = Path("judge_swap_null_dbg_results")
    for dataset in DBG_DATASETS:
        cache_dir = dbg_base / dataset / "cache"
        if cache_dir.exists():
            data = load_verif_dbg_cache(cache_dir, logger)
            all_data.update(data)
        else:
            logger.warning(f"Cache not found: {cache_dir}")

    # Author obfuscation
    author_obf_dir = Path("judge_swap_null_author_obfuscation/quality")
    if author_obf_dir.exists():
        data = load_author_obf_cache(author_obf_dir, logger)
        all_data.update(data)
    else:
        logger.warning(f"Author obfuscation dir not found: {author_obf_dir}")

    # CNN
    cnn_cache_dir = Path("panickserry_results/cnn_results/cnn/cache")
    cnn_proxy_dir = Path("panickserry_results/cnn_winrates")
    if cnn_cache_dir.exists() and cnn_proxy_dir.exists():
        data = load_cnn_cache(cnn_cache_dir, cnn_proxy_dir, logger)
        all_data.update(data)
    else:
        logger.warning(f"CNN cache or proxy dir not found: {cnn_cache_dir}, {cnn_proxy_dir}")

    # XSUM
    xsum_cache_dir = Path("panickserry_results/xsum_result/xsum/cache")
    xsum_proxy_dir = Path("panickserry_results/xsum_winrates")
    if xsum_cache_dir.exists() and xsum_proxy_dir.exists():
        data = load_xsum_cache(xsum_cache_dir, xsum_proxy_dir, logger)
        all_data.update(data)
    else:
        logger.warning(f"XSUM cache or proxy dir not found: {xsum_cache_dir}, {xsum_proxy_dir}")

    logger.info(f"Total cache examples loaded: {len(all_data)}")
    return all_data


# -------------------------
# --- DATA LOADING: PROXY DEFINITIONS (for gold winrates) ---
# -------------------------

def load_proxy_definitions(logger: logging.Logger) -> Dict:
    """
    Load proxy definitions from all sources to get gold winrates.

    Returns:
        Dict with structure:
        {
            (dataset, judge, reference): {
                'judge_winrate': float,
                'proxies': {
                    proxy_name: {
                        'proxy_winrate': float,
                        'n_lsp': int,
                        'n_ilsp': int
                    }
                }
            }
        }
    """
    proxy_defs = {}

    # DBG datasets
    logger.info("Loading DBG proxy definitions...")
    dbg_base = Path("dbg-score-paper/proxy_preference_data")
    for dataset in DBG_DATASETS:
        dataset_dir = dbg_base / dataset
        if not dataset_dir.exists():
            logger.warning(f"DBG proxy dir not found: {dataset_dir}")
            continue

        for json_file in dataset_dir.glob("judge_*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                judge = data['judge_name']
                reference = data['reference_name']
                key = (dataset, judge, reference)

                # Get judge winrate (should be same across proxies)
                judge_winrate = None
                proxies_info = {}

                for proxy_name, proxy_data in data.get('proxies', {}).items():
                    if judge_winrate is None:
                        judge_winrate = proxy_data.get('judge_winrate')
                    else:
                        # Assert consistency
                        curr = proxy_data.get('judge_winrate', 0)
                        if curr is not None and judge_winrate is not None:
                            if abs(judge_winrate - curr) >= 1e-6:
                                logger.warning(f"Judge winrate mismatch for {key}: {judge_winrate} vs {curr}")

                    proxies_info[proxy_name] = {
                        'proxy_winrate': proxy_data.get('proxy_winrate'),
                        'n_lsp': proxy_data.get('n_right', 0),
                        'n_ilsp': proxy_data.get('n_wrong', 0)
                    }

                proxy_defs[key] = {
                    'judge_winrate': judge_winrate,
                    'proxies': proxies_info
                }

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    # Author obfuscation
    logger.info("Loading author obfuscation proxy definitions...")
    author_obf_proxy_dir = Path("author_obfuscation/data/quality/proxies")
    if author_obf_proxy_dir.exists():
        for json_file in author_obf_proxy_dir.glob("evaluator_*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                judge = data['reference_evaluator']
                reference = data['reference_evaluatee']
                dataset = 'quality'
                key = (dataset, judge, reference)

                # Get judge winrate (should be same across proxies)
                judge_winrate = None
                proxies_info = {}

                for proxy_name, proxy_data in data.get('proxies', {}).items():
                    if judge_winrate is None:
                        judge_winrate = proxy_data.get('reference_winrate')
                    else:
                        curr = proxy_data.get('reference_winrate', 0)
                        if curr is not None and judge_winrate is not None:
                            if abs(judge_winrate - curr) >= 1e-6:
                                logger.warning(f"Judge winrate mismatch for {key}")

                    proxies_info[proxy_name] = {
                        'proxy_winrate': proxy_data.get('proxy_winrate'),
                        'n_lsp': proxy_data.get('n_lsp', 0),
                        'n_ilsp': proxy_data.get('n_ilsp', 0)
                    }

                proxy_defs[key] = {
                    'judge_winrate': judge_winrate,
                    'proxies': proxies_info
                }

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    # Verif datasets
    logger.info("Loading verif proxy definitions...")
    verif_base = Path("llm-sp-verif")
    for dataset in VERIF_DATASETS:
        dataset_dir = verif_base / dataset
        if not dataset_dir.exists():
            logger.warning(f"Verif proxy dir not found: {dataset_dir}")
            continue

        # Find proxy files in family subdirectories
        for proxy_dir in dataset_dir.glob("*/proxies"):
            for json_file in proxy_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    judge = json_file.stem  # filename is judge name

                    for reference, ref_data in data.items():
                        key = (dataset, judge, reference)

                        judge_winrate = None
                        proxies_info = {}

                        for proxy_name, proxy_data in ref_data.items():
                            if not isinstance(proxy_data, dict):
                                continue

                            if judge_winrate is None:
                                judge_winrate = proxy_data.get('judge_winrate')
                            else:
                                curr_wr = proxy_data.get('judge_winrate')
                                if curr_wr is not None and judge_winrate is not None:
                                    if abs(judge_winrate - curr_wr) >= 1e-6:
                                        logger.warning(f"Judge winrate mismatch for {key}")

                            # Extract n_lsp, n_ilsp from data
                            lsp_data = proxy_data.get('data', {}).get('lsp', [])
                            ilsp_data = proxy_data.get('data', {}).get('ilsp', [])

                            proxies_info[proxy_name] = {
                                'proxy_winrate': proxy_data.get('proxy_winrate'),
                                'n_lsp': len(lsp_data),
                                'n_ilsp': len(ilsp_data)
                            }

                        if proxies_info:
                            proxy_defs[key] = {
                                'judge_winrate': judge_winrate,
                                'proxies': proxies_info
                            }

                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")

    # CNN dataset
    logger.info("Loading CNN proxy definitions...")
    cnn_proxy_dir = Path("panickserry_results/cnn_winrates")
    if cnn_proxy_dir.exists():
        for json_file in cnn_proxy_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                judge = data['reference_evaluator']
                reference = data['reference_evaluatee']

                # Normalize judge names to match cache
                judge_norm = judge.replace("GPT-3.5", "gpt-3.5-turbo").replace("GPT-4", "gpt-4")

                dataset = 'cnn'
                key = (dataset, judge_norm, reference)

                # Get judge winrate (should be same across proxies)
                judge_winrate = None
                proxies_info = {}

                for proxy_name, proxy_data in data.get('proxies', {}).items():
                    if judge_winrate is None:
                        judge_winrate = proxy_data.get('reference_winrate')
                    else:
                        curr = proxy_data.get('reference_winrate', 0)
                        if curr is not None and judge_winrate is not None:
                            if abs(judge_winrate - curr) >= 1e-6:
                                logger.warning(f"Judge winrate mismatch for {key}")

                    proxies_info[proxy_name] = {
                        'proxy_winrate': proxy_data.get('proxy_winrate'),
                        'n_lsp': proxy_data.get('n_lsp', 0),
                        'n_ilsp': proxy_data.get('n_ilsp', 0)
                    }

                proxy_defs[key] = {
                    'judge_winrate': judge_winrate,
                    'proxies': proxies_info
                }

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    # XSUM dataset
    logger.info("Loading XSUM proxy definitions...")
    xsum_proxy_dir = Path("panickserry_results/xsum_winrates")
    if xsum_proxy_dir.exists():
        for json_file in xsum_proxy_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                judge = data['reference_evaluator']
                reference = data['reference_evaluatee']

                # Normalize judge names to match cache
                judge_norm = judge.replace("GPT-3.5", "gpt-3.5-turbo").replace("GPT-4", "gpt-4")

                dataset = 'xsum'
                key = (dataset, judge_norm, reference)

                # Get judge winrate (should be same across proxies)
                judge_winrate = None
                proxies_info = {}

                for proxy_name, proxy_data in data.get('proxies', {}).items():
                    if judge_winrate is None:
                        judge_winrate = proxy_data.get('reference_winrate')
                    else:
                        curr = proxy_data.get('reference_winrate', 0)
                        if curr is not None and judge_winrate is not None:
                            if abs(judge_winrate - curr) >= 1e-6:
                                logger.warning(f"Judge winrate mismatch for {key}")

                    proxies_info[proxy_name] = {
                        'proxy_winrate': proxy_data.get('proxy_winrate'),
                        'n_lsp': proxy_data.get('n_lsp', 0),
                        'n_ilsp': proxy_data.get('n_ilsp', 0)
                    }

                proxy_defs[key] = {
                    'judge_winrate': judge_winrate,
                    'proxies': proxies_info
                }

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    logger.info(f"Loaded proxy definitions for {len(proxy_defs)} (dataset, judge, reference) triplets")
    return proxy_defs


# -------------------------
# --- PART 1: MEAN DIFF BY N_PROXIES ---
# -------------------------

def compute_mean_diff_by_nproxies(
    cache_data: Dict[Tuple[str, str, str, str], Dict],
    category_filter: Optional[str],  # 'ilsp', 'lsp', or None for joint
    max_proxies: int = 20,
    collapse_threshold: int = 8,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Compute mean difference (J - avg(K)) for each number of proxies.
    Collapses n >= collapse_threshold into one bin.

    Args:
        cache_data: The cache data
        category_filter: 'ilsp', 'lsp', or None for joint
        max_proxies: Maximum number of proxies to consider
        collapse_threshold: Collapse all n >= this into one bin
        logger: Logger

    Returns:
        DataFrame with columns: n_proxies, n_proxies_label, n_examples, mean_diff, std_diff
    """
    np.random.seed(RANDOM_SEED)

    # Filter by category if specified
    if category_filter:
        filtered_data = {k: v for k, v in cache_data.items() if v['category'] == category_filter}
    else:
        filtered_data = cache_data

    if logger:
        logger.info(f"Computing mean_diff by n_proxies for category={category_filter}, n_examples={len(filtered_data)}")

    # Count proxies per example
    proxy_counts = {k: len(v['k_probs']) for k, v in filtered_data.items()}
    max_available = max(proxy_counts.values()) if proxy_counts else 0

    results = []

    for n_proxies in range(1, min(max_available + 1, max_proxies + 1)):
        # Determine if this bin should be collapsed
        if n_proxies >= collapse_threshold:
            # Only process once for the collapsed bin
            if n_proxies > collapse_threshold:
                continue
            # This is the collapsed bin (n >= threshold)
            eligible = {k: v for k, v in filtered_data.items() if len(v['k_probs']) >= collapse_threshold}
            n_proxies_label = f"{collapse_threshold}+"
            actual_n = collapse_threshold  # Use threshold for sampling
        else:
            # Normal bin
            eligible = {k: v for k, v in filtered_data.items() if len(v['k_probs']) >= n_proxies}
            n_proxies_label = str(n_proxies)
            actual_n = n_proxies

        n_eligible = len(eligible)

        if n_eligible == 0:
            continue

        # For each eligible example, sample n_proxies and compute diff
        diffs = []
        for key, entry in eligible.items():
            j_prob = entry['j_prob']
            k_probs_dict = entry['k_probs']

            # Sample actual_n proxies without replacement (or all if collapsed)
            proxy_names = list(k_probs_dict.keys())
            n_to_sample = min(actual_n, len(proxy_names))
            sampled = np.random.choice(proxy_names, size=n_to_sample, replace=False)
            sampled_k_probs = [k_probs_dict[p] for p in sampled]
            avg_k = np.mean(sampled_k_probs)
            diff = j_prob - avg_k
            diffs.append(diff)

        diffs = np.array(diffs)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0

        results.append({
            'n_proxies': n_proxies if n_proxies < collapse_threshold else collapse_threshold,
            'n_proxies_label': n_proxies_label,
            'n_examples': n_eligible,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
        })

    return pd.DataFrame(results)


def compute_mean_diff_all_categories(
    cache_data: Dict[Tuple[str, str, str, str], Dict],
    max_proxies: int = 20,
    collapse_threshold: int = 8,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Compute mean_diff by n_proxies for all three categories (ILSP, LSP, joint).
    """
    all_results = []

    for category in ['ilsp', 'lsp', 'joint']:
        cat_filter = None if category == 'joint' else category
        df = compute_mean_diff_by_nproxies(
            cache_data, cat_filter, max_proxies, collapse_threshold, logger
        )
        if len(df) > 0:
            df['category'] = category
            all_results.append(df)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def plot_mean_diff_combined(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    logger: logging.Logger
):
    """
    Create combined plot with ILSP (red), LSP (green), and joint (blue) on one plot.
    Uses standard deviation for error bars.
    """
    if len(df) == 0:
        logger.warning(f"No data to plot for {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Get unique x positions
    x_positions = sorted(df['n_proxies'].unique())
    x_labels = []
    for x in x_positions:
        subset = df[df['n_proxies'] == x]
        if len(subset) > 0:
            x_labels.append(subset['n_proxies_label'].iloc[0])

    # Slight offset for overlapping categories
    offsets = {'ilsp': -0.15, 'lsp': 0.0, 'joint': 0.15}

    for category in ['ilsp', 'lsp', 'joint']:
        cat_df = df[df['category'] == category]
        if len(cat_df) == 0:
            continue

        color = CATEGORY_COLORS[category]
        offset = offsets[category]

        x_vals = cat_df['n_proxies'].values + offset

        # Plot with std as error bars
        ax.errorbar(
            x_vals, cat_df['mean_diff'],
            yerr=cat_df['std_diff'],
            fmt='o-', color=color, alpha=0.8, capsize=4, capthick=2,
            markersize=10, linewidth=2, label=category.upper()
        )

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Number of Proxies Used', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Mean Difference (J - avg(K))', fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)

    # Set x-ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)

    # Legend
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='upper right')

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ['png', 'pdf']:
        fig.savefig(output_path.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight')

    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")


def generate_part1_latex_table(
    all_stats: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger
):
    """Generate LaTeX table summarizing Part 1 results."""
    # Filter to global stats only
    global_df = all_stats[all_stats['group_type'] == 'global'].copy()

    if len(global_df) == 0:
        logger.warning("No global stats for LaTeX table")
        return

    # Pivot: rows = n_proxies, columns = category
    pivot = global_df.pivot(index='n_proxies_label', columns='category', values=['mean_diff', 'std_diff', 'n_examples'])

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean Difference (J - K) by Number of Proxies}",
        r"\label{tab:mean_diff_by_nproxies}",
        r"\begin{tabular}{l rrr rrr rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{ILSP} & \multicolumn{3}{c}{LSP} & \multicolumn{3}{c}{Joint} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r"$n$ & Mean & Std & $N$ & Mean & Std & $N$ & Mean & Std & $N$ \\",
        r"\midrule",
    ]

    # Sort by n_proxies
    n_values = sorted(global_df['n_proxies'].unique())

    for n in n_values:
        row_data = global_df[global_df['n_proxies'] == n]
        label = row_data['n_proxies_label'].iloc[0] if len(row_data) > 0 else str(n)

        parts = [label]
        for cat in ['ilsp', 'lsp', 'joint']:
            cat_row = row_data[row_data['category'] == cat]
            if len(cat_row) > 0:
                mean = cat_row['mean_diff'].iloc[0]
                std = cat_row['std_diff'].iloc[0]
                n_ex = int(cat_row['n_examples'].iloc[0])
                parts.extend([f"{mean:.3f}", f"{std:.3f}", f"{n_ex:,}"])
            else:
                parts.extend(["--", "--", "--"])

        latex_lines.append(" & ".join(parts) + r" \\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))

    logger.info(f"Saved LaTeX table: {output_path}")


def run_part1_analysis(
    cache_data: Dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Run Part 1: Mean difference stability across varying proxy counts.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: MEAN DIFFERENCE BY NUMBER OF PROXIES")
    logger.info("=" * 80)

    part1_dir = output_dir / "part1_mean_diff_by_nproxies"
    plots_dir = part1_dir / "plots"
    csv_dir = part1_dir / "csv"
    latex_dir = part1_dir / "latex"

    # Count overall category distribution
    n_ilsp = sum(1 for v in cache_data.values() if v['category'] == 'ilsp')
    n_lsp = sum(1 for v in cache_data.values() if v['category'] == 'lsp')
    n_total = n_ilsp + n_lsp
    logger.info(f"Category distribution: ILSP={n_ilsp}, LSP={n_lsp}, Total={n_total}")

    all_stats = []

    # --- Global level ---
    logger.info("\n--- Global level ---")
    df_global = compute_mean_diff_all_categories(cache_data, logger=logger)
    if len(df_global) > 0:
        df_global['group_type'] = 'global'
        df_global['group_key'] = 'all'
        all_stats.append(df_global)

        plot_mean_diff_combined(
            df_global,
            "Mean Difference (J - K) by # Proxies",
            plots_dir / "global_mean_diff_by_nproxies.png",
            logger
        )

    # --- By dataset level ---
    logger.info("\n--- Dataset level ---")
    grouped_by_dataset = defaultdict(dict)
    for (dataset, judge, reference, ex_id), entry in cache_data.items():
        grouped_by_dataset[dataset][(dataset, judge, reference, ex_id)] = entry

    for dataset, dataset_data in grouped_by_dataset.items():
        df_dataset = compute_mean_diff_all_categories(dataset_data, logger=logger)
        if len(df_dataset) > 0:
            df_dataset['group_type'] = 'dataset'
            df_dataset['group_key'] = dataset
            all_stats.append(df_dataset)

            plot_mean_diff_combined(
                df_dataset,
                f"{dataset}: Mean Difference (J - K) by # Proxies",
                plots_dir / "by_dataset" / f"{dataset}_mean_diff_by_nproxies.png",
                logger
            )

    # --- By judge level (model level) ---
    logger.info("\n--- Judge (model) level ---")
    grouped_by_judge = defaultdict(dict)
    for (dataset, judge, reference, ex_id), entry in cache_data.items():
        grouped_by_judge[judge][(dataset, judge, reference, ex_id)] = entry

    for judge, judge_data in grouped_by_judge.items():
        df_judge = compute_mean_diff_all_categories(judge_data, logger=logger)
        if len(df_judge) > 0:
            df_judge['group_type'] = 'judge'
            df_judge['group_key'] = judge
            all_stats.append(df_judge)

            # Clean judge name for filename
            judge_clean = judge.replace('/', '_').replace(' ', '_')
            plot_mean_diff_combined(
                df_judge,
                f"{judge}: Mean Diff by # Proxies",
                plots_dir / "by_judge" / f"{judge_clean}_mean_diff_by_nproxies.png",
                logger
            )

    # Combine and save all stats
    if all_stats:
        combined_df = pd.concat(all_stats, ignore_index=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(csv_dir / "all_mean_diff_by_nproxies_stats.csv", index=False)
        logger.info(f"Saved combined stats CSV with {len(combined_df)} rows")

        # Generate LaTeX table
        generate_part1_latex_table(combined_df, latex_dir / "mean_diff_by_nproxies.tex", logger)

    logger.info("Part 1 complete.")


# -------------------------
# --- PART 2: WINRATE BALANCE SCATTER ---
# -------------------------

def compute_winrate_balance_data(
    cache_data: Dict[Tuple[str, str, str, str], Dict],
    proxy_defs: Dict,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute data for winrate balance scatter plots.

    For each (dataset, judge, reference):
    - J's gold winrate against R
    - Weighted avg of K's gold winrates against R

    Returns DataFrame with columns:
        dataset, judge, reference, j_winrate, weighted_k_winrate, n_ilsp, n_lsp, family, size, paper
    """
    logger.info("Computing winrate balance data...")

    # Group cache data by (dataset, judge, reference)
    cache_by_triplet = defaultdict(list)
    for (dataset, judge, reference, ex_id), entry in cache_data.items():
        cache_by_triplet[(dataset, judge, reference)].append((ex_id, entry))

    rows = []

    for (dataset, judge, reference), examples in cache_by_triplet.items():
        # Get proxy definitions for this triplet
        key = (dataset, judge, reference)
        if key not in proxy_defs:
            logger.debug(f"No proxy definition for {key}")
            continue

        pdef = proxy_defs[key]
        j_winrate = pdef.get('judge_winrate')

        if j_winrate is None:
            logger.debug(f"No judge winrate for {key}")
            continue

        # Count ILSP examples and proxy usage
        n_ilsp = sum(1 for _, e in examples if e['category'] == 'ilsp')
        n_lsp = sum(1 for _, e in examples if e['category'] == 'lsp')

        if n_ilsp == 0:
            continue

        # Compute weighted K winrate
        # Weight = (1 / n_proxies_for_this_example) summed across examples
        proxy_weights = defaultdict(float)
        total_weight = 0

        for ex_id, entry in examples:
            if entry['category'] != 'ilsp':
                continue

            n_proxies = len(entry['k_probs'])
            if n_proxies == 0:
                continue

            weight_per_proxy = 1.0 / n_proxies
            for proxy_name in entry['k_probs'].keys():
                proxy_weights[proxy_name] += weight_per_proxy
                total_weight += weight_per_proxy

        if total_weight == 0:
            continue

        # Compute weighted average K winrate
        weighted_k_winrate = 0
        for proxy_name, weight in proxy_weights.items():
            proxy_info = pdef['proxies'].get(proxy_name, {})
            k_winrate = proxy_info.get('proxy_winrate')
            if k_winrate is not None:
                weighted_k_winrate += (weight / total_weight) * k_winrate

        rows.append({
            'dataset': dataset,
            'judge': judge,
            'reference': reference,
            'j_winrate': j_winrate,
            'weighted_k_winrate': weighted_k_winrate,
            'n_ilsp': n_ilsp,
            'n_lsp': n_lsp,
            'n_proxies_used': len(proxy_weights),
            'family': extract_model_family(judge),
            'size': extract_model_size(judge),
            'paper': DATASET_TO_PAPER.get(dataset, 'other')
        })

    df = pd.DataFrame(rows)
    logger.info(f"Computed winrate balance for {len(df)} (dataset, judge, reference) triplets")
    return df


def compute_binned_residuals(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Compute binned residuals for winrate balance data.

    For each (judge, reference) point i: r_i = y_i - x_i
    Bin by x (J's winrate) into deciles.

    Returns DataFrame with:
        bin_idx, bin_center, mean_residual, std_residual, se_residual, n_points
    """
    if len(df) < n_bins:
        n_bins = max(1, len(df) // 2)

    # Compute residuals
    df = df.copy()
    df['residual'] = df['weighted_k_winrate'] - df['j_winrate']

    # Sort by x (j_winrate) and assign bins
    df_sorted = df.sort_values('j_winrate').reset_index(drop=True)
    df_sorted['bin_idx'] = pd.cut(df_sorted.index, bins=n_bins, labels=False)

    # Compute stats per bin
    rows = []
    for bin_idx in range(n_bins):
        bin_df = df_sorted[df_sorted['bin_idx'] == bin_idx]
        if len(bin_df) == 0:
            continue

        bin_center = bin_df['j_winrate'].mean()
        residuals = bin_df['residual'].values
        mean_r = np.mean(residuals)
        std_r = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0
        se_r = std_r / np.sqrt(len(residuals)) if len(residuals) > 0 else 0.0

        rows.append({
            'bin_idx': bin_idx,
            'bin_center': bin_center,
            'mean_residual': mean_r,
            'std_residual': std_r,
            'se_residual': se_r,
            'ci_low': mean_r - 1.96 * se_r,
            'ci_high': mean_r + 1.96 * se_r,
            'n_points': len(bin_df)
        })

    return pd.DataFrame(rows)


def plot_winrate_balance_scatter_with_residuals(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    output_path: Path,
    logger: logging.Logger
):
    """
    Create scatter plot of J winrate vs weighted K winrate.
    Uses full legend with shapes based on paper (family for color, paper for marker shape).
    """
    if len(df) == 0:
        logger.warning(f"No data to plot for {title}")
        return

    fig, ax_main = plt.subplots(figsize=(10, 10))

    # --- Main scatter plot ---
    # Sort by size for layering
    df_sorted = df.sort_values('size', ascending=False)

    # Plot each point
    for _, row in df_sorted.iterrows():
        family = row['family']
        size = row['size']
        paper = row.get('paper', 'other')

        color = FAMILY_COLORS.get(family, FAMILY_COLORS['other'])
        marker = get_paper_marker(paper)
        marker_size = size_to_marker_size(size)

        ax_main.scatter(
            row['j_winrate'],
            row['weighted_k_winrate'],
            c=color,
            s=marker_size,
            marker=marker,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )

    # y=x line
    lims = [0, 1]
    ax_main.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)

    # Compute LOBF (without adding correlation stats to title)
    if len(df) >= 2:
        result = linregress(df['j_winrate'], df['weighted_k_winrate'])

        # Plot LOBF
        x_line = np.linspace(df['j_winrate'].min(), df['j_winrate'].max(), 100)
        y_line = result.slope * x_line + result.intercept
        ax_main.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.7)

    ax_main.set_xlabel("J's Gold Winrate against R", fontsize=LABEL_FONTSIZE)
    ax_main.set_ylabel("Weighted Avg K's Gold Winrate against R", fontsize=LABEL_FONTSIZE)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_aspect('equal')
    ax_main.set_title(f"{title}\n({subtitle})", fontsize=TITLE_FONTSIZE)
    ax_main.tick_params(axis='both', labelsize=TICK_FONTSIZE)

    # --- Legend ---
    # Row 1: Family colors
    families_present = df['family'].unique()
    family_handles = [
        mpatches.Patch(color=FAMILY_COLORS.get(f, FAMILY_COLORS['other']), label=f.capitalize())
        for f in sorted(families_present)
    ]

    # Row 2: Paper markers
    papers_present = df['paper'].unique() if 'paper' in df.columns else []
    paper_handles = []
    paper_labels = {
        'llm-sp-verif': 'Verifiable',
        'dbg-score-paper': 'DBG Score',
        'author_obfuscation': 'Author Obf.',
        'panickserry': 'panickserry'
    }
    for paper in sorted(papers_present):
        marker = get_paper_marker(paper)
        label = paper_labels.get(paper, paper)
        paper_handles.append(
            Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                   markersize=12, label=label, markeredgecolor='black', markeredgewidth=0.5)
        )

    # Row 3: Size legend
    sizes_present = df['size'].unique()
    size_handles = []
    size_ranges = [(3, '≤3B', 100), (7, '4-9B', 250), (14, '10-32B', 450), (70, '≥70B', 750)]
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

    # Combine legend handles
    all_handles = family_handles + paper_handles + size_handles
    ax_main.legend(handles=all_handles, loc='upper left', fontsize=LEGEND_FONTSIZE - 1, ncol=2)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ['png', 'pdf']:
        fig.savefig(output_path.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight')

    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")


def aggregate_winrate_by_judge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate winrate data by judge (across references and datasets).
    """
    rows = []

    for judge in df['judge'].unique():
        judge_df = df[df['judge'] == judge]

        # Weight by n_ilsp
        total_ilsp = judge_df['n_ilsp'].sum()
        if total_ilsp == 0:
            continue

        weighted_j = (judge_df['j_winrate'] * judge_df['n_ilsp']).sum() / total_ilsp
        weighted_k = (judge_df['weighted_k_winrate'] * judge_df['n_ilsp']).sum() / total_ilsp

        # Get most common paper for this judge
        papers = judge_df['paper'].value_counts()
        most_common_paper = papers.index[0] if len(papers) > 0 else 'other'

        rows.append({
            'judge': judge,
            'j_winrate': weighted_j,
            'weighted_k_winrate': weighted_k,
            'n_ilsp': total_ilsp,
            'n_lsp': judge_df['n_lsp'].sum(),
            'n_references': len(judge_df),
            'family': extract_model_family(judge),
            'size': extract_model_size(judge),
            'paper': most_common_paper
        })

    return pd.DataFrame(rows)


def aggregate_winrate_by_dataset_judge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate winrate data by (dataset, judge).
    """
    rows = []

    for (dataset, judge), group in df.groupby(['dataset', 'judge']):
        total_ilsp = group['n_ilsp'].sum()
        if total_ilsp == 0:
            continue

        weighted_j = (group['j_winrate'] * group['n_ilsp']).sum() / total_ilsp
        weighted_k = (group['weighted_k_winrate'] * group['n_ilsp']).sum() / total_ilsp

        rows.append({
            'dataset': dataset,
            'judge': judge,
            'j_winrate': weighted_j,
            'weighted_k_winrate': weighted_k,
            'n_ilsp': total_ilsp,
            'n_lsp': group['n_lsp'].sum(),
            'n_references': len(group),
            'family': extract_model_family(judge),
            'size': extract_model_size(judge),
            'paper': DATASET_TO_PAPER.get(dataset, 'other')
        })

    return pd.DataFrame(rows)


def generate_part2_latex_tables(
    df_triplet: pd.DataFrame,
    binned_residuals: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
):
    """Generate LaTeX tables for Part 2 results."""
    latex_dir = output_dir

    # Table 1: Summary statistics
    if len(df_triplet) >= 2:
        result = linregress(df_triplet['j_winrate'], df_triplet['weighted_k_winrate'])
        rho, _ = pearsonr(df_triplet['j_winrate'], df_triplet['weighted_k_winrate'])

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Winrate Balance Summary Statistics}",
            r"\label{tab:winrate_balance_summary}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
            f"Number of (J, R) pairs & {len(df_triplet)} \\\\",
            f"Pearson $\\rho$ & {rho:.3f} \\\\",
            f"$R^2$ & {result.rvalue**2:.3f} \\\\",
            f"Slope & {result.slope:.3f} \\\\",
            f"Intercept & {result.intercept:.3f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        latex_dir.mkdir(parents=True, exist_ok=True)
        with open(latex_dir / "winrate_balance_summary.tex", 'w') as f:
            f.write("\n".join(lines))

        logger.info(f"Saved LaTeX summary table")

    # Table 2: Binned residuals
    if len(binned_residuals) > 0:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Binned Residuals: K Winrate - J Winrate by J Winrate Decile}",
            r"\label{tab:binned_residuals}",
            r"\begin{tabular}{ccccc}",
            r"\toprule",
            r"Bin & Center & Mean Residual & 95\% CI & $N$ \\",
            r"\midrule",
        ]

        for _, row in binned_residuals.iterrows():
            bin_idx = int(row['bin_idx']) + 1
            center = row['bin_center']
            mean_r = row['mean_residual']
            ci_low = row['ci_low']
            ci_high = row['ci_high']
            n = int(row['n_points'])

            lines.append(
                f"{bin_idx} & {center:.2f} & {mean_r:+.3f} & [{ci_low:+.3f}, {ci_high:+.3f}] & {n} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(latex_dir / "binned_residuals.tex", 'w') as f:
            f.write("\n".join(lines))

        logger.info(f"Saved binned residuals LaTeX table")


def run_part2_analysis(
    cache_data: Dict,
    proxy_defs: Dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Run Part 2: Winrate balance scatter plots.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: WINRATE BALANCE SCATTER PLOTS")
    logger.info("=" * 80)

    part2_dir = output_dir / "part2_winrate_balance"
    plots_dir = part2_dir / "plots"
    csv_dir = part2_dir / "csv"
    latex_dir = part2_dir / "latex"

    # Compute base data
    df_triplet = compute_winrate_balance_data(cache_data, proxy_defs, logger)

    if len(df_triplet) == 0:
        logger.warning("No winrate balance data computed. Skipping Part 2.")
        return

    # Save triplet-level data
    csv_dir.mkdir(parents=True, exist_ok=True)
    df_triplet.to_csv(csv_dir / "winrate_balance_by_triplet.csv", index=False)

    # Plot per (dataset, judge, reference)
    plot_winrate_balance_scatter_with_residuals(
        df_triplet,
        "Gold Winrate Balance: J vs Weighted K",
        "",  # Will be filled by function
        plots_dir / "winrate_balance_by_triplet.png",
        logger
    )

    # Compute and save binned residuals
    binned_residuals = compute_binned_residuals(df_triplet)
    binned_residuals.to_csv(csv_dir / "binned_residuals.csv", index=False)

    # Aggregate by judge
    df_judge = aggregate_winrate_by_judge(df_triplet)
    df_judge.to_csv(csv_dir / "winrate_balance_by_judge.csv", index=False)

    plot_winrate_balance_scatter_with_residuals(
        df_judge,
        "Gold Winrate Balance (by judge)",
        "",
        plots_dir / "winrate_balance_by_judge.png",
        logger
    )

    # Aggregate by (dataset, judge)
    df_ds_judge = aggregate_winrate_by_dataset_judge(df_triplet)
    df_ds_judge.to_csv(csv_dir / "winrate_balance_by_dataset_judge.csv", index=False)

    plot_winrate_balance_scatter_with_residuals(
        df_ds_judge,
        "Gold Winrate Balance (by dataset, judge)",
        "",
        plots_dir / "winrate_balance_by_dataset_judge.png",
        logger
    )

    # Generate LaTeX tables
    generate_part2_latex_tables(df_triplet, binned_residuals, latex_dir, logger)

    # Compute and log correlations
    for name, df in [('triplet', df_triplet), ('judge', df_judge), ('dataset_judge', df_ds_judge)]:
        if len(df) >= 2:
            result = linregress(df['j_winrate'], df['weighted_k_winrate'])
            rho, p_val = pearsonr(df['j_winrate'], df['weighted_k_winrate'])
            logger.info(f"\n{name} level correlation:")
            logger.info(f"  ρ = {rho:.4f} (p={p_val:.4e})")
            logger.info(f"  R² = {result.rvalue**2:.4f}")
            logger.info(f"  slope = {result.slope:.4f}")
            logger.info(f"  intercept = {result.intercept:.4f}")
            logger.info(f"  n = {len(df)}")

    logger.info("Part 2 complete.")


# -------------------------
# --- PART 3: JUDGE-PROXY COMPOSITION ---
# -------------------------

def compute_judge_proxy_composition(
    cache_data: Dict,
    logger: logging.Logger
) -> Dict[str, pd.DataFrame]:
    """
    Compute judge-proxy composition tables per dataset.

    Returns:
        Dict mapping dataset -> DataFrame with rows=judges, columns=proxy families
    """
    logger.info("Computing judge-proxy composition...")

    # Group by dataset
    by_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_by_dataset_judge = defaultdict(lambda: defaultdict(int))

    for (dataset, judge, reference, ex_id), entry in cache_data.items():
        judge_family = extract_model_family(judge)

        for proxy_name in entry['k_probs'].keys():
            proxy_family = extract_model_family(proxy_name)
            by_dataset[dataset][judge][proxy_family] += 1
            total_by_dataset_judge[dataset][judge] += 1

    tables = {}

    for dataset in sorted(by_dataset.keys()):
        judges = sorted(by_dataset[dataset].keys())
        proxy_families = sorted(set(
            pf for j in judges for pf in by_dataset[dataset][j].keys()
        ))

        rows = []
        for judge in judges:
            row = {'judge': judge}
            total = total_by_dataset_judge[dataset][judge]
            row['total_n'] = total

            for pf in proxy_families:
                count = by_dataset[dataset][judge].get(pf, 0)
                pct = 100.0 * count / total if total > 0 else 0.0
                row[pf] = pct

            rows.append(row)

        tables[dataset] = pd.DataFrame(rows)

    logger.info(f"Computed composition tables for {len(tables)} datasets")
    return tables


def generate_composition_latex_tables(
    tables: Dict[str, pd.DataFrame],
    output_dir: Path,
    logger: logging.Logger
):
    """Generate LaTeX tables for judge-proxy composition."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, df in tables.items():
        if len(df) == 0:
            continue

        proxy_families = [c for c in df.columns if c not in ['judge', 'total_n']]

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{Judge-Proxy Composition: {dataset}}}",
            f"\\label{{tab:composition_{dataset}}}",
            r"\begin{tabular}{l" + "r" * (len(proxy_families) + 1) + "}",
            r"\toprule",
            "Judge & " + " & ".join([pf.capitalize() for pf in proxy_families]) + r" & $N$ \\",
            r"\midrule",
        ]

        for _, row in df.iterrows():
            judge = row['judge']
            # Truncate long judge names
            if len(judge) > 20:
                judge = judge[:17] + "..."

            parts = [judge]
            for pf in proxy_families:
                pct = row[pf]
                if pct > 0:
                    parts.append(f"{pct:.1f}\\%")
                else:
                    parts.append("--")
            parts.append(f"{int(row['total_n']):,}")

            lines.append(" & ".join(parts) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(output_dir / f"composition_{dataset}.tex", 'w') as f:
            f.write("\n".join(lines))

    logger.info(f"Saved {len(tables)} composition LaTeX tables")


def run_part3_analysis(
    cache_data: Dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Run Part 3: Judge-proxy composition tables.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PART 3: JUDGE-PROXY COMPOSITION TABLES")
    logger.info("=" * 80)

    part3_dir = output_dir / "part3_composition"
    csv_dir = part3_dir / "csv"
    latex_dir = part3_dir / "latex"

    # Compute composition tables
    tables = compute_judge_proxy_composition(cache_data, logger)

    # Save CSVs
    csv_dir.mkdir(parents=True, exist_ok=True)
    for dataset, df in tables.items():
        df.to_csv(csv_dir / f"composition_{dataset}.csv", index=False)

    # Generate LaTeX tables
    generate_composition_latex_tables(tables, latex_dir, logger)

    logger.info("Part 3 complete.")


# -------------------------
# --- PART 4: SENSITIVITY ANALYSIS (OUT-OF-FAMILY) ---
# -------------------------

def filter_out_of_family_proxies(
    cache_data: Dict[Tuple[str, str, str, str], Dict],
    logger: logging.Logger
) -> Dict[Tuple[str, str, str, str], Dict]:
    """
    Filter cache data to only include out-of-family proxies.
    """
    filtered = {}
    n_removed = 0
    n_kept = 0

    for key, entry in cache_data.items():
        dataset, judge, reference, ex_id = key
        judge_family = extract_model_family(judge)

        # Filter k_probs to only include out-of-family proxies
        filtered_k_probs = {}
        for proxy_name, prob in entry['k_probs'].items():
            proxy_family = extract_model_family(proxy_name)
            if proxy_family != judge_family:
                filtered_k_probs[proxy_name] = prob
                n_kept += 1
            else:
                n_removed += 1

        if filtered_k_probs:
            filtered[key] = {
                'j_prob': entry['j_prob'],
                'k_probs': filtered_k_probs,
                'category': entry['category']
            }

    logger.info(f"Out-of-family filtering: kept {n_kept} proxy evaluations, removed {n_removed}")
    logger.info(f"Filtered data has {len(filtered)} examples (was {len(cache_data)})")

    return filtered


def compute_sensitivity_metrics(
    cache_data_full: Dict,
    cache_data_filtered: Dict,
    logger: logging.Logger
) -> Dict:
    """
    Compute metrics comparing full vs out-of-family filtered data.
    """
    def compute_metrics(data):
        ilsp_diffs = []
        lsp_diffs = []

        for key, entry in data.items():
            j_prob = entry['j_prob']
            avg_k = np.mean(list(entry['k_probs'].values()))
            diff = j_prob - avg_k

            if entry['category'] == 'ilsp':
                ilsp_diffs.append(diff)
            elif entry['category'] == 'lsp':
                lsp_diffs.append(diff)

        return {
            'n_ilsp': len(ilsp_diffs),
            'n_lsp': len(lsp_diffs),
            'mean_ilsp_diff': np.mean(ilsp_diffs) if ilsp_diffs else 0,
            'std_ilsp_diff': np.std(ilsp_diffs, ddof=1) if len(ilsp_diffs) > 1 else 0,
            'mean_lsp_diff': np.mean(lsp_diffs) if lsp_diffs else 0,
            'std_lsp_diff': np.std(lsp_diffs, ddof=1) if len(lsp_diffs) > 1 else 0,
        }

    metrics_full = compute_metrics(cache_data_full)
    metrics_filtered = compute_metrics(cache_data_filtered)

    return {
        'full': metrics_full,
        'out_of_family': metrics_filtered,
        'diff_mean_ilsp': metrics_filtered['mean_ilsp_diff'] - metrics_full['mean_ilsp_diff'],
        'diff_mean_lsp': metrics_filtered['mean_lsp_diff'] - metrics_full['mean_lsp_diff'],
    }


def plot_sensitivity_comparison(
    cache_data_full: Dict,
    cache_data_filtered: Dict,
    output_path: Path,
    logger: logging.Logger
):
    """
    Create comparison plot for sensitivity analysis.
    """
    # Compute mean_diff by n_proxies for both
    df_full = compute_mean_diff_all_categories(cache_data_full, logger=logger)
    df_filtered = compute_mean_diff_all_categories(cache_data_filtered, logger=logger)

    if len(df_full) == 0 or len(df_filtered) == 0:
        logger.warning("Insufficient data for sensitivity comparison plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, category in enumerate(['ilsp', 'lsp', 'joint']):
        ax = axes[idx]

        full_cat = df_full[df_full['category'] == category]
        filt_cat = df_filtered[df_filtered['category'] == category]

        if len(full_cat) > 0:
            ax.errorbar(
                full_cat['n_proxies'] - 0.1, full_cat['mean_diff'],
                yerr=full_cat['std_diff'],
                fmt='o-', color='blue', alpha=0.8, capsize=3,
                markersize=8, linewidth=2, label='All proxies'
            )

        if len(filt_cat) > 0:
            ax.errorbar(
                filt_cat['n_proxies'] + 0.1, filt_cat['mean_diff'],
                yerr=filt_cat['std_diff'],
                fmt='s-', color='orange', alpha=0.8, capsize=3,
                markersize=8, linewidth=2, label='Out-of-family only'
            )

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Number of Proxies', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Mean Diff (J - K)', fontsize=LABEL_FONTSIZE)
        ax.set_title(f'{category.upper()}', fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONTSIZE - 2)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ['png', 'pdf']:
        fig.savefig(output_path.with_suffix(f'.{fmt}'), dpi=300, bbox_inches='tight')

    plt.close(fig)
    logger.info(f"Saved sensitivity comparison plot: {output_path}")


def generate_sensitivity_latex_table(
    metrics: Dict,
    output_path: Path,
    logger: logging.Logger
):
    """Generate LaTeX table for sensitivity analysis."""
    full = metrics['full']
    oof = metrics['out_of_family']

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sensitivity Analysis: All Proxies vs Out-of-Family Only}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{ILSP} & \multicolumn{2}{c}{LSP} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Condition & Mean Diff & Std & Mean Diff & Std \\",
        r"\midrule",
        f"All proxies & {full['mean_ilsp_diff']:.4f} & {full['std_ilsp_diff']:.4f} & "
        f"{full['mean_lsp_diff']:.4f} & {full['std_lsp_diff']:.4f} \\\\",
        f"Out-of-family only & {oof['mean_ilsp_diff']:.4f} & {oof['std_ilsp_diff']:.4f} & "
        f"{oof['mean_lsp_diff']:.4f} & {oof['std_lsp_diff']:.4f} \\\\",
        r"\midrule",
        f"Difference & {metrics['diff_mean_ilsp']:.4f} & -- & {metrics['diff_mean_lsp']:.4f} & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    logger.info(f"Saved sensitivity LaTeX table: {output_path}")


def generate_sensitivity_markdown(
    metrics: Dict,
    output_path: Path,
    logger: logging.Logger
):
    """Generate markdown explanation for sensitivity analysis."""
    full = metrics['full']
    oof = metrics['out_of_family']

    content = f"""# Sensitivity Analysis: Out-of-Family Proxies

## Purpose

This analysis tests whether using only out-of-family proxies (i.e., excluding proxies from
the same model family as the judge) significantly changes our findings about self-preference.

The concern is that same-family models might share similar biases, which could inflate or
deflate self-preference measurements in ways that don't generalize.

## Methodology

1. **Full data**: All available (judge, proxy) pairs
2. **Out-of-family**: Only (judge, proxy) pairs where judge and proxy are from different
   model families (e.g., Llama judge with Qwen/Gemma proxies, but not Llama proxies)

For each condition, we compute the mean difference `J - avg(K)` where:
- J = judge's self-preference (position-averaged P(self) on ILSP examples)
- K = proxy's preference for judge's response (position-averaged)

## Results

### ILSP (Illegitimate Self-Preference)

| Condition | N | Mean Diff | Std |
|-----------|---|-----------|-----|
| All proxies | {full['n_ilsp']:,} | {full['mean_ilsp_diff']:.4f} | {full['std_ilsp_diff']:.4f} |
| Out-of-family | {oof['n_ilsp']:,} | {oof['mean_ilsp_diff']:.4f} | {oof['std_ilsp_diff']:.4f} |
| **Difference** | | **{metrics['diff_mean_ilsp']:+.4f}** | |

### LSP (Legitimate Self-Preference)

| Condition | N | Mean Diff | Std |
|-----------|---|-----------|-----|
| All proxies | {full['n_lsp']:,} | {full['mean_lsp_diff']:.4f} | {full['std_lsp_diff']:.4f} |
| Out-of-family | {oof['n_lsp']:,} | {oof['mean_lsp_diff']:.4f} | {oof['std_lsp_diff']:.4f} |
| **Difference** | | **{metrics['diff_mean_lsp']:+.4f}** | |

## Interpretation

The difference in mean ILSP diff between all proxies and out-of-family only is
**{abs(metrics['diff_mean_ilsp']):.4f}** ({'larger' if metrics['diff_mean_ilsp'] > 0 else 'smaller'}
with all proxies).

This {'suggests' if abs(metrics['diff_mean_ilsp']) < 0.02 else 'indicates'} that
{'same-family proxies do not substantially affect the self-preference measurement'
if abs(metrics['diff_mean_ilsp']) < 0.02 else
'same-family proxies may have some effect on self-preference measurement'}.

## Conclusion

{'**The results are robust to proxy family composition.** Using only out-of-family proxies '
'does not substantially change the estimated self-preference effect.'
if abs(metrics['diff_mean_ilsp']) < 0.02 else
'**Family composition may affect results.** Consider this when interpreting self-preference estimates.'}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

    logger.info(f"Saved sensitivity markdown: {output_path}")


def run_part4_analysis(
    cache_data: Dict,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Run Part 4: Sensitivity analysis with out-of-family proxies.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PART 4: SENSITIVITY ANALYSIS (OUT-OF-FAMILY PROXIES)")
    logger.info("=" * 80)

    part4_dir = output_dir / "part4_sensitivity"
    plots_dir = part4_dir / "plots"
    csv_dir = part4_dir / "csv"
    latex_dir = part4_dir / "latex"

    # Filter to out-of-family proxies
    cache_data_filtered = filter_out_of_family_proxies(cache_data, logger)

    # Compute comparison metrics
    metrics = compute_sensitivity_metrics(cache_data, cache_data_filtered, logger)

    logger.info("\nSensitivity metrics:")
    logger.info(f"  Full - ILSP: mean={metrics['full']['mean_ilsp_diff']:.4f}, std={metrics['full']['std_ilsp_diff']:.4f}")
    logger.info(f"  OOF  - ILSP: mean={metrics['out_of_family']['mean_ilsp_diff']:.4f}, std={metrics['out_of_family']['std_ilsp_diff']:.4f}")
    logger.info(f"  Diff in ILSP mean: {metrics['diff_mean_ilsp']:+.4f}")

    # Save metrics
    csv_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_dir / "sensitivity_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot comparison
    plot_sensitivity_comparison(
        cache_data, cache_data_filtered,
        plots_dir / "sensitivity_comparison.png",
        logger
    )

    # Generate LaTeX table
    generate_sensitivity_latex_table(metrics, latex_dir / "sensitivity.tex", logger)

    # Generate markdown explanation
    generate_sensitivity_markdown(metrics, part4_dir / "SENSITIVITY_ANALYSIS.md", logger)

    logger.info("Part 4 complete.")


# -------------------------
# --- MAIN ---
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Proxy robustness analysis")
    parser.add_argument("--output_dir", type=str, default="proxy_robustness_analysis_2",
                       help="Output directory")
    parser.add_argument("--skip_part1", action="store_true",
                       help="Skip Part 1 (mean diff by n_proxies)")
    parser.add_argument("--skip_part2", action="store_true",
                       help="Skip Part 2 (winrate balance)")
    parser.add_argument("--skip_part3", action="store_true",
                       help="Skip Part 3 (composition tables)")
    parser.add_argument("--skip_part4", action="store_true",
                       help="Skip Part 4 (sensitivity analysis)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    # Load cache data
    logger.info("Loading cache data...")
    cache_data = load_all_cache_data(logger)

    if not cache_data:
        logger.error("No cache data loaded. Exiting.")
        return

    # Load proxy definitions
    logger.info("Loading proxy definitions...")
    proxy_defs = load_proxy_definitions(logger)

    # Run analyses
    if not args.skip_part1:
        run_part1_analysis(cache_data, output_dir, logger)

    if not args.skip_part2:
        run_part2_analysis(cache_data, proxy_defs, output_dir, logger)

    if not args.skip_part3:
        run_part3_analysis(cache_data, output_dir, logger)

    if not args.skip_part4:
        run_part4_analysis(cache_data, output_dir, logger)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
