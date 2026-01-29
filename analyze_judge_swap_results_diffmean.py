

import argparse
import json
import logging
import os
import sys
import getpass
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import bootstrap


# -------------------------
# --- CONSTANTS ---
# -------------------------

VERIFIABILITY_DATASETS = {"math500", "mbpp-plus", "mmlu"}


# -------------------------
# --- LOGGING SETUP ---
# -------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging with metadata."""
    log_dir = output_dir / "file_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analyze_judge_swap_diffdist_{timestamp}.log"
    
    logger = logging.getLogger("analyze_judge_swap_diffdist")
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
    logger.info("JUDGE SWAP ANALYSIS - MEAN-OF-DIFFERENCES VERSION")
    logger.info("=" * 80)
    logger.info(f"User: {getpass.getuser()}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    
    return logger


# -------------------------
# --- DATA LOADING ---
# -------------------------

def load_cached_preferences(cache_dir: Path, logger: logging.Logger) -> Dict:
    """
    Load all cached preference results from NEW cache structure: judge/reference/*.jsonl
    K_vs_R files contain multiple proxies keyed by (example_id, proxy_name).
    
    Returns:
        Dict mapping (judge, proxy, reference) -> {
            'J_vs_R_game1': [...],
            'J_vs_R_game2': [...],
            'K_vs_R_game1': [...],
            'K_vs_R_game2': [...]
        }
    """
    results = {}
    
    # Traverse new cache structure: judge / reference / game files
    for judge_dir in cache_dir.iterdir():
        if not judge_dir.is_dir():
            continue
        
        judge = judge_dir.name
        
        for ref_dir in judge_dir.iterdir():
            if not ref_dir.is_dir():
                continue
            
            reference = ref_dir.name
            
            # Load J vs R files (shared across all proxies)
            j_vs_r_game1_file = ref_dir / "J_vs_R_game1.jsonl"
            j_vs_r_game2_file = ref_dir / "J_vs_R_game2.jsonl"
            
            j_vs_r_game1 = []
            j_vs_r_game2 = []
            
            if j_vs_r_game1_file.exists():
                with open(j_vs_r_game1_file) as f:
                    j_vs_r_game1 = [json.loads(line) for line in f]
            
            if j_vs_r_game2_file.exists():
                with open(j_vs_r_game2_file) as f:
                    j_vs_r_game2 = [json.loads(line) for line in f]
            
            # Load K vs R files (multiple proxies per file)
            k_vs_r_game1_file = ref_dir / "K_vs_R_game1.jsonl"
            k_vs_r_game2_file = ref_dir / "K_vs_R_game2.jsonl"
            
            # Group K vs R results by proxy_name
            proxies_k_data = {}  # proxy_name -> {game1: [...], game2: [...]}
            
            if k_vs_r_game1_file.exists():
                with open(k_vs_r_game1_file) as f:
                    for line in f:
                        item = json.loads(line)
                        proxy_name = item.get('proxy', '')  # Changed from 'proxy_name' to 'proxy'
                        if proxy_name not in proxies_k_data:
                            proxies_k_data[proxy_name] = {'game1': [], 'game2': []}
                        proxies_k_data[proxy_name]['game1'].append(item)
            
            if k_vs_r_game2_file.exists():
                with open(k_vs_r_game2_file) as f:
                    for line in f:
                        item = json.loads(line)
                        proxy_name = item.get('proxy', '')  # Changed from 'proxy_name' to 'proxy'
                        if proxy_name not in proxies_k_data:
                            proxies_k_data[proxy_name] = {'game1': [], 'game2': []}
                        proxies_k_data[proxy_name]['game2'].append(item)
            
            # Create entries for each proxy
            for proxy_name, k_data in proxies_k_data.items():
                key = (judge, proxy_name, reference)
                results[key] = {
                    'J_vs_R_game1': j_vs_r_game1,
                    'J_vs_R_game2': j_vs_r_game2,
                    'K_vs_R_game1': k_data['game1'],
                    'K_vs_R_game2': k_data['game2']
                }
                logger.debug(f"Loaded {judge}/{proxy_name} vs {reference}: "
                           f"J_vs_R_g1={len(j_vs_r_game1)}, J_vs_R_g2={len(j_vs_r_game2)}, "
                           f"K_vs_R_g1={len(k_data['game1'])}, K_vs_R_g2={len(k_data['game2'])}")
    
    logger.info(f"Loaded preferences for {len(results)} (judge, proxy, reference) triplets")
    return results


def load_author_obfuscation_cache(quality_dir: Path, logger: logging.Logger) -> Dict:
    """
    Load preferences from author_obfuscation format: quality/judge/*.json
    
    Returns:
        Dict mapping (judge, proxy, reference) -> {
            'J_vs_R_game1': [...],
            'J_vs_R_game2': [...],
            'K_vs_R_game1': [...],
            'K_vs_R_game2': [...]
        }
    """
    results = {}

    dataset_name = quality_dir.name
    proxy_dir = Path(__file__).parent / "author_obfuscation" / "data" / dataset_name / "proxies"
    if not proxy_dir.exists():
        logger.warning(f"Proxy directory not found: {proxy_dir}. Will fall back to per-item 'category' when available.")

    # Cache proxy label sets keyed by (judge, reference, proxy)
    label_cache = {}  # (judge, reference, proxy) -> (lsp_set, ilsp_set)

    def _get_label_sets(judge: str, reference: str, proxy: str):
        """Get (lsp_ids, ilsp_ids) sets for a judge/reference/proxy from proxy_dir."""
        cache_key = (judge, reference, proxy)
        if cache_key in label_cache:
            return label_cache[cache_key]

        if not proxy_dir.exists():
            label_cache[cache_key] = (set(), set())
            return label_cache[cache_key]

        proxy_file = proxy_dir / f"evaluator_{judge}_vs_{reference}.json"
        if not proxy_file.exists():
            logger.warning(f"Missing proxy file for label mapping: {proxy_file}")
            label_cache[cache_key] = (set(), set())
            return label_cache[cache_key]

        try:
            with open(proxy_file) as f:
                proxy_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load proxy file {proxy_file}: {e}")
            label_cache[cache_key] = (set(), set())
            return label_cache[cache_key]

        proxy_info = (proxy_data.get("proxies") or {}).get(proxy)
        if not proxy_info:
            logger.warning(f"Proxy '{proxy}' not found in {proxy_file.name}; available={list((proxy_data.get('proxies') or {}).keys())}")
            label_cache[cache_key] = (set(), set())
            return label_cache[cache_key]

        lsp_set = set(proxy_info.get("lsp_ids", []))
        ilsp_set = set(proxy_info.get("ilsp_ids", []))
        label_cache[cache_key] = (lsp_set, ilsp_set)
        return label_cache[cache_key]
    
    # Walk quality_dir / judge / *.json
    for judge_dir in quality_dir.iterdir():
        if not judge_dir.is_dir():
            continue

        # Avoid accidentally ingesting prior analysis outputs.
        if judge_dir.name == "analysis":
            continue
        
        judge = judge_dir.name
        
        for json_file in judge_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if not isinstance(data, dict) or 'metadata' not in data:
                    logger.warning(f"Skipping invalid file: {json_file}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue
            
            metadata = data['metadata']
            judge = metadata['judge']
            proxy = metadata['proxy']
            reference = metadata['reference']

            lsp_set, ilsp_set = _get_label_sets(judge, reference, proxy)
            
            key = (judge, proxy, reference)
            
            games = {}
            for game_name in ['J_vs_R_game1', 'J_vs_R_game2', 'K_vs_R_game1', 'K_vs_R_game2']:
                items = []
                for item in data[game_name]:
                    example_id = item['example_id']
                    if example_id in lsp_set:
                        category = "lsp"
                    elif example_id in ilsp_set:
                        category = "ilsp"
                    else:
                        # Fallback: some author_obfuscation outputs include category per item
                        category = item.get('category', 'unknown')

                    new_item = {
                        'example_id': example_id,
                        'prob_response1': item['prob_A'],
                        'prob_response2': item['prob_B'],
                        'raw_probs': item['raw_probs'],
                        'normalized_sum': item['normalized_sum'],
                        'judge': judge,
                        'reference': reference,
                        'dataset': dataset_name,
                        'category': category
                    }
                    if game_name.startswith('K_vs_R'):
                        new_item['proxy'] = proxy
                    items.append(new_item)
                games[game_name] = items
            
            results[key] = games
            logger.debug(f"Loaded {judge}/{proxy} vs {reference}: "
                        f"J_vs_R_g1={len(games['J_vs_R_game1'])}, J_vs_R_g2={len(games['J_vs_R_game2'])}, "
                        f"K_vs_R_g1={len(games['K_vs_R_game1'])}, K_vs_R_g2={len(games['K_vs_R_game2'])}")
    
    logger.info(f"Loaded preferences for {len(results)} (judge, proxy, reference) triplets")
    return results


def compute_averaged_probabilities(
    game1_results: List[Dict],
    game2_results: List[Dict],
    logger: logging.Logger
) -> Tuple[List[float], List[int], List[int]]:
    """
    Compute averaged probabilities across position-swapped games.
    
    Game 1: Target=A, Reference=B -> P(Target) = prob_response1
    Game 2: Reference=A, Target=B -> P(Target) = prob_response2
    
    Average: P(Target) = (P(Target in Game1) + P(Target in Game2)) / 2
    
    Returns:
        (averaged_probs, ids, judge_correct_labels)
    """
    # Create lookup by ID
    game1_by_id = {r["example_id"]: r for r in game1_results}
    game2_by_id = {r["example_id"]: r for r in game2_results}
    
    common_ids = set(game1_by_id.keys()) & set(game2_by_id.keys())
    
    if len(common_ids) != len(game1_results):
        logger.warning(f"ID mismatch: Game1 has {len(game1_results)}, Game2 has {len(game2_results)}, Common: {len(common_ids)}")
    
    averaged_probs = []
    ids = []
    labels = []
    
    for ex_id in sorted(common_ids):
        g1 = game1_by_id[ex_id]
        g2 = game2_by_id[ex_id]
        
        # Game 1: Target is response1 (position A)
        prob_target_g1 = g1["prob_response1"]
        
        # Game 2: Target is response2 (position B)
        prob_target_g2 = g2["prob_response2"]
        
        # Average across positions
        avg_prob = (prob_target_g1 + prob_target_g2) / 2.0
        
        averaged_probs.append(avg_prob)
        ids.append(ex_id)
        labels.append(g1["category"])  # LSP/ILSP label
    
    return averaged_probs, ids, labels


def build_aligned_pairs_for_triplet(
    games: Dict,
    logger: logging.Logger,
) -> List[Tuple[int, float, float, str]]:
    """Build aligned (J_prob, K_prob) pairs for a single (judge, proxy, reference) triplet.

    Uses position-averaged probabilities for both J vs R and K vs R, then aligns examples
    by example_id. Pairs are constructed only for IDs present in both J and K.

    This is the core primitive used for verifiability-dataset aggregations:
    concatenating these pairs across proxies/references naturally upsamples J(JvR)
    whenever an example_id appears multiple times under different proxy tests.

    Args:
        games (Dict): Dict containing J_vs_R_game1/2 and K_vs_R_game1/2.
        logger (logging.Logger): Logger.

    Returns:
        List[Tuple[int, float, float, str]]: (example_id, j_prob, k_prob, label)

    Raises:
        KeyError: If required games are missing.
    """
    required_games = ["J_vs_R_game1", "J_vs_R_game2", "K_vs_R_game1", "K_vs_R_game2"]
    for g in required_games:
        if g not in games:
            raise KeyError(f"Missing required game '{g}'")

    j_probs, j_ids, j_labels = compute_averaged_probabilities(
        games["J_vs_R_game1"], games["J_vs_R_game2"], logger
    )
    k_probs, k_ids, k_labels = compute_averaged_probabilities(
        games["K_vs_R_game1"], games["K_vs_R_game2"], logger
    )

    j_by_id = {ex_id: (prob, label) for ex_id, prob, label in zip(j_ids, j_probs, j_labels)}
    pairs: List[Tuple[int, float, float, str]] = []
    for ex_id, prob_k, label_k in zip(k_ids, k_probs, k_labels):
        if ex_id not in j_by_id:
            continue
        prob_j, label_j = j_by_id[ex_id]
        if label_j != label_k:
            logger.warning(f"  Label mismatch for {ex_id}: J={label_j}, K={label_k}")
        pairs.append((ex_id, prob_j, prob_k, label_j))

    return pairs


def split_pairs_by_label(
    pairs: List[Tuple[int, float, float, str]],
    logger: logging.Logger,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Split aligned pairs into J and K probability lists for all/lsp/ilsp.

    Args:
        pairs (List[Tuple[int, float, float, str]]): (example_id, j_prob, k_prob, label)
        logger (logging.Logger): Logger.

    Returns:
        (j_probs_split, k_probs_split) dicts with keys {'lsp','ilsp','all'}.
    """
    j_probs_split = {"lsp": [], "ilsp": [], "all": []}
    k_probs_split = {"lsp": [], "ilsp": [], "all": []}

    for ex_id, prob_j, prob_k, label in pairs:
        j_probs_split["all"].append(prob_j)
        k_probs_split["all"].append(prob_k)

        if label == "lsp":
            j_probs_split["lsp"].append(prob_j)
            k_probs_split["lsp"].append(prob_k)
        elif label == "ilsp":
            j_probs_split["ilsp"].append(prob_j)
            k_probs_split["ilsp"].append(prob_k)
        else:
            logger.warning(f"  Unknown label for {ex_id}: {label}. Skipping from LSP/ILSP splits.")

    return j_probs_split, k_probs_split


def normalize_dataset_name(dataset: str) -> str:
    """Normalize dataset names so CLI aliases resolve to on-disk folder names.

    Args:
        dataset (str): Dataset name passed by the user.

    Returns:
        str: Normalized dataset name.

    Raises:
        ValueError: If dataset is empty.
    """
    if not dataset or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")

    dataset_norm = dataset.strip()
    if dataset_norm == "mbpp":
        return "mbpp-plus"
    return dataset_norm


def compute_aggregated_probs_mean_by_example(
    proxy_stats_list: List[Dict],
    all_preferences: Dict,
    logger: logging.Logger,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, int]]:
    """Aggregate (J vs R) and (K vs R) per judge/reference by averaging K across proxies per example.

    This matches the intended interpretation of "aggregated proxies" for verifiability datasets:
    for each example_id, compute mean P(K) across all proxies that contain that example.
    Then pair that with that example's P(J) (from J vs R) and run stats/plots on the aligned set.

    Args:
        proxy_stats_list (List[Dict]): Per-proxy summary stats dicts for a fixed (judge, reference).
        all_preferences (Dict): Loaded cache mapping (judge, proxy, reference) -> games.
        logger (logging.Logger): Logger.

    Returns:
        Tuple of (j_probs_split, k_probs_split, meta_counts)
            j_probs_split/k_probs_split are dicts with keys {'lsp','ilsp','all'}.
            meta_counts includes n_proxies, n_examples_total_unique, n_lsp, n_ilsp.

    Raises:
        ValueError: If proxy_stats_list is empty.
    """
    if not proxy_stats_list:
        raise ValueError("proxy_stats_list must be non-empty")

    # Use any triplet to obtain J vs R maps (same judge/reference; J files are shared).
    anchor = proxy_stats_list[0]
    anchor_key = (anchor["judge"], anchor["proxy"], anchor["reference"])
    if anchor_key not in all_preferences:
        raise ValueError(f"Missing cache for anchor triplet: {anchor_key}")

    anchor_games = all_preferences[anchor_key]
    j_probs, j_ids, j_labels = compute_averaged_probabilities(
        anchor_games["J_vs_R_game1"], anchor_games["J_vs_R_game2"], logger
    )
    j_by_id = {ex_id: (prob, label) for ex_id, prob, label in zip(j_ids, j_probs, j_labels)}

    # Accumulate K probabilities per example across proxies.
    k_probs_by_id: Dict[int, List[float]] = defaultdict(list)

    for pstats in proxy_stats_list:
        triplet_key = (pstats["judge"], pstats["proxy"], pstats["reference"])
        games = all_preferences.get(triplet_key)
        if not games:
            logger.warning(f"Missing cached games for triplet={triplet_key}; skipping from aggregation")
            continue

        k_probs, k_ids, k_labels = compute_averaged_probabilities(
            games["K_vs_R_game1"], games["K_vs_R_game2"], logger
        )
        for ex_id, prob_k, label_k in zip(k_ids, k_probs, k_labels):
            if ex_id not in j_by_id:
                continue
            # Prefer the J label as canonical; warn if mismatch.
            _, label_j = j_by_id[ex_id]
            if label_j != label_k:
                logger.warning(f"Label mismatch in aggregation for {ex_id}: J={label_j}, K={label_k}")
            k_probs_by_id[ex_id].append(prob_k)

    # Build aligned splits using mean K per example.
    j_probs_split = {"lsp": [], "ilsp": [], "all": []}
    k_probs_split = {"lsp": [], "ilsp": [], "all": []}

    n_lsp = 0
    n_ilsp = 0
    for ex_id in sorted(k_probs_by_id.keys()):
        prob_j, label = j_by_id[ex_id]
        prob_k_mean = float(np.mean(k_probs_by_id[ex_id]))

        j_probs_split["all"].append(prob_j)
        k_probs_split["all"].append(prob_k_mean)

        if label == "lsp":
            n_lsp += 1
            j_probs_split["lsp"].append(prob_j)
            k_probs_split["lsp"].append(prob_k_mean)
        elif label == "ilsp":
            n_ilsp += 1
            j_probs_split["ilsp"].append(prob_j)
            k_probs_split["ilsp"].append(prob_k_mean)
        else:
            logger.warning(f"Unknown label in aggregation for {ex_id}: {label}. Skipping from LSP/ILSP splits.")

    meta_counts = {
        "n_proxies": len(proxy_stats_list),
        "n_examples_total_unique": len(k_probs_by_id),
        "n_lsp": n_lsp,
        "n_ilsp": n_ilsp,
    }
    return j_probs_split, k_probs_split, meta_counts


def compute_differences_from_aligned_splits(
    j_probs_split: Dict[str, List[float]],
    k_probs_split: Dict[str, List[float]],
    logger: logging.Logger,
) -> Dict[str, List[float]]:
    """Compute per-example differences (J - K) for aligned splits.

    Args:
        j_probs_split (Dict[str, List[float]]): Aligned J probabilities (keys: all/lsp/ilsp).
        k_probs_split (Dict[str, List[float]]): Aligned K probabilities (keys: all/lsp/ilsp).
        logger (logging.Logger): Logger.

    Returns:
        Dict[str, List[float]]: Differences per split.

    Raises:
        ValueError: If a split has mismatched lengths.
    """
    diffs: Dict[str, List[float]] = {}
    for key in ("all", "lsp", "ilsp"):
        j_list = j_probs_split.get(key, [])
        k_list = k_probs_split.get(key, [])
        if len(j_list) != len(k_list):
            raise ValueError(f"Mismatched aligned split lengths for '{key}': J={len(j_list)} K={len(k_list)}")
        diffs[key] = [float(j - k) for j, k in zip(j_list, k_list)]

    if len(diffs.get("all", [])) == 0:
        logger.warning("No aligned examples to compute differences.")
    return diffs


# -------------------------
# --- VISUALIZATION ---
# -------------------------

def plot_histogram_with_stats(ax, data, objective, title=None, xlabel=None, 
                               color='cornflowerblue', no_spine=False, **hist_kwargs):
    """Plot histogram with mean, confidence interval, and objective score."""
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return
    
    mean = np.mean(data)
    CI = None
    if len(data) >= 2:
        try:
            res = bootstrap((np.array(data),), np.mean, random_state=42)
            CI = res.confidence_interval
        except Exception:
            CI = None
    
    ax.hist(data, color=color, edgecolor='white', linewidth=0.5, **hist_kwargs)
    ax.axvline(mean, color='black', linestyle='--', linewidth=2, label='μ')
    ax.axvline(objective, color='red', linestyle='--', linewidth=2, label='Objective')
    if CI is not None:
        ax.axvspan(CI.low, CI.high, alpha=0.2, color='orange', label='95% CI')
    
    ax.set_xlim(0, 1)

    if title:
        ax.set_title(title, fontweight='bold', fontsize=11, pad=15)
        subtitle = f"μ: {mean:.3f} | Obj: {objective:.3f} | n={len(data)}"
        ax.text(0.5, 0.98, subtitle, ha='center', va='top', transform=ax.transAxes,
                fontsize=8, color='gray')
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    
    if no_spine:
        ax.spines['left'].set_visible(False)
    
    if hist_kwargs.get('density', False):
        ax.set_ylabel('Density', fontsize=9)
    else:
        ax.set_ylabel('Count', fontsize=9)
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=8)


def create_difference_distribution_plot(
    diffs: List[float],
    judge: str,
    label: str,
    plot_file: Path,
    logger: logging.Logger,
) -> None:
    """Create a single-line distribution plot over per-example differences (J - K).

    Args:
        diffs (List[float]): Per-example (J - K) differences.
        judge (str): Judge model name.
        label (str): Label for the aggregation regime.
        plot_file (Path): Output path.
        logger (logging.Logger): Logger.

    Returns:
        None
    """
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    if not diffs:
        logger.warning(f"No diffs to plot for judge={judge} ({label})")
        return

    arr = np.array(diffs, dtype=float)
    mean = float(np.mean(arr))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(arr, bins=60, density=True, histtype="step", linewidth=2.0, color="black")
    ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"mean={mean:.4f}")
    ax.set_title(f"{judge}: Distribution of per-example (J - K) differences ({label})")
    ax.set_xlabel("J - K")
    ax.set_ylabel("Density")
    ax.set_xlim(-1.0, 1.0)
    ax.legend(loc="upper left")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=200)
    plt.close(fig)
    logger.info(f"  Saved difference-distribution plot to {plot_file}")


def create_comparison_plot(
    j_vs_r_probs: Dict,
    k_vs_r_probs: Dict,
    judge: str,
    proxy: str,
    reference: str,
    output_file: Path,
    logger: logging.Logger
):
    """
    Create 2x3 plot grid comparing J(J vs R) and J(K vs R).
    
    Rows: J vs R (top), K vs R (bottom)
    Columns: ILSP, LSP, Combined
    """
    logger.info(f"Creating comparison plot for Judge={judge}, Proxy={proxy}, Ref={reference}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 150,
    })
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharey='row')
    
    # Super title
    fig.suptitle(f"Judge Swap Null Test: Judge={judge}, Proxy={proxy}, Reference={reference}",
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Row 1: J(J vs R)
    plot_histogram_with_stats(
        ax=axes[0, 0],
        data=j_vs_r_probs['ilsp'],
        objective=0.0,
        title="J(J vs R) - ILSP",
        xlabel="P(J)",
        color='red',
        density=True,
        bins=20,
        alpha=0.7
    )
    
    plot_histogram_with_stats(
        ax=axes[0, 1],
        data=j_vs_r_probs['lsp'],
        objective=1.0,
        title="J(J vs R) - LSP",
        xlabel="P(J)",
        color='green',
        density=True,
        bins=20,
        alpha=0.7,
        no_spine=True
    )
    
    lsp_prop_j = len(j_vs_r_probs['lsp']) / (len(j_vs_r_probs['lsp']) + len(j_vs_r_probs['ilsp']))
    plot_histogram_with_stats(
        ax=axes[0, 2],
        data=j_vs_r_probs['all'],
        objective=lsp_prop_j,
        title="J(J vs R) - Combined",
        xlabel="P(J)",
        color='cornflowerblue',
        density=True,
        bins=20,
        alpha=0.7,
        no_spine=True
    )
    
    # Row 2: J(K vs R)
    plot_histogram_with_stats(
        ax=axes[1, 0],
        data=k_vs_r_probs['ilsp'],
        objective=0.0,
        title="J(K vs R) - ILSP",
        xlabel="P(K)",
        color='red',
        density=True,
        bins=20,
        alpha=0.7
    )
    
    plot_histogram_with_stats(
        ax=axes[1, 1],
        data=k_vs_r_probs['lsp'],
        objective=1.0,
        title="J(K vs R) - LSP",
        xlabel="P(K)",
        color='green',
        density=True,
        bins=20,
        alpha=0.7,
        no_spine=True
    )
    
    lsp_prop_k = len(k_vs_r_probs['lsp']) / (len(k_vs_r_probs['lsp']) + len(k_vs_r_probs['ilsp']))
    plot_histogram_with_stats(
        ax=axes[1, 2],
        data=k_vs_r_probs['all'],
        objective=lsp_prop_k,
        title="J(K vs R) - Combined",
        xlabel="P(K)",
        color='cornflowerblue',
        density=True,
        bins=20,
        alpha=0.7,
        no_spine=True
    )
    
    # Add legend to last plot
    axes[1, 2].legend(loc='upper left', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    logger.info(f"Saved plot to {output_file}")
    plt.close()


# -------------------------
# --- STATISTICS ---
# -------------------------

def compute_hypothesis_tests(
    j_vs_r_probs: List[float],
    k_vs_r_probs: List[float],
    logger: logging.Logger
) -> Dict:
    """
    Compute hypothesis tests using the distribution of paired differences.

    This treats inputs as aligned per-example samples and computes diffs = (J - K).
    Primary statistic is mean(diffs) with a one-sample t-test of H0: mean(diffs)=0.
    
    Returns:
        Dict with test statistics (using same variable names as original for compatibility)
    """
    stats_dict = {}

    if len(j_vs_r_probs) == 0 or len(k_vs_r_probs) == 0:
        logger.warning("Hypothesis tests skipped: empty input arrays")
        stats_dict.update({
            "mean_j_vs_r": float("nan"),
            "std_j_vs_r": float("nan"),
            "mean_k_vs_r": float("nan"),
            "std_k_vs_r": float("nan"),
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "ttest_two_sided_t": float("nan"),
            "ttest_two_sided_p": float("nan"),
            "ttest_one_sided_t": float("nan"),
            "ttest_one_sided_p": float("nan"),
            "ks_statistic": float("nan"),
            "ks_pvalue": float("nan"),
            "correlation": float("nan"),
            "correlation_pvalue": float("nan"),
            "cohens_d": float("nan"),
        })
        return stats_dict
    
    # Basic statistics
    stats_dict["mean_j_vs_r"] = np.mean(j_vs_r_probs)
    stats_dict["std_j_vs_r"] = np.std(j_vs_r_probs, ddof=1) if len(j_vs_r_probs) >= 2 else 0.0
    stats_dict["mean_k_vs_r"] = np.mean(k_vs_r_probs)
    stats_dict["std_k_vs_r"] = np.std(k_vs_r_probs, ddof=1) if len(k_vs_r_probs) >= 2 else 0.0
    
    # Compute paired differences (truncate defensively if lengths differ)
    n_j = len(j_vs_r_probs)
    n_k = len(k_vs_r_probs)
    if n_j != n_k:
        n = min(n_j, n_k)
        logger.warning(f"Length mismatch for paired diffs: J={n_j}, K={n_k}. Truncating to n={n}.")
        j_aligned = j_vs_r_probs[:n]
        k_aligned = k_vs_r_probs[:n]
    else:
        j_aligned = j_vs_r_probs
        k_aligned = k_vs_r_probs
    diffs = np.array([float(j - k) for j, k in zip(j_aligned, k_aligned)], dtype=float)

    stats_dict["mean_diff"] = float(np.mean(diffs))
    stats_dict["std_diff"] = float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else 0.0
    stats_dict["se_diff"] = float(stats_dict["std_diff"] / np.sqrt(len(diffs))) if len(diffs) > 0 else float("nan")

    # One-sample t-tests on diffs
    if len(diffs) >= 2:
        try:
            t_stat_two, p_two = stats.ttest_1samp(diffs, 0.0)
            stats_dict["ttest_two_sided_t"] = float(t_stat_two)
            stats_dict["ttest_two_sided_p"] = float(p_two)
        except Exception:
            stats_dict["ttest_two_sided_t"] = float("nan")
            stats_dict["ttest_two_sided_p"] = float("nan")

        try:
            t_stat_one, p_one = stats.ttest_1samp(diffs, 0.0, alternative='greater')
            stats_dict["ttest_one_sided_t"] = float(t_stat_one)
            stats_dict["ttest_one_sided_p"] = float(p_one)
        except TypeError:
            # Older SciPy: approximate one-sided p from two-sided
            t_stat_two = stats_dict.get("ttest_two_sided_t", float("nan"))
            p_two = stats_dict.get("ttest_two_sided_p", float("nan"))
            stats_dict["ttest_one_sided_t"] = float(t_stat_two)
            if np.isnan(p_two) or np.isnan(t_stat_two):
                stats_dict["ttest_one_sided_p"] = float("nan")
            else:
                stats_dict["ttest_one_sided_p"] = float(p_two / 2.0) if t_stat_two > 0 else float(1.0 - (p_two / 2.0))
        except Exception:
            stats_dict["ttest_one_sided_t"] = float("nan")
            stats_dict["ttest_one_sided_p"] = float("nan")
    else:
        stats_dict["ttest_two_sided_t"] = float("nan")
        stats_dict["ttest_two_sided_p"] = float("nan")
        stats_dict["ttest_one_sided_t"] = float("nan")
        stats_dict["ttest_one_sided_p"] = float("nan")
    
    # KS test for distribution similarity
    try:
        ks_stat, ks_p = stats.ks_2samp(j_vs_r_probs, k_vs_r_probs)
        stats_dict["ks_statistic"] = ks_stat
        stats_dict["ks_pvalue"] = ks_p
    except Exception:
        stats_dict["ks_statistic"] = float("nan")
        stats_dict["ks_pvalue"] = float("nan")
    
    # Pearson correlation (not meaningful here; keep nan for compatibility)
    stats_dict["correlation"] = float("nan")
    stats_dict["correlation_pvalue"] = float("nan")

    # Effect size (Cohen's d on paired differences)
    stats_dict["cohens_d"] = float(stats_dict["mean_diff"] / stats_dict["std_diff"]) if stats_dict["std_diff"] > 0 else 0.0
    
    logger.info("Hypothesis Test Results (MEAN-OF-DIFFERENCES):")
    logger.info(f"  Mean(J vs R): {stats_dict['mean_j_vs_r']:.4f} ± {stats_dict['std_j_vs_r']:.4f} (n={n_j})")
    logger.info(f"  Mean(K vs R): {stats_dict['mean_k_vs_r']:.4f} ± {stats_dict['std_k_vs_r']:.4f} (n={n_k})")
    logger.info(f"  Mean(diffs=J-K): {stats_dict['mean_diff']:.4f} ± {stats_dict['std_diff']:.4f} (std), se={stats_dict.get('se_diff', float('nan')):.4f}")
    logger.info(f"  Two-sided t-test on diffs: t={stats_dict['ttest_two_sided_t']:.3f}, p={stats_dict['ttest_two_sided_p']:.6f}")
    logger.info(f"  One-sided t-test on diffs: t={stats_dict['ttest_one_sided_t']:.3f}, p={stats_dict['ttest_one_sided_p']:.6f}")
    logger.info(f"  KS test: D={stats_dict['ks_statistic']:.3f}, p={stats_dict['ks_pvalue']:.6f}")
    logger.info(f"  Cohen's d (diffs): {stats_dict['cohens_d']:.3f}")
    
    return stats_dict


# -------------------------
# --- MAIN EXECUTION ---
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze judge swap null hypothesis test results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing cached preference results")
    parser.add_argument("--dataset", type=str, default="alpaca_eval",
                       help="Dataset name")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (defaults to results_dir/analysis)")
    args = parser.parse_args()
    
    # Setup
    dataset = normalize_dataset_name(args.dataset)
    results_dir = Path(args.results_dir) / dataset
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return
    
    # Output to jsd_{results_dir} instead of results_dir/analysis (avoid overwriting older jsn_* runs)
    base_results_dir = Path(args.results_dir).name
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"jsd_{base_results_dir}") / dataset / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset: {dataset}")

    is_verifiability_dataset = dataset in VERIFIABILITY_DATASETS
    if is_verifiability_dataset:
        logger.info("Verifiability dataset mode enabled: will compute per-proxy + aggregated stats, but plot aggregated only.")
    
    # Load cached preferences
    cache_dir = results_dir / "cache"
    if cache_dir.exists():
        all_preferences = load_cached_preferences(cache_dir, logger)
    else:
        # Try author_obfuscation format
        all_preferences = load_author_obfuscation_cache(results_dir, logger)
    
    if not all_preferences:
        logger.error("No preferences loaded. Exiting.")
        return
    
    # Process each (judge, proxy, reference) triplet
    summary_stats = []
    
    for (judge, proxy, reference), games in all_preferences.items():
        logger.info("=" * 80)
        logger.info(f"Analyzing: Judge={judge}, Proxy={proxy}, Reference={reference}")
        logger.info("=" * 80)
        
        # Check that all games are present
        required_games = ["J_vs_R_game1", "J_vs_R_game2", "K_vs_R_game1", "K_vs_R_game2"]
        if not all(g in games for g in required_games):
            logger.warning(f"Missing games for {judge}/{proxy}/{reference}. Skipping.")
            continue
        
        # Build aligned pairs (one per K example that can be matched to J)
        pairs = build_aligned_pairs_for_triplet(games, logger)
        j_probs_split, k_probs_split = split_pairs_by_label(pairs, logger)

        logger.info(f"  Aligned pairs (K matched to J): {len(pairs)}")
        logger.info(f"  Pairs - LSP: {len(j_probs_split['lsp'])}, ILSP: {len(j_probs_split['ilsp'])}")
        
        # Plots
        # - Default datasets: plot each (judge, proxy, reference)
        # - Verifiability datasets: skip per-proxy plots; plot aggregated-only later
        if not is_verifiability_dataset:
            if len(pairs) > 0:
                plot_file = output_dir / "plots" / f"{judge}_{proxy}_vs_{reference}.png"
                create_comparison_plot(
                    j_probs_split, k_probs_split, judge, proxy, reference, plot_file, logger
                )
            else:
                logger.warning(f"  Skipping plot - no aligned data for {judge} vs {proxy} vs {reference}")
        
        # Baseline statistics: full distribution + LSP/ILSP splits
        baseline_tests = {
            "all": compute_hypothesis_tests(j_probs_split["all"], k_probs_split["all"], logger),
            "ilsp": compute_hypothesis_tests(j_probs_split["ilsp"], k_probs_split["ilsp"], logger),
            "lsp": compute_hypothesis_tests(j_probs_split["lsp"], k_probs_split["lsp"], logger),
        }

        baseline_record = {
            "judge": judge,
            "proxy": proxy,
            "reference": reference,
            "n_pairs_total": len(j_probs_split["all"]),
            "n_lsp": len(j_probs_split["lsp"]),
            "n_ilsp": len(j_probs_split["ilsp"]),
            "tests": baseline_tests,
        }
        summary_stats.append(baseline_record)

        # Save individual stats
        stats_file = output_dir / "stats" / f"{judge}_{proxy}_vs_{reference}.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(baseline_record, f, indent=2)
        logger.info(f"  Saved statistics to {stats_file}")
    
    # Save summary
    summary_file = output_dir / "summary_statistics.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"\nSaved summary statistics to {summary_file}")
    
    # Group by (judge, reference) to analyze per-proxy and aggregated statistics
    logger.info("\n" + "=" * 80)
    logger.info("PER-JUDGE-REFERENCE AGGREGATION")
    logger.info("=" * 80)
    
    from collections import defaultdict
    by_judge_ref = defaultdict(list)
    for stats in summary_stats:
        key = (stats['judge'], stats['reference'])
        by_judge_ref[key].append(stats)
    
    aggregated_stats = []
    aggregated_stats_downsampled = []
    
    for (judge, reference), proxy_stats_list in by_judge_ref.items():
        logger.info(f"\nJudge={judge}, Reference={reference}")
        logger.info(f"  Number of proxies: {len(proxy_stats_list)}")
        
        # Log per-proxy statistics
        logger.info("  Per-proxy statistics:")
        def _safe_p(rec: Dict, subset: str) -> float:
            try:
                p = (rec.get("tests") or {}).get(subset, {}).get("ttest_one_sided_p")
                return float(p) if p is not None else float("nan")
            except Exception:
                return float("nan")

        for rec in sorted(proxy_stats_list, key=lambda x: _safe_p(x, "ilsp")):
            proxy = rec["proxy"]
            n_ilsp = rec.get("n_ilsp", 0)
            n_all = rec.get("n_pairs_total", 0)
            p_one_ilsp = _safe_p(rec, "ilsp")
            p_one_all = _safe_p(rec, "all")
            logger.info(f"    {proxy}: n_all={n_all}, n_ilsp={n_ilsp}, p_one_all={p_one_all:.4f}, p_one_ilsp={p_one_ilsp:.4f}")

        # Find most robust proxy (highest ILSP one-sided p-value, ignoring NaNs)
        finite = [rec for rec in proxy_stats_list if not np.isnan(_safe_p(rec, "ilsp"))]
        if finite:
            most_robust = max(finite, key=lambda x: _safe_p(x, "ilsp"))
            most_robust_p = _safe_p(most_robust, "ilsp")
        else:
            most_robust = proxy_stats_list[0]
            most_robust_p = float("nan")
        logger.info(f"  Most robust proxy (highest ILSP p-value): {most_robust['proxy']} (p={most_robust_p:.4f})")
        
        # Aggregation methods
        # - Default: concatenate ILSP examples across proxies
        # - Verifiability datasets:
        #   (a) UPSAMPLE: concatenate aligned (J,K) pairs across proxies (duplicates J when an example appears in many proxies)
        #   (b) DOWNSAMPLE: index by J example_id and average K across all proxies per example
        if is_verifiability_dataset:
            all_pairs: List[Tuple[int, float, float, str]] = []
            for rec in proxy_stats_list:
                triplet_key = (rec["judge"], rec["proxy"], rec["reference"])
                games = all_preferences.get(triplet_key)
                if not games:
                    continue
                all_pairs.extend(build_aligned_pairs_for_triplet(games, logger))

            j_agg, k_agg = split_pairs_by_label(all_pairs, logger)
            logger.info(f"  Aggregated pairs across proxies (UPSAMPLE): {len(all_pairs)} (ILSP={len(j_agg['ilsp'])}, LSP={len(j_agg['lsp'])})")

            agg_tests = {
                "all": compute_hypothesis_tests(j_agg["all"], k_agg["all"], logger),
                "ilsp": compute_hypothesis_tests(j_agg["ilsp"], k_agg["ilsp"], logger),
                "lsp": compute_hypothesis_tests(j_agg["lsp"], k_agg["lsp"], logger),
            }

            # Plot aggregated only
            if len(all_pairs) > 0:
                plot_file = output_dir / "plots_aggregated" / f"{judge}_AGG_ALL_PROXIES_UPSAMPLEJ_vs_{reference}.png"
                create_comparison_plot(
                    j_agg,
                    k_agg,
                    judge,
                    f"AGG_ALL_PROXIES_UPSAMPLEJ(n={len(proxy_stats_list)})",
                    reference,
                    plot_file,
                    logger,
                )
            else:
                logger.warning(f"  Skipping aggregated UPSAMPLE plot - no aligned data for Judge={judge}, Reference={reference}")

            aggregated_stats.append({
                "judge": judge,
                "reference": reference,
                "aggregation_method": "concat_pairs_over_proxies_upsample_j",
                "n_proxies": len(proxy_stats_list),
                "n_pairs_total": len(j_agg["all"]),
                "n_lsp": len(j_agg["lsp"]),
                "n_ilsp": len(j_agg["ilsp"]),
                "most_robust_proxy": most_robust["proxy"],
                "most_robust_p_ilsp": most_robust_p,
                "tests": agg_tests,
            })

            # DOWN-SAMPLED: index by J example_id and average K across proxies per example.
            try:
                j_ds, k_ds, meta = compute_aggregated_probs_mean_by_example(proxy_stats_list, all_preferences, logger)
            except Exception as e:
                logger.warning(f"  Downsampled aggregation failed for Judge={judge}, Reference={reference}: {e}")
                continue

            logger.info(
                f"  Aggregated examples across proxies (DOWNSAMPLE): {meta['n_examples_total_unique']} "
                f"(ILSP={meta['n_ilsp']}, LSP={meta['n_lsp']})"
            )

            ds_tests = {
                "all": compute_hypothesis_tests(j_ds["all"], k_ds["all"], logger),
                "ilsp": compute_hypothesis_tests(j_ds["ilsp"], k_ds["ilsp"], logger),
                "lsp": compute_hypothesis_tests(j_ds["lsp"], k_ds["lsp"], logger),
            }

            if meta["n_examples_total_unique"] > 0:
                plot_file = output_dir / "plots_aggregated_downsampled" / f"{judge}_AGG_ALL_PROXIES_DOWNSAMPLED_vs_{reference}.png"
                create_comparison_plot(
                    j_ds,
                    k_ds,
                    judge,
                    f"AGG_ALL_PROXIES_DOWNSAMPLED(n={meta['n_proxies']})",
                    reference,
                    plot_file,
                    logger,
                )
            else:
                logger.warning(f"  Skipping aggregated DOWNSAMPLED plot - no aligned data for Judge={judge}, Reference={reference}")

            aggregated_stats_downsampled.append({
                "judge": judge,
                "reference": reference,
                "aggregation_method": "mean_k_over_proxies_by_j_example_id",
                **meta,
                "most_robust_proxy": most_robust["proxy"],
                "most_robust_p_ilsp": most_robust_p,
                "tests": ds_tests,
            })
        else:
            # Backward-compatible behavior: concatenate ILSP examples across proxies.
            all_j_ilsp = []
            all_k_ilsp = []

            for pstats in proxy_stats_list:
                triplet_key = (pstats['judge'], pstats['proxy'], pstats['reference'])
                if triplet_key in all_preferences:
                    games = all_preferences[triplet_key]
                    j_probs, j_ids, j_labels = compute_averaged_probabilities(
                        games["J_vs_R_game1"], games["J_vs_R_game2"], logger
                    )
                    k_probs, k_ids, k_labels = compute_averaged_probabilities(
                        games["K_vs_R_game1"], games["K_vs_R_game2"], logger
                    )

                    j_ids_set = set(j_ids)
                    k_ids_set = set(k_ids)
                    common_ids = j_ids_set & k_ids_set

                    j_by_id = {ex_id: (prob, label) for ex_id, prob, label in zip(j_ids, j_probs, j_labels)}
                    k_by_id = {ex_id: (prob, label) for ex_id, prob, label in zip(k_ids, k_probs, k_labels)}

                    for ex_id in common_ids:
                        j_prob, j_label = j_by_id[ex_id]
                        k_prob, _ = k_by_id[ex_id]
                        if j_label == "ilsp":
                            all_j_ilsp.append(j_prob)
                            all_k_ilsp.append(k_prob)

            logger.info(f"  Aggregated ILSP examples across all proxies: {len(all_j_ilsp)}")

            if len(all_j_ilsp) > 0 and len(all_k_ilsp) > 0:
                agg_stats = compute_hypothesis_tests(all_j_ilsp, all_k_ilsp, logger)

                # Aggregated plot for DBG/author-obf style datasets (ILSP-only concat).
                j_plot = {"ilsp": all_j_ilsp, "lsp": [], "all": all_j_ilsp}
                k_plot = {"ilsp": all_k_ilsp, "lsp": [], "all": all_k_ilsp}
                plot_file = output_dir / "plots_aggregated" / f"{judge}_AGG_ILSP_ALL_PROXIES_vs_{reference}.png"
                create_comparison_plot(
                    j_plot,
                    k_plot,
                    judge,
                    f"AGG_ILSP_ALL_PROXIES(n={len(proxy_stats_list)})",
                    reference,
                    plot_file,
                    logger,
                )

                aggregated_stats.append({
                    'judge': judge,
                    'reference': reference,
                    'aggregation_method': 'concat_ilsp',
                    'n_proxies': len(proxy_stats_list),
                    'n_ilsp_total': len(all_j_ilsp),
                    'most_robust_proxy': most_robust['proxy'],
                    'most_robust_p_ilsp': most_robust_p,
                    **agg_stats
                })
    
    # Save aggregated statistics
    agg_file = output_dir / "aggregated_by_judge_reference.json"
    with open(agg_file, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    logger.info(f"\nSaved aggregated statistics to {agg_file}")

    if is_verifiability_dataset:
        agg_ds_file = output_dir / "aggregated_by_judge_reference_downsampled.json"
        with open(agg_ds_file, "w") as f:
            json.dump(aggregated_stats_downsampled, f, indent=2)
        logger.info(f"Saved DOWN-SAMPLED aggregated statistics to {agg_ds_file}")

    # -----------------------------
    # Aggregation over references
    # -----------------------------
    logger.info("\n" + "=" * 80)
    logger.info("PER-JUDGE AGGREGATION (ALL REFERENCES + ALL PROXIES)")
    logger.info("=" * 80)

    by_judge = defaultdict(list)
    for rec in summary_stats:
        by_judge[rec["judge"]].append(rec)

    aggregated_by_judge = []
    aggregated_by_judge_downsampled = []

    for judge, triplet_records in by_judge.items():
        logger.info(f"\nJudge={judge}")
        logger.info(f"  Triplets: {len(triplet_records)}")

        all_pairs: List[Tuple[int, float, float, str]] = []
        for rec in triplet_records:
            triplet_key = (rec["judge"], rec["proxy"], rec["reference"])
            games = all_preferences.get(triplet_key)
            if not games:
                continue
            all_pairs.extend(build_aligned_pairs_for_triplet(games, logger))

        j_split, k_split = split_pairs_by_label(all_pairs, logger)
        logger.info(f"  Aggregated pairs: {len(all_pairs)} (ILSP={len(j_split['ilsp'])}, LSP={len(j_split['lsp'])})")

        judge_tests = {
            "all": compute_hypothesis_tests(j_split["all"], k_split["all"], logger),
            "ilsp": compute_hypothesis_tests(j_split["ilsp"], k_split["ilsp"], logger),
            "lsp": compute_hypothesis_tests(j_split["lsp"], k_split["lsp"], logger),
        }

        aggregated_by_judge.append({
            "judge": judge,
            "aggregation_method": "concat_pairs_over_refs_and_proxies_upsample_j",
            "n_triplets": len(triplet_records),
            "n_pairs_total": len(j_split["all"]),
            "n_lsp": len(j_split["lsp"]),
            "n_ilsp": len(j_split["ilsp"]),
            "tests": judge_tests,
        })

        # Plot aggregated over references for all datasets.
        if len(all_pairs) > 0:
            plot_file = output_dir / "plots_aggregated_all_refs" / f"{judge}_AGG_ALL_REFS_ALL_PROXIES.png"
            create_comparison_plot(
                j_split,
                k_split,
                judge,
                "AGG_ALL_REFS_ALL_PROXIES",
                "ALL_REFERENCES",
                plot_file,
                logger,
            )

            # For non-verifiability datasets, also emit an upsampled (concat-pairs) diff distribution.
            if not is_verifiability_dataset:
                diffs = compute_differences_from_aligned_splits(j_split, k_split, logger)
                diff_plot = output_dir / "plots_aggregated_all_refs_diffdist" / f"{judge}_AGG_ALL_REFS_ALL_PROXIES_UPSAMPLEJ_DIFFDIST.png"
                create_difference_distribution_plot(
                    diffs["all"],
                    judge,
                    "AGG_ALL_REFS_ALL_PROXIES_UPSAMPLEJ",
                    diff_plot,
                    logger,
                )

        # DOWN-SAMPLED aggregation across references: index by (reference, example_id) and average K over proxies per example.
        if is_verifiability_dataset:
            by_ref_local = defaultdict(list)
            for rec in triplet_records:
                by_ref_local[rec["reference"]].append(rec)

            j_all_ds = {"lsp": [], "ilsp": [], "all": []}
            k_all_ds = {"lsp": [], "ilsp": [], "all": []}
            diff_all_ds: List[float] = []
            n_refs_used = 0

            for reference, proxy_recs in by_ref_local.items():
                try:
                    j_ds, k_ds, meta = compute_aggregated_probs_mean_by_example(proxy_recs, all_preferences, logger)
                except Exception as e:
                    logger.warning(f"  Downsampled aggregation failed for Judge={judge}, Reference={reference}: {e}")
                    continue

                if meta.get("n_examples_total_unique", 0) == 0:
                    continue

                n_refs_used += 1
                for key in ("all", "lsp", "ilsp"):
                    j_all_ds[key].extend(j_ds[key])
                    k_all_ds[key].extend(k_ds[key])

                diffs = compute_differences_from_aligned_splits(j_ds, k_ds, logger)
                diff_all_ds.extend(diffs["all"])

            logger.info(
                f"  DOWN-SAMPLED (all refs): aligned examples={len(j_all_ds['all'])} "
                f"(ILSP={len(j_all_ds['ilsp'])}, LSP={len(j_all_ds['lsp'])}, refs_used={n_refs_used})"
            )

            if len(j_all_ds["all"]) > 0:
                judge_tests_ds = {
                    "all": compute_hypothesis_tests(j_all_ds["all"], k_all_ds["all"], logger),
                    "ilsp": compute_hypothesis_tests(j_all_ds["ilsp"], k_all_ds["ilsp"], logger),
                    "lsp": compute_hypothesis_tests(j_all_ds["lsp"], k_all_ds["lsp"], logger),
                }

                aggregated_by_judge_downsampled.append({
                    "judge": judge,
                    "aggregation_method": "mean_k_over_proxies_by_j_example_id_per_reference_then_concat_refs",
                    "n_triplets": len(triplet_records),
                    "n_refs_used": n_refs_used,
                    "n_pairs_total": len(j_all_ds["all"]),
                    "n_lsp": len(j_all_ds["lsp"]),
                    "n_ilsp": len(j_all_ds["ilsp"]),
                    "tests": judge_tests_ds,
                })

                # Alternate plot requested: single-line distribution over per-example differences (J-K)
                diff_plot = output_dir / "plots_aggregated_all_refs_diffdist" / f"{judge}_AGG_ALL_REFS_ALL_PROXIES_DOWNSAMPLED_DIFFDIST.png"
                create_difference_distribution_plot(
                    diff_all_ds,
                    judge,
                    "AGG_ALL_REFS_ALL_PROXIES_DOWNSAMPLED",
                    diff_plot,
                    logger,
                )

    agg_judge_file = output_dir / "aggregated_by_judge_all_references.json"
    with open(agg_judge_file, "w") as f:
        json.dump(aggregated_by_judge, f, indent=2)
    logger.info(f"\nSaved aggregated judge-level statistics to {agg_judge_file}")

    if is_verifiability_dataset:
        agg_judge_ds_file = output_dir / "aggregated_by_judge_all_references_downsampled.json"
        with open(agg_judge_ds_file, "w") as f:
            json.dump(aggregated_by_judge_downsampled, f, indent=2)
        logger.info(f"Saved DOWN-SAMPLED aggregated judge-level statistics to {agg_judge_ds_file}")
    
    # Count rejections (track both all and ilsp for verifiability datasets)
    alpha_two = 0.05
    alpha_one = 0.05

    def _count_rejections(records: List[Dict], subset: str, p_key: str, alpha: float) -> int:
        count = 0
        for rec in records:
            tests = rec.get("tests") or {}
            sub = tests.get(subset) or {}
            p_val = sub.get(p_key)
            if p_val is None:
                continue
            try:
                if not np.isnan(p_val) and p_val < alpha:
                    count += 1
            except Exception:
                continue
        return count

    rejections_two_all = _count_rejections(summary_stats, "all", "ttest_two_sided_p", alpha_two)
    rejections_one_all = _count_rejections(summary_stats, "all", "ttest_one_sided_p", alpha_one)
    rejections_two_ilsp = _count_rejections(summary_stats, "ilsp", "ttest_two_sided_p", alpha_two)
    rejections_one_ilsp = _count_rejections(summary_stats, "ilsp", "ttest_one_sided_p", alpha_one)

    agg_rejections_two_all = _count_rejections(aggregated_stats, "all", "ttest_two_sided_p", alpha_two)
    agg_rejections_one_all = _count_rejections(aggregated_stats, "all", "ttest_one_sided_p", alpha_one)
    agg_rejections_two_ilsp = _count_rejections(aggregated_stats, "ilsp", "ttest_two_sided_p", alpha_two)
    agg_rejections_one_ilsp = _count_rejections(aggregated_stats, "ilsp", "ttest_one_sided_p", alpha_one)
    
    logger.info("=" * 80)
    logger.info("SUMMARY - PER-PROXY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total comparisons (individual proxies): {len(summary_stats)}")
    logger.info(f"Rejections [ALL] (two-sided, α={alpha_two}): {rejections_two_all}/{len(summary_stats)} ({100*rejections_two_all/len(summary_stats):.1f}%)")
    logger.info(f"Rejections [ALL] (one-sided, α={alpha_one}): {rejections_one_all}/{len(summary_stats)} ({100*rejections_one_all/len(summary_stats):.1f}%)")
    logger.info(f"Rejections [ILSP] (two-sided, α={alpha_two}): {rejections_two_ilsp}/{len(summary_stats)} ({100*rejections_two_ilsp/len(summary_stats):.1f}%)")
    logger.info(f"Rejections [ILSP] (one-sided, α={alpha_one}): {rejections_one_ilsp}/{len(summary_stats)} ({100*rejections_one_ilsp/len(summary_stats):.1f}%)")
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY - AGGREGATED STATISTICS (ALL PROXIES COMBINED)")
    logger.info("=" * 80)
    logger.info(f"Total judge-reference pairs: {len(aggregated_stats)}")
    if len(aggregated_stats) == 0:
        logger.info("No aggregated stats computed (likely no common ILSP examples across proxies).")
    else:
        logger.info(f"Rejections [ALL] (two-sided, α={alpha_two}): {agg_rejections_two_all}/{len(aggregated_stats)} ({100*agg_rejections_two_all/len(aggregated_stats):.1f}%)")
        logger.info(f"Rejections [ALL] (one-sided, α={alpha_one}): {agg_rejections_one_all}/{len(aggregated_stats)} ({100*agg_rejections_one_all/len(aggregated_stats):.1f}%)")
        logger.info(f"Rejections [ILSP] (two-sided, α={alpha_two}): {agg_rejections_two_ilsp}/{len(aggregated_stats)} ({100*agg_rejections_two_ilsp/len(aggregated_stats):.1f}%)")
        logger.info(f"Rejections [ILSP] (one-sided, α={alpha_one}): {agg_rejections_one_ilsp}/{len(aggregated_stats)} ({100*agg_rejections_one_ilsp/len(aggregated_stats):.1f}%)")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
