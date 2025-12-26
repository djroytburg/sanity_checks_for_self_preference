# DBG: analyze_judge_swap_results.py: Analyze and visualize judge swap null hypothesis test results
# Plots P(J chooses J over R) vs P(J chooses K over R) split by LSP/ILSP
# Written by: Dani
# Created: Dec 24, 2025, 02:45 EST
# Last Modified: Dec 24, 2025, 02:45 EST

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
# --- LOGGING SETUP ---
# -------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging with metadata."""
    log_dir = output_dir / "file_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analyze_judge_swap_{timestamp}.log"
    
    logger = logging.getLogger("analyze_judge_swap")
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
    logger.info("JUDGE SWAP ANALYSIS - RUN STARTED")
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
    res = bootstrap((np.array(data),), np.mean, random_state=42)
    CI = res.confidence_interval
    
    ax.hist(data, color=color, edgecolor='white', linewidth=0.5, **hist_kwargs)
    ax.axvline(mean, color='black', linestyle='--', linewidth=2, label='μ')
    ax.axvline(objective, color='red', linestyle='--', linewidth=2, label='Objective')
    ax.axvspan(CI.low, CI.high, alpha=0.2, color='orange', label='95% CI')
    
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
    Compute hypothesis tests for H0: P(J chooses J) = P(J chooses K).
    
    Returns:
        Dict with test statistics
    """
    stats_dict = {}
    
    # Basic statistics
    stats_dict["mean_j_vs_r"] = np.mean(j_vs_r_probs)
    stats_dict["std_j_vs_r"] = np.std(j_vs_r_probs)
    stats_dict["mean_k_vs_r"] = np.mean(k_vs_r_probs)
    stats_dict["std_k_vs_r"] = np.std(k_vs_r_probs)
    
    # Mean difference
    stats_dict["mean_diff"] = stats_dict["mean_j_vs_r"] - stats_dict["mean_k_vs_r"]
    
    # Paired t-test (two-sided): H0: mean(J) = mean(K)
    t_stat_two, p_two = stats.ttest_rel(j_vs_r_probs, k_vs_r_probs)
    stats_dict["ttest_two_sided_t"] = t_stat_two
    stats_dict["ttest_two_sided_p"] = p_two
    
    # One-sided t-test: H0: mean(J) <= mean(K), H1: mean(J) > mean(K)
    # (Self-preference hypothesis: J prefers its own more than K)
    t_stat_one, p_one = stats.ttest_rel(j_vs_r_probs, k_vs_r_probs, alternative='greater')
    stats_dict["ttest_one_sided_t"] = t_stat_one
    stats_dict["ttest_one_sided_p"] = p_one
    
    # KS test for distribution similarity
    ks_stat, ks_p = stats.ks_2samp(j_vs_r_probs, k_vs_r_probs)
    stats_dict["ks_statistic"] = ks_stat
    stats_dict["ks_pvalue"] = ks_p
    
    # Pearson correlation
    corr, corr_p = stats.pearsonr(j_vs_r_probs, k_vs_r_probs)
    stats_dict["correlation"] = corr
    stats_dict["correlation_pvalue"] = corr_p
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((stats_dict["std_j_vs_r"]**2 + stats_dict["std_k_vs_r"]**2) / 2)
    stats_dict["cohens_d"] = stats_dict["mean_diff"] / pooled_std if pooled_std > 0 else 0
    
    logger.info("Hypothesis Test Results:")
    logger.info(f"  Mean(J vs R): {stats_dict['mean_j_vs_r']:.4f} ± {stats_dict['std_j_vs_r']:.4f}")
    logger.info(f"  Mean(K vs R): {stats_dict['mean_k_vs_r']:.4f} ± {stats_dict['std_k_vs_r']:.4f}")
    logger.info(f"  Mean difference: {stats_dict['mean_diff']:.4f}")
    logger.info(f"  Two-sided t-test: t={t_stat_two:.3f}, p={p_two:.6f}")
    logger.info(f"  One-sided t-test: t={t_stat_one:.3f}, p={p_one:.6f}")
    logger.info(f"  KS test: D={ks_stat:.3f}, p={ks_p:.6f}")
    logger.info(f"  Correlation: r={corr:.3f}, p={corr_p:.6f}")
    logger.info(f"  Cohen's d: {stats_dict['cohens_d']:.3f}")
    
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
    results_dir = Path(args.results_dir) / args.dataset
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load cached preferences
    cache_dir = results_dir / "cache"
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return
    
    all_preferences = load_cached_preferences(cache_dir, logger)
    
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
        
        # Compute averaged probabilities
        j_vs_r_probs, j_ids, j_labels = compute_averaged_probabilities(
            games["J_vs_R_game1"], games["J_vs_R_game2"], logger
        )
        k_vs_r_probs, k_ids, k_labels = compute_averaged_probabilities(
            games["K_vs_R_game1"], games["K_vs_R_game2"], logger
        )
        
        logger.info(f"  J(J vs R): {len(j_vs_r_probs)} examples")
        logger.info(f"  J(K vs R): {len(k_vs_r_probs)} examples")
        
        # Filter to common example IDs (since proxies may have different example sets)
        j_ids_set = set(j_ids)
        k_ids_set = set(k_ids)
        common_ids = j_ids_set & k_ids_set
        
        if len(common_ids) < len(j_ids) or len(common_ids) < len(k_ids):
            logger.warning(f"  Example ID mismatch: J has {len(j_ids)}, K has {len(k_ids)}, Common: {len(common_ids)}")
        
        # Build dictionaries for fast lookup
        j_by_id = {ex_id: (prob, label) for ex_id, prob, label in zip(j_ids, j_vs_r_probs, j_labels)}
        k_by_id = {ex_id: (prob, label) for ex_id, prob, label in zip(k_ids, k_vs_r_probs, k_labels)}
        
        # Filter to common IDs and split by LSP/ILSP
        j_probs_split = {'lsp': [], 'ilsp': [], 'all': []}
        k_probs_split = {'lsp': [], 'ilsp': [], 'all': []}
        
        for ex_id in sorted(common_ids):
            j_prob, j_label = j_by_id[ex_id]
            k_prob, k_label = k_by_id[ex_id]
            
            # Sanity check: labels should match
            if j_label != k_label:
                logger.warning(f"  Label mismatch for {ex_id}: J={j_label}, K={k_label}")
            
            j_probs_split['all'].append(j_prob)
            k_probs_split['all'].append(k_prob)
            
            if j_label == "lsp":
                j_probs_split['lsp'].append(j_prob)
                k_probs_split['lsp'].append(k_prob)
            else:
                j_probs_split['ilsp'].append(j_prob)
                k_probs_split['ilsp'].append(k_prob)
        
        logger.info(f"  J(J vs R) - LSP: {len(j_probs_split['lsp'])}, ILSP: {len(j_probs_split['ilsp'])}")
        logger.info(f"  J(K vs R) - LSP: {len(k_probs_split['lsp'])}, ILSP: {len(k_probs_split['ilsp'])}")
        
        # Create plot only if we have data
        total_examples = len(j_probs_split['lsp']) + len(j_probs_split['ilsp'])
        if total_examples > 0:
            plot_file = output_dir / "plots" / f"{judge}_{proxy}_vs_{reference}.png"
            create_comparison_plot(
                j_probs_split, k_probs_split, judge, proxy, reference, plot_file, logger
            )
        else:
            logger.warning(f"  Skipping plot - no data for {judge} vs {proxy} vs {reference}")
        
        # Compute statistics on ILSP examples only (to avoid over-coding from legitimate eval)
        logger.info("  Computing hypothesis tests on ILSP examples only...")
        stats_dict = compute_hypothesis_tests(
            j_probs_split['ilsp'], 
            k_probs_split['ilsp'], 
            logger
        )
        stats_dict["judge"] = judge
        stats_dict["proxy"] = proxy
        stats_dict["reference"] = reference
        stats_dict["n_examples_total"] = len(j_vs_r_probs)
        stats_dict["n_lsp"] = len(j_probs_split['lsp'])
        stats_dict["n_ilsp"] = len(j_probs_split['ilsp'])
        stats_dict["n_examples_tested"] = len(j_probs_split['ilsp'])  # Only ILSP tested
        
        summary_stats.append(stats_dict)
        
        # Save individual stats
        stats_file = output_dir / "stats" / f"{judge}_{proxy}_vs_{reference}.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
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
    
    for (judge, reference), proxy_stats_list in by_judge_ref.items():
        logger.info(f"\nJudge={judge}, Reference={reference}")
        logger.info(f"  Number of proxies: {len(proxy_stats_list)}")
        
        # Log per-proxy statistics
        logger.info("  Per-proxy statistics:")
        for pstats in sorted(proxy_stats_list, key=lambda x: x['ttest_one_sided_p']):
            proxy = pstats['proxy']
            n = pstats['n_ilsp']
            p_one = pstats['ttest_one_sided_p']
            p_two = pstats['ttest_two_sided_p']
            logger.info(f"    {proxy}: n={n}, p_one={p_one:.4f}, p_two={p_two:.4f}")
        
        # Find most robust proxy (least likely to reject null - highest p-value)
        most_robust = max(proxy_stats_list, key=lambda x: x['ttest_one_sided_p'])
        logger.info(f"  Most robust proxy (highest p-value): {most_robust['proxy']} (p={most_robust['ttest_one_sided_p']:.4f})")
        
        # Aggregate all ILSP examples across proxies
        # Need to reload data to combine across proxies
        all_j_ilsp = []
        all_k_ilsp = []
        
        for pstats in proxy_stats_list:
            # Get the cached data for this triplet
            triplet_key = (pstats['judge'], pstats['proxy'], pstats['reference'])
            if triplet_key in all_preferences:
                games = all_preferences[triplet_key]
                j_probs, j_ids, j_labels = compute_averaged_probabilities(
                    games["J_vs_R_game1"], games["J_vs_R_game2"], logger
                )
                k_probs, k_ids, k_labels = compute_averaged_probabilities(
                    games["K_vs_R_game1"], games["K_vs_R_game2"], logger
                )
                
                # Filter to common IDs (same as in main analysis)
                j_ids_set = set(j_ids)
                k_ids_set = set(k_ids)
                common_ids = j_ids_set & k_ids_set
                
                # Build dictionaries for fast lookup
                j_by_id = {ex_id: (prob, label) for ex_id, prob, label in zip(j_ids, j_probs, j_labels)}
                k_by_id = {ex_id: (prob, label) for ex_id, prob, label in zip(k_ids, k_probs, k_labels)}
                
                # Extract ILSP only from common IDs
                for ex_id in common_ids:
                    j_prob, j_label = j_by_id[ex_id]
                    k_prob, k_label = k_by_id[ex_id]
                    
                    if j_label == "ilsp":
                        all_j_ilsp.append(j_prob)
                        all_k_ilsp.append(k_prob)
        
        logger.info(f"  Aggregated ILSP examples across all proxies: {len(all_j_ilsp)}")
        
        # Run hypothesis tests on aggregated data
        if len(all_j_ilsp) > 0 and len(all_k_ilsp) > 0:
            agg_stats = compute_hypothesis_tests(all_j_ilsp, all_k_ilsp, logger)
            logger.info(f"  Aggregated statistics:")
            logger.info(f"    J(J vs R) mean: {agg_stats['mean_j_vs_r']:.4f}")
            logger.info(f"    J(K vs R) mean: {agg_stats['mean_k_vs_r']:.4f}")
            logger.info(f"    Difference: {agg_stats['mean_diff']:.4f}")
            logger.info(f"    T-test one-sided p: {agg_stats['ttest_one_sided_p']:.4f}")
            logger.info(f"    T-test two-sided p: {agg_stats['ttest_two_sided_p']:.4f}")
            
            aggregated_stats.append({
                'judge': judge,
                'reference': reference,
                'n_proxies': len(proxy_stats_list),
                'n_ilsp_total': len(all_j_ilsp),
                'most_robust_proxy': most_robust['proxy'],
                'most_robust_p': most_robust['ttest_one_sided_p'],
                **agg_stats
            })
    
    # Save aggregated statistics
    agg_file = output_dir / "aggregated_by_judge_reference.json"
    with open(agg_file, 'w') as f:
        json.dump(aggregated_stats, f, indent=2)
    logger.info(f"\nSaved aggregated statistics to {agg_file}")
    
    # Count rejections
    alpha_two = 0.05
    alpha_one = 0.05
    rejections_two = sum(1 for s in summary_stats if s["ttest_two_sided_p"] < alpha_two)
    rejections_one = sum(1 for s in summary_stats if s["ttest_one_sided_p"] < alpha_one)
    
    # Count aggregated rejections
    agg_rejections_two = sum(1 for s in aggregated_stats if s["ttest_two_sided_p"] < alpha_two)
    agg_rejections_one = sum(1 for s in aggregated_stats if s["ttest_one_sided_p"] < alpha_one)
    
    logger.info("=" * 80)
    logger.info("SUMMARY - PER-PROXY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total comparisons (individual proxies): {len(summary_stats)}")
    logger.info(f"Rejections (two-sided, α={alpha_two}): {rejections_two}/{len(summary_stats)} ({100*rejections_two/len(summary_stats):.1f}%)")
    logger.info(f"Rejections (one-sided, α={alpha_one}): {rejections_one}/{len(summary_stats)} ({100*rejections_one/len(summary_stats):.1f}%)")
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY - AGGREGATED STATISTICS (ALL PROXIES COMBINED)")
    logger.info("=" * 80)
    logger.info(f"Total judge-reference pairs: {len(aggregated_stats)}")
    logger.info(f"Rejections (two-sided, α={alpha_two}): {agg_rejections_two}/{len(aggregated_stats)} ({100*agg_rejections_two/len(aggregated_stats):.1f}%)")
    logger.info(f"Rejections (one-sided, α={alpha_one}): {agg_rejections_one}/{len(aggregated_stats)} ({100*agg_rejections_one/len(aggregated_stats):.1f}%)")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
