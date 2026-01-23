# analyze_self-rec_null_dbg.py: Analyze correlation between self-preference and self-recognition
# Tests H0: Corr(J_pref, J_rec) = 0
# Created: Dec 24, 2025, 17:30 EST
# Last Modified: Dec 24, 2025, 17:30 EST

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# -------------------------
# --- LOGGING SETUP ---
# -------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging."""
    log_dir = output_dir / "file_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analyze_self_rec_null_dbg_{timestamp}.log"
    
    logger = logging.getLogger("analyze_self_rec_null_dbg")
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
    logger.info("SELF-RECOGNITION CORRELATION ANALYSIS")
    logger.info("=" * 80)
    
    return logger


# -------------------------
# --- DATA LOADING ---
# -------------------------

def load_preference_data(
    pref_results_dir: Path,
    judge_name: str,
    reference_name: str,
    logger: logging.Logger
) -> Dict[str, Dict]:
    """
    Load preference data from J vs R cache ONLY (never K vs R).
    
    Returns:
        Dict mapping example_id -> {
            'game1': {'prob_first': float, 'prob_second': float},
            'game2': {'prob_first': float, 'prob_second': float},
            'preference_for_self': float  # Averaged across games
        }
    """
    cache_dir = pref_results_dir / "cache" / judge_name / reference_name
    
    if not cache_dir.exists():
        logger.error(f"Preference cache directory not found: {cache_dir}")
        return {}
    
    # Load J vs R game1 and game2
    game1_file = cache_dir / "J_vs_R_game1.jsonl"
    game2_file = cache_dir / "J_vs_R_game2.jsonl"
    
    preference_data = defaultdict(dict)
    
    # Load game1 (J's output is first)
    if game1_file.exists():
        with open(game1_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    example_id = item['example_id']
                    # Preference cache uses prob_response1/prob_response2 instead of prob_first/prob_second
                    preference_data[example_id]['game1'] = {
                        'prob_first': item['prob_response1'],
                        'prob_second': item['prob_response2']
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing game1 line: {e}")
    else:
        logger.error(f"Game1 file not found: {game1_file}")
    
    # Load game2 (J's output is second)
    if game2_file.exists():
        with open(game2_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    example_id = item['example_id']
                    # Preference cache uses prob_response1/prob_response2 instead of prob_first/prob_second
                    preference_data[example_id]['game2'] = {
                        'prob_first': item['prob_response1'],
                        'prob_second': item['prob_response2']
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing game2 line: {e}")
    else:
        logger.error(f"Game2 file not found: {game2_file}")
    
    # Calculate average preference for self across games
    for example_id in list(preference_data.keys()):
        if 'game1' not in preference_data[example_id] or 'game2' not in preference_data[example_id]:
            logger.warning(f"Example {example_id} missing game data, skipping")
            del preference_data[example_id]
            continue
        
        # In game1: J is first, so prob_first = preference for self
        # In game2: J is second, so prob_second = preference for self
        pref_self_game1 = preference_data[example_id]['game1']['prob_first']
        pref_self_game2 = preference_data[example_id]['game2']['prob_second']
        
        preference_data[example_id]['preference_for_self'] = (pref_self_game1 + pref_self_game2) / 2
    
    logger.info(f"Loaded preference data for {len(preference_data)} examples")
    return dict(preference_data)


def load_recognition_data(
    rec_results_dir: Path,
    judge_name: str,
    reference_name: str,
    logger: logging.Logger
) -> Dict[str, Dict]:
    """
    Load self-recognition data from J_rec cache.
    
    Returns:
        Dict mapping example_id -> {
            'game1': {'prob_first': float, 'prob_second': float},
            'game2': {'prob_first': float, 'prob_second': float},
            'correct_recognition': float  # Averaged across games
        }
    """
    cache_dir = rec_results_dir / "cache" / judge_name / reference_name
    
    if not cache_dir.exists():
        logger.error(f"Recognition cache directory not found: {cache_dir}")
        return {}
    
    # Load J_rec game1 and game2
    game1_file = cache_dir / "J_rec_game1.jsonl"
    game2_file = cache_dir / "J_rec_game2.jsonl"
    
    recognition_data = defaultdict(dict)
    
    # Load game1 (J's output is first, correct answer = A)
    if game1_file.exists():
        with open(game1_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    example_id = item['example_id']
                    recognition_data[example_id]['game1'] = {
                        'prob_first': item['prob_first'],
                        'prob_second': item['prob_second']
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing recognition game1 line: {e}")
    else:
        logger.error(f"Recognition game1 file not found: {game1_file}")
    
    # Load game2 (J's output is second, correct answer = B)
    if game2_file.exists():
        with open(game2_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    example_id = item['example_id']
                    recognition_data[example_id]['game2'] = {
                        'prob_first': item['prob_first'],
                        'prob_second': item['prob_second']
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing recognition game2 line: {e}")
    else:
        logger.error(f"Recognition game2 file not found: {game2_file}")
    
    # Calculate average correct recognition probability across games
    for example_id in list(recognition_data.keys()):
        if 'game1' not in recognition_data[example_id] or 'game2' not in recognition_data[example_id]:
            logger.warning(f"Example {example_id} missing recognition game data, skipping")
            del recognition_data[example_id]
            continue
        
        # In game1: J is first (position A), correct = prob_first
        # In game2: J is second (position B), correct = prob_second
        correct_game1 = recognition_data[example_id]['game1']['prob_first']
        correct_game2 = recognition_data[example_id]['game2']['prob_second']
        
        recognition_data[example_id]['correct_recognition'] = (correct_game1 + correct_game2) / 2
    
    logger.info(f"Loaded recognition data for {len(recognition_data)} examples")
    return dict(recognition_data)


def load_lsp_ilsp_labels(
    proxy_data_dir: Path,
    dataset: str,
    judge_name: str,
    reference_name: str,
    logger: logging.Logger
) -> Dict[str, str]:
    """
    Load LSP/ILSP labels from proxy_preference_data.
    
    Returns:
        Dict mapping example_id -> 'LSP' or 'ILSP'
    """
    dataset_dir = proxy_data_dir / dataset
    
    # Find the JSON file for this judge-reference pair
    pattern = f"judge_{judge_name}_vs_{reference_name}.json"
    json_files = list(dataset_dir.glob(pattern))
    
    if not json_files:
        logger.warning(f"No proxy data file found matching: {pattern}")
        return {}
    
    json_file = json_files[0]
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = {}
    
    # Collect from all proxies (they should have same LSP/ILSP for same examples)
    for proxy_name, proxy_data in data['proxies'].items():
        for ex in proxy_data['data']['lsp']:
            labels[ex['id']] = 'LSP'
        for ex in proxy_data['data']['ilsp']:
            labels[ex['id']] = 'ILSP'
    
    logger.info(f"Loaded labels for {len(labels)} examples")
    return labels


def match_preference_recognition(
    preference_data: Dict,
    recognition_data: Dict,
    labels: Dict,
    logger: logging.Logger
) -> List[Dict]:
    """
    Match preference and recognition data by example_id.
    
    Returns:
        List of dicts with:
            - example_id
            - preference_for_self
            - correct_recognition
            - category (LSP/ILSP)
    """
    matched = []
    
    common_ids = set(preference_data.keys()) & set(recognition_data.keys())
    logger.info(f"Found {len(common_ids)} examples with both preference and recognition data")
    
    for example_id in common_ids:
        matched.append({
            'example_id': example_id,
            'preference_for_self': preference_data[example_id]['preference_for_self'],
            'correct_recognition': recognition_data[example_id]['correct_recognition'],
            'category': labels.get(example_id, 'unknown')
        })
    
    return matched


# -------------------------
# --- STATISTICAL ANALYSIS ---
# -------------------------

def compute_correlations(
    matched_data: List[Dict],
    logger: logging.Logger
) -> Dict:
    """
    Compute correlations between preference and recognition.
    """
    pref_scores = np.array([d['preference_for_self'] for d in matched_data])
    rec_scores = np.array([d['correct_recognition'] for d in matched_data])
    
    results = {
        'n_examples': len(matched_data),
        'mean_preference': float(np.mean(pref_scores)),
        'std_preference': float(np.std(pref_scores, ddof=1)),  # Use sample std
        'mean_recognition': float(np.mean(rec_scores)),
        'std_recognition': float(np.std(rec_scores, ddof=1)),  # Use sample std
        'min_preference': float(np.min(pref_scores)),
        'max_preference': float(np.max(pref_scores)),
        'min_recognition': float(np.min(rec_scores)),
        'max_recognition': float(np.max(rec_scores))
    }
    
    # Pearson correlation
    try:
        r, p = stats.pearsonr(pref_scores, rec_scores)
        results['pearson_r'] = float(r)
        results['pearson_p'] = float(p)
    except Exception as e:
        logger.error(f"Error computing Pearson correlation: {e}")
    
    # Spearman correlation
    try:
        rho, p = stats.spearmanr(pref_scores, rec_scores)
        results['spearman_rho'] = float(rho)
        results['spearman_p'] = float(p)
    except Exception as e:
        logger.error(f"Error computing Spearman correlation: {e}")
    
    # Point-biserial (binary recognition: correct/incorrect)
    try:
        rec_binary = (rec_scores > 0.5).astype(int)
        pb_r, pb_p = stats.pointbiserialr(rec_binary, pref_scores)
        results['pointbiserial_r'] = float(pb_r)
        results['pointbiserial_p'] = float(pb_p)
    except Exception as e:
        logger.error(f"Error computing point-biserial correlation: {e}")
    
    return results


def compute_stratified_correlations(
    matched_data: List[Dict],
    logger: logging.Logger
) -> Dict:
    """
    Compute correlations separately for LSP and ILSP.
    """
    lsp_data = [d for d in matched_data if d['category'] == 'LSP']
    ilsp_data = [d for d in matched_data if d['category'] == 'ILSP']
    
    results = {}
    
    if lsp_data:
        logger.info(f"Computing LSP correlations (n={len(lsp_data)})")
        results['LSP'] = compute_correlations(lsp_data, logger)
    
    if ilsp_data:
        logger.info(f"Computing ILSP correlations (n={len(ilsp_data)})")
        results['ILSP'] = compute_correlations(ilsp_data, logger)
    
    return results


def compare_lsp_vs_ilsp(
    matched_data: List[Dict],
    logger: logging.Logger
) -> Dict:
    """
    Compare recognition accuracy between LSP and ILSP cases using t-tests.
    """
    lsp_rec = [d['correct_recognition'] for d in matched_data if d['category'] == 'LSP']
    ilsp_rec = [d['correct_recognition'] for d in matched_data if d['category'] == 'ILSP']
    
    results = {
        'n_lsp': len(lsp_rec),
        'n_ilsp': len(ilsp_rec)
    }
    
    if lsp_rec:
        results['lsp_mean_recognition'] = float(np.mean(lsp_rec))
        results['lsp_std_recognition'] = float(np.std(lsp_rec, ddof=1))  # Use sample std
    
    if ilsp_rec:
        results['ilsp_mean_recognition'] = float(np.mean(ilsp_rec))
        results['ilsp_std_recognition'] = float(np.std(ilsp_rec, ddof=1))  # Use sample std
    
    # t-test
    if lsp_rec and ilsp_rec:
        try:
            t_stat, t_p = stats.ttest_ind(lsp_rec, ilsp_rec)
            results['ttest_t'] = float(t_stat)
            results['ttest_p'] = float(t_p)
            
            # Cohen's d - CORRECTED to weight by sample size
            n1, n2 = len(lsp_rec), len(ilsp_rec)
            var1, var2 = np.var(lsp_rec, ddof=1), np.var(ilsp_rec, ddof=1)  # Use sample variance (ddof=1)
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            pooled_std = np.sqrt(pooled_var)
            if pooled_std > 0:
                d = (np.mean(lsp_rec) - np.mean(ilsp_rec)) / pooled_std
                results['cohens_d'] = float(d)
        except Exception as e:
            logger.error(f"Error in t-test: {e}")
    
    return results


# -------------------------
# --- VISUALIZATION ---
# -------------------------

def create_scatter_plot(
    matched_data: List[Dict],
    output_dir: Path,
    judge_name: str,
    reference_name: str,
    overall_corr: Dict,
    logger: logging.Logger
):
    """
    Create scatter plot of recognition vs preference with LSP/ILSP coloring.
    """
    # Use clean white background style
    plt.style.use('default')
    sns.set_style("whitegrid", {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#e0e0e0",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5
    })
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Separate by category
    lsp_data = [d for d in matched_data if d['category'] == 'LSP']
    ilsp_data = [d for d in matched_data if d['category'] == 'ILSP']
    unknown_data = [d for d in matched_data if d['category'] == 'unknown']
    
    # Plot with distinct colors (circles for LSP, squares for ILSP)
    if lsp_data:
        lsp_pref = [d['preference_for_self'] for d in lsp_data]
        lsp_rec = [d['correct_recognition'] for d in lsp_data]
        ax.scatter(lsp_rec, lsp_pref, alpha=0.6, s=60, label=f'Legitimate Self-Preference (n={len(lsp_data)})', 
                  color='#2ca02c', marker='o', edgecolors='darkgreen', linewidths=0.5)
    
    if ilsp_data:
        ilsp_pref = [d['preference_for_self'] for d in ilsp_data]
        ilsp_rec = [d['correct_recognition'] for d in ilsp_data]
        ax.scatter(ilsp_rec, ilsp_pref, alpha=0.6, s=60, label=f'Illegitimate Self-Preference (n={len(ilsp_data)})', 
                  color='#d62728', marker='s', edgecolors='darkred', linewidths=0.5)
    
    if unknown_data:
        unk_pref = [d['preference_for_self'] for d in unknown_data]
        unk_rec = [d['correct_recognition'] for d in unknown_data]
        ax.scatter(unk_rec, unk_pref, alpha=0.3, s=40, label=f'Unknown (n={len(unknown_data)})', 
                  color='gray', marker='x')
    
    # Reference lines (chance level = 0.5)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    
    # Labels and title
    ax.set_xlabel('Self-Recognition Probability', fontsize=14, fontweight='bold', color='black')
    ax.set_ylabel('Self-Preference Probability (Prefers Own)', fontsize=14, fontweight='bold', color='black')
    
    title = f'Self-Recognition vs Self-Preference Strength\n{judge_name}-{reference_name}'
    if 'pearson_r' in overall_corr:
        title += f'\nPearson r = {overall_corr["pearson_r"]:.3f} (p = {overall_corr["pearson_p"]:.4f})'
    ax.set_title(title, fontsize=16, fontweight='bold', color='black', pad=20)
    
    ax.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='black', title='Category')
    ax.grid(True, alpha=0.3, linewidth=0.8)
    
    # Set axis limits with some padding
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Make tick labels bold
    ax.tick_params(labelsize=11, colors='black')
    
    plt.tight_layout()
    
    # Save with white background
    plot_file = output_dir / f"scatter_{judge_name}_vs_{reference_name}.png"
    plt.savefig(plot_file, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scatter plot: {plot_file}")


def create_histogram(
    matched_data: List[Dict],
    output_dir: Path,
    judge_name: str,
    reference_name: str,
    logger: logging.Logger
):
    """
    Create histograms of recognition and preference distributions.
    """
    # Use clean white background style
    plt.style.use('default')
    sns.set_style("whitegrid", {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#e0e0e0",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Recognition histogram
    rec_scores = [d['correct_recognition'] for d in matched_data]
    axes[0].hist(rec_scores, bins=20, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=1.2)
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Chance (0.5)')
    axes[0].axvline(x=np.mean(rec_scores), color='darkgreen', linestyle='-', linewidth=2.5, 
                   label=f'Mean = {np.mean(rec_scores):.3f}')
    axes[0].set_xlabel('Self-Recognition Probability', fontsize=13, fontweight='bold', color='black')
    axes[0].set_ylabel('Count', fontsize=13, fontweight='bold', color='black')
    axes[0].set_title('Distribution of Self-Recognition', fontsize=14, fontweight='bold', color='black')
    axes[0].legend(fontsize=11, framealpha=0.95, edgecolor='black')
    axes[0].grid(True, alpha=0.3, linewidth=0.8)
    axes[0].tick_params(labelsize=11, colors='black')
    
    # Preference histogram
    pref_scores = [d['preference_for_self'] for d in matched_data]
    axes[1].hist(pref_scores, bins=20, alpha=0.7, color='#ff7f0e', edgecolor='black', linewidth=1.2)
    axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Chance (0.5)')
    axes[1].axvline(x=np.mean(pref_scores), color='darkgreen', linestyle='-', linewidth=2.5, 
                   label=f'Mean = {np.mean(pref_scores):.3f}')
    axes[1].set_xlabel('Self-Preference Probability', fontsize=13, fontweight='bold', color='black')
    axes[1].set_ylabel('Count', fontsize=13, fontweight='bold', color='black')
    axes[1].set_title('Distribution of Self-Preference', fontsize=14, fontweight='bold', color='black')
    axes[1].legend(fontsize=11, framealpha=0.95, edgecolor='black')
    axes[1].grid(True, alpha=0.3, linewidth=0.8)
    axes[1].tick_params(labelsize=11, colors='black')
    
    fig.suptitle(f'{judge_name} vs {reference_name}', fontsize=16, fontweight='bold', 
                color='black', y=1.02)
    plt.tight_layout()
    
    # Save with white background
    plot_file = output_dir / f"histograms_{judge_name}_vs_{reference_name}.png"
    plt.savefig(plot_file, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved histograms: {plot_file}")


# -------------------------
# --- MAIN EXECUTION ---
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze self-recognition vs self-preference correlation")
    parser.add_argument("--rec_results_dir", type=str, default="self_rec_null_dbg_results",
                       help="Directory containing self-recognition results")
    parser.add_argument("--pref_results_dir", type=str, default="judge_swap_null_dbg_results",
                       help="Directory containing preference results")
    parser.add_argument("--proxy_data_dir", type=str, default="dbg-score-paper/proxy_preference_data",
                       help="Directory containing LSP/ILSP labels")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["alpaca_eval", "translation", "truthfulness"],
                       help="Dataset to analyze")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: rec_results_dir/dataset/analysis)")
    args = parser.parse_args()
    
    # Setup paths
    rec_results_dir = Path(args.rec_results_dir) / args.dataset
    pref_results_dir = Path(args.pref_results_dir) / args.dataset
    proxy_data_dir = Path(args.proxy_data_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = rec_results_dir / "analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info(f"Configuration:")
    logger.info(f"  Recognition results: {rec_results_dir}")
    logger.info(f"  Preference results: {pref_results_dir}")
    logger.info(f"  Proxy data: {proxy_data_dir}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Output: {output_dir}")
    
    # Find all judge-reference pairs
    rec_cache_dir = rec_results_dir / "cache"
    
    if not rec_cache_dir.exists():
        logger.error(f"Recognition cache not found: {rec_cache_dir}")
        return
    
    # Iterate over all judge-reference pairs
    all_results = {}
    
    for judge_dir in sorted(rec_cache_dir.iterdir()):
        if not judge_dir.is_dir():
            continue
        
        judge_name = judge_dir.name
        
        for ref_dir in sorted(judge_dir.iterdir()):
            if not ref_dir.is_dir():
                continue
            
            reference_name = ref_dir.name
            
            logger.info("=" * 80)
            logger.info(f"Analyzing: {judge_name} vs {reference_name}")
            logger.info("=" * 80)
            
            # Load data
            preference_data = load_preference_data(pref_results_dir, judge_name, reference_name, logger)
            recognition_data = load_recognition_data(rec_results_dir, judge_name, reference_name, logger)
            labels = load_lsp_ilsp_labels(proxy_data_dir, args.dataset, judge_name, reference_name, logger)
            
            if not preference_data or not recognition_data:
                logger.warning(f"Skipping {judge_name} vs {reference_name} due to missing data")
                continue
            
            # Match data
            matched_data = match_preference_recognition(preference_data, recognition_data, labels, logger)
            
            if not matched_data:
                logger.warning(f"No matched data for {judge_name} vs {reference_name}")
                continue
            
            # Compute statistics
            logger.info("\n--- Overall Correlations ---")
            overall_corr = compute_correlations(matched_data, logger)
            
            logger.info("\n--- Stratified Correlations ---")
            stratified_corr = compute_stratified_correlations(matched_data, logger)
            
            logger.info("\n--- LSP vs ILSP Comparison ---")
            lsp_ilsp_comp = compare_lsp_vs_ilsp(matched_data, logger)
            
            # Store results
            all_results[f"{judge_name}_vs_{reference_name}"] = {
                'judge': judge_name,
                'reference': reference_name,
                'overall': overall_corr,
                'stratified': stratified_corr,
                'lsp_vs_ilsp': lsp_ilsp_comp
            }
            
            # Create visualizations
            plots_dir = output_dir / "plots" / judge_name
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("\n--- Creating Visualizations ---")
            create_scatter_plot(matched_data, plots_dir, judge_name, reference_name, overall_corr, logger)
            create_histogram(matched_data, plots_dir, judge_name, reference_name, logger)
            
            # Print summary
            logger.info("\n" + "=" * 80)
            logger.info("SUMMARY")
            logger.info("=" * 80)
            logger.info(f"N examples: {overall_corr['n_examples']}")
            logger.info(f"Mean preference: {overall_corr['mean_preference']:.4f} ± {overall_corr['std_preference']:.4f}")
            logger.info(f"Mean recognition: {overall_corr['mean_recognition']:.4f} ± {overall_corr['std_recognition']:.4f}")
            
            if 'pearson_r' in overall_corr:
                logger.info(f"\nPearson correlation: r = {overall_corr['pearson_r']:.4f}, p = {overall_corr['pearson_p']:.4e}")
                sig = "✓ SIGNIFICANT" if overall_corr['pearson_p'] < 0.05 else "✗ NOT significant"
                logger.info(f"  {sig}")
            
            if 'spearman_rho' in overall_corr:
                logger.info(f"\nSpearman correlation: ρ = {overall_corr['spearman_rho']:.4f}, p = {overall_corr['spearman_p']:.4e}")
            
            if lsp_ilsp_comp.get('ttest_p'):
                logger.info(f"\nLSP vs ILSP recognition:")
                logger.info(f"  LSP: {lsp_ilsp_comp['lsp_mean_recognition']:.4f} (n={lsp_ilsp_comp['n_lsp']})")
                logger.info(f"  ILSP: {lsp_ilsp_comp['ilsp_mean_recognition']:.4f} (n={lsp_ilsp_comp['n_ilsp']})")
                logger.info(f"  t-test: t = {lsp_ilsp_comp['ttest_t']:.4f}, p = {lsp_ilsp_comp['ttest_p']:.4e}")
                sig = "✓ SIGNIFICANT" if lsp_ilsp_comp['ttest_p'] < 0.05 else "✗ NOT significant"
                logger.info(f"  {sig}")
            
            logger.info("=" * 80 + "\n")
    
    # Save all results to JSON
    results_file = output_dir / "correlation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Analysis complete! Results saved to: {output_dir}")
    logger.info(f"Summary JSON: {results_file}")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
