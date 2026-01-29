# compute_entropy_gap_correlations.py: Compute R² correlations for entropy gap vs original HSPP
# Created: January 18, 2026, 03:58 AM EST
# Last Modified: January 18, 2026, 03:58 AM EST

import json
import os
from pathlib import Path
from collections import defaultdict
import logging
import numpy as np
import re
from scipy import stats


# ----------------------
# --- SETUP LOGGING ---
# ----------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_logs/compute_entropy_gap_correlations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ----------------------
# --- UTILITIES ---
# ----------------------

def extract_model_family(model_name):
    """
    Extract and normalize model family from model name.
    
    Args:
        model_name (str): Full model name
        
    Returns:
        str: Normalized model family
    """
    model_lower = model_name.lower()
    
    family_mappings = {
        'llama': ['llama', 'meta-llama', 'meta_llama'],
        'gemma': ['gemma', 'google/gemma', 'gemma-2'],
        'qwen': ['qwen', 'qwen2', 'qwen2.5'],
        'gpt': ['gpt', 'openai', 'gpt-3.5', 'gpt-4'],
        'mistral': ['mistral', 'mistralai'],
        'phi': ['phi', 'microsoft/phi'],
        'deepseek': ['deepseek', 'deepseek-ai'],
    }
    
    for canonical, variations in family_mappings.items():
        for variation in variations:
            if variation in model_lower:
                return canonical
    
    return 'other'


# ----------------------
# --- DATA LOADING ---
# ----------------------

def load_spread_statistics():
    """
    Load entropy gap statistics from spread statistics analysis.
    
    Returns:
        dict: {dataset: {judge: entropy_gap}}
    """
    stats_file = Path('spread_statistics_analysis/all_spread_statistics.json')
    
    if not stats_file.exists():
        logger.error(f"Spread statistics file not found: {stats_file}")
        return {}
    
    with open(stats_file, 'r') as f:
        all_stats = json.load(f)
    
    result = defaultdict(dict)
    
    for entry in all_stats:
        dataset = entry['dataset']
        judge = entry['judge']
        entropy_gap = entry.get('gap_entropy', 0)
        
        result[dataset][judge] = entropy_gap
    
    logger.info(f"Loaded spread statistics for {len(all_stats)} (dataset, judge) pairs")
    return result


def load_self_preference_data(results_dir, dataset):
    """
    Load self-preference data from aggregated results.
    
    Args:
        results_dir (str): Results directory name
        dataset (str): Dataset name
        
    Returns:
        dict: {judge_name: {original_sp, updated_sp}}
    """
    file_path = Path(results_dir) / dataset / 'analysis' / 'aggregated_by_judge_reference.json'
    
    if not file_path.exists():
        logger.warning(f'File not found: {file_path}')
        return {}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    judge_sp = defaultdict(lambda: {'original_sp': [], 'updated_sp': []})
    
    for entry in data:
        judge = entry.get('judge')
        if 'tests' in entry and 'ilsp' in entry['tests']:
            ilsp = entry['tests']['ilsp']
        else:
            ilsp = entry
        
        mean_j_vs_r = ilsp.get('mean_j_vs_r', 0)
        mean_diff = ilsp.get('mean_diff', 0)
        
        judge_sp[judge]['original_sp'].append(mean_j_vs_r * 100)
        judge_sp[judge]['updated_sp'].append(mean_diff * 100)
    
    # Average across references
    result = {}
    for judge, sp_data in judge_sp.items():
        result[judge] = {
            'original_sp': np.mean(sp_data['original_sp']),
            'updated_sp': np.mean(sp_data['updated_sp']),
        }
    
    logger.info(f'Loaded self-preference data for {len(result)} judges from {dataset}')
    return result


# ----------------------
# --- CORRELATION ANALYSIS ---
# ----------------------

def compute_correlations():
    """
    Compute R² correlations between entropy gap and original HSPP.
    
    Returns:
        dict: Correlation statistics overall and per-family
    """
    # Load spread statistics
    spread_stats = load_spread_statistics()
    
    # Define datasets
    experiments = [
        ('judge_swap_null_verif_smoke2', 'math500'),
        ('judge_swap_null_verif_smoke2', 'mmlu'),
        ('judge_swap_null_verif_smoke2', 'mbpp-plus'),
        ('judge_swap_null_dbg_results', 'translation'),
        ('judge_swap_null_dbg_results', 'truthfulness'),
        ('judge_swap_null_author_obfuscation', 'quality'),
        ('judge_swap_null_dbg_results', 'alpaca_eval'),
        ('CNN_and_XSUM results/cnn_results','cnn'),
        ('CNN_and_XSUM results/xsum_result','xsum'),
        
    ]
    
    # Collect all data points
    all_points = []
    family_points = defaultdict(list)
    
    for results_dir, dataset in experiments:
        logger.info(f"\nProcessing {dataset}...")
        
        # Load self-preference data
        sp_data = load_self_preference_data(results_dir, dataset)
        
        if not sp_data:
            logger.warning(f"No self-preference data for {dataset}")
            continue
        
        # Match with entropy gaps
        for judge, sp in sp_data.items():
            if judge not in spread_stats.get(dataset, {}):
                logger.debug(f"No entropy gap for {judge} in {dataset}")
                continue
            
            entropy_gap = spread_stats[dataset][judge]
            original_sp = sp['original_sp']
            family = extract_model_family(judge)
            
            point = {
                'dataset': dataset,
                'judge': judge,
                'family': family,
                'entropy_gap': entropy_gap,
                'original_sp': original_sp
            }
            
            all_points.append(point)
            family_points[family].append(point)
        
        logger.info(f"  Added {len([p for p in all_points if p['dataset'] == dataset])} points from {dataset}")
    
    logger.info(f"\n=== TOTAL DATA POINTS: {len(all_points)} ===")
    
    # Compute overall correlation
    all_x = [p['entropy_gap'] for p in all_points]
    all_y = [p['original_sp'] for p in all_points]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_x, all_y)
    overall_r2 = r_value ** 2
    
    logger.info(f"\n=== OVERALL CORRELATION ===")
    logger.info(f"R² = {overall_r2:.4f}, p = {p_value:.3e}, n = {len(all_points)}")
    logger.info(f"slope = {slope:.4f}, intercept = {intercept:.4f}")
    
    # Compute per-family correlations
    family_correlations = {}
    
    logger.info(f"\n=== PER-FAMILY CORRELATIONS ===")
    for family in sorted(family_points.keys()):
        points = family_points[family]
        
        if len(points) < 2:
            logger.warning(f"{family}: Only {len(points)} point(s), skipping regression")
            continue
        
        x = [p['entropy_gap'] for p in points]
        y = [p['original_sp'] for p in points]
        
        slope_f, intercept_f, r_value_f, p_value_f, std_err_f = stats.linregress(x, y)
        r2_f = r_value_f ** 2
        
        family_correlations[family] = {
            'r_squared': float(r2_f),
            'p_value': float(p_value_f),
            'slope': float(slope_f),
            'intercept': float(intercept_f),
            'n_points': len(points)
        }
        
        logger.info(f"{family.capitalize()}: R² = {r2_f:.4f}, p = {p_value_f:.3e}, n = {len(points)}")
        logger.info(f"  slope = {slope_f:.4f}, intercept = {intercept_f:.4f}")
    
    # Prepare results
    results = {
        'overall': {
            'r_squared': float(overall_r2),
            'p_value': float(p_value),
            'slope': float(slope),
            'intercept': float(intercept),
            'n_points': len(all_points)
        },
        'by_family': family_correlations,
        'data_points': all_points
    }
    
    return results


# ----------------------
# --- OUTPUT GENERATION ---
# ----------------------

def generate_outputs(results):
    """
    Generate JSON and LaTeX outputs for correlation results.
    
    Args:
        results (dict): Correlation results
    """
    # Create output directory
    stats_dir = Path('entropy_gap_statistics')
    stats_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_file = stats_dir / 'entropy_gap_correlations.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nSaved JSON to: {json_file}")
    
    # Generate LaTeX table
    latex_output = []
    latex_output.append(r"\begin{table}[h]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{Entropy Gap vs. Original HSPP Correlations}")
    latex_output.append(r"\begin{tabular}{lrrrrr}")
    latex_output.append(r"\hline")
    latex_output.append(r"Group & $R^2$ & $p$-value & Slope & Intercept & $n$ \\")
    latex_output.append(r"\hline")
    
    # Overall
    overall = results['overall']
    latex_output.append(r"\textbf{Overall} & "
                       f"{overall['r_squared']:.4f} & "
                       f"{overall['p_value']:.2e} & "
                       f"{overall['slope']:.4f} & "
                       f"{overall['intercept']:.4f} & "
                       f"{overall['n_points']} \\\\")
    latex_output.append(r"\hline")
    
    # Per family
    latex_output.append(r"\multicolumn{6}{l}{\textbf{By Model Family}} \\")
    for family in sorted(results['by_family'].keys()):
        stats = results['by_family'][family]
        latex_output.append(f"{family.capitalize()} & "
                           f"{stats['r_squared']:.4f} & "
                           f"{stats['p_value']:.2e} & "
                           f"{stats['slope']:.4f} & "
                           f"{stats['intercept']:.4f} & "
                           f"{stats['n_points']} \\\\")
    
    latex_output.append(r"\hline")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"\label{tab:entropy_gap_correlations}")
    latex_output.append(r"\end{table}")
    
    latex_str = "\n".join(latex_output)
    
    tex_file = stats_dir / 'entropy_gap_correlations.tex'
    with open(tex_file, 'w') as f:
        f.write(latex_str)
    
    logger.info(f"Saved LaTeX to: {tex_file}")


# ----------------------
# --- MAIN EXECUTION ---
# ----------------------

def main():
    """
    Main function to compute entropy gap correlations.
    """
    logger.info("Starting entropy gap correlation analysis")
    logger.info("Analyzing: Entropy Gap vs. Original HSPP (aggregated by judge)")
    
    # Compute correlations
    results = compute_correlations()
    
    # Generate outputs
    generate_outputs(results)
    
    logger.info("\n✓ Entropy gap correlation analysis complete!")


if __name__ == '__main__':
    main()
