# create_entropy_gap_per_reference_scatter.py: Per-reference drill-down scatter plots
# Created: January 18, 2026, 03:25 AM EST
# Last Modified: January 18, 2026, 03:25 AM EST

import json
import os
from pathlib import Path
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import numpy as np
import re

# Use same utilities as aggregated version
import sys
sys.path.insert(0, str(Path(__file__).parent))
from create_entropy_gap_scatter import (
    extract_model_family, extract_model_size, size_to_marker_size,
    get_family_color, load_spread_statistics
)

# ----------------------
# --- SETUP LOGGING ---
# ----------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_logs/create_entropy_gap_per_reference_scatter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ----------------------
# --- DATA LOADING ---
# ----------------------

def load_self_preference_per_reference(results_dir, dataset):
    """
    Load self-preference data per (judge, reference) pair.
    
    Args:
        results_dir (str): Results directory name
        dataset (str): Dataset name
        
    Returns:
        dict: {(judge, reference): {original_sp, updated_sp}}
    """
    file_path = Path(results_dir) / dataset / 'analysis' / 'aggregated_by_judge_reference.json'
    
    if not file_path.exists():
        logger.warning(f'File not found: {file_path}')
        return {}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    result = {}
    for entry in data:
        judge = entry['judge']
        reference = entry['reference']
        
        # Get original mean_j_vs_r
        if 'tests' in entry and 'ilsp' in entry['tests']:
            ilsp = entry['tests']['ilsp']
        else:
            ilsp = entry
        
        mean_j_vs_r = ilsp.get('mean_j_vs_r', 0)
        mean_diff = ilsp.get('mean_diff', 0)
        
        result[(judge, reference)] = {
            'original_sp': mean_j_vs_r * 100,
            'updated_sp': mean_diff * 100,  # Just mean_diff, not sum
        }
    
    logger.info(f'Loaded {len(result)} (judge, reference) pairs from {file_path}')
    return result


# ----------------------
# --- DATA PREPARATION ---
# ----------------------

def prepare_per_reference_scatter_data(spread_stats, sp_data, dataset, per_ref_entropy):
    """
    Prepare scatter plot data with one point per (judge, reference).
    
    Args:
        spread_stats (list): Spread statistics
        sp_data (dict): {(judge, reference): {original_sp, updated_sp}}
        dataset (str): Dataset name
        per_ref_entropy (dict): Per-reference entropy from per_reference_entropy.json
        
    Returns:
        dict: {(judge, ref): {entropy_gap, original_sp, updated_sp, family, size}}
    """
    result = {}
    
    # Create one point per (judge, reference)
    for (judge, reference), sp in sp_data.items():
        # Get per-reference entropy
        key = f"{dataset}||{judge}||{reference}"
        if key not in per_ref_entropy:
            logger.debug(f'No per-reference entropy for ({dataset}, {judge}, {reference})')
            continue
        
        entropy_gap = per_ref_entropy[key]['entropy_gap']
        
        family = extract_model_family(judge)
        size = extract_model_size(judge)
        
        result_key = (judge, reference)
        result[result_key] = {
            'entropy_gap': entropy_gap,
            'original_sp': sp['original_sp'],
            'updated_sp': sp['updated_sp'],
            'family': family,
            'size': size,
            'judge': judge,
            'reference': reference
        }
    
    return result


# ----------------------
# --- PLOTTING ---
# ----------------------

def create_per_reference_scatter(scatter_data, dataset_name):
    """
    Create scatter plot with one point per (judge, reference).
    
    Args:
        scatter_data (dict): {(judge, ref): metrics}
        dataset_name (str): Dataset display name
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Sort by size (largest first) so larger points render in back
    sorted_items = sorted(scatter_data.items(), 
                         key=lambda x: x[1]['size'], 
                         reverse=True)
    
    # Plot each (judge, reference) point
    for (judge, reference), metrics in sorted_items:
        family = metrics['family']
        size = metrics['size']
        color = get_family_color(family)
        marker_size = size_to_marker_size(size)
        
        x = metrics['entropy_gap']
        original_y = metrics['original_sp'] / 100  # Convert to 0-1 scale
        updated_y = metrics['updated_sp'] / 100    # Convert to 0-1 scale
        
        # Plot original (shaded, low alpha)
        ax.scatter(x, original_y,
                  color=color, s=marker_size, alpha=0.3,
                  edgecolors=color, linewidth=1.5)
        
        # Plot updated (solid, full alpha)
        ax.scatter(x, updated_y,
                  color=color, s=marker_size, alpha=1.0,
                  edgecolors='black', linewidth=1)
        
        # Draw connecting line
        ax.plot([x, x], [original_y, updated_y],
               color=color, alpha=0.3, linestyle='--', linewidth=1)
    
    # Add dotted red line at y=0
    ax.axhline(y=0, color='red', linestyle=':', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Entropy Gap (H(ILSP) - H(LSP), bits)', fontsize=12)
    ax.set_ylabel('Harmful Self-Preference Propensity', fontsize=12)
    ax.set_title(f'{dataset_name} (Per Reference)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.7, linestyle='-', linewidth=0.8, color='gray')
    ax.set_ylim(-0.5, 1)
    
    return fig, ax


# ----------------------
# --- MAIN EXECUTION ---
# ----------------------

def main():
    """
    Main function to generate per-reference drill-down scatter plots.
    """
    logger.info("Starting per-reference entropy gap scatter plot generation")
    
    # Load per-reference entropy
    logger.info("Loading per-reference entropy...")
    with open('per_reference_entropy.json', 'r') as f:
        per_ref_entropy = json.load(f)
    logger.info(f"Loaded {len(per_ref_entropy)} per-reference entropy values")
    
    # Load spread statistics (not needed anymore, but kept for compatibility)
    logger.info("Loading spread statistics...")
    spread_stats = load_spread_statistics()
    
    # Define datasets
    experiments = [
        ('judge_swap_null_verif_smoke2', 'MATH500', 'math500'),
        ('judge_swap_null_verif_smoke2', 'MMLU', 'mmlu'),
        ('judge_swap_null_verif_smoke2', 'MBPP+', 'mbpp-plus'),
        ('judge_swap_null_dbg_results', 'Translation', 'translation'),
        ('judge_swap_null_dbg_results', 'Truthfulness', 'truthfulness'),
        ('judge_swap_null_author_obfuscation', 'Quality', 'quality'),
        ('judge_swap_null_dbg_results', 'Alpaca Eval', 'alpaca_eval'),
        ('CNN_and_XSUM results/cnn_results','Cnn','cnn'),
        ('CNN_and_XSUM results/xsum_result','Xsum','xsum'),
    ]
    
    # Create output directory
    output_dir = Path('entropy_gap_per_reference_scatter_plots')
    output_dir.mkdir(exist_ok=True)
    for subdir in ['png', 'pdf', 'svg']:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    logger.info(f'Output directory: {output_dir}')
    
    # Process each dataset
    for results_dir, display_name, dataset_key in experiments:
        logger.info(f'\\nProcessing {display_name}...')
        
        # Load self-preference data per reference
        sp_data = load_self_preference_per_reference(results_dir, dataset_key)
        
        if not sp_data:
            logger.warning(f'No self-preference data found for {dataset_key}')
            continue
        
        # Prepare scatter data
        scatter_data = prepare_per_reference_scatter_data(spread_stats, sp_data, dataset_key, per_ref_entropy)
        
        if not scatter_data:
            logger.warning(f'No scatter data prepared for {dataset_key}')
            continue
        
        logger.info(f'Found {len(scatter_data)} (judge, reference) pairs with entropy gap data')
        
        # Create scatter plot
        fig, ax = create_per_reference_scatter(scatter_data, display_name)
        
        # Save in multiple formats
        base_name = f'{dataset_key}_per_ref_entropy_scatter'
        
        png_path = output_dir / 'png' / f'{base_name}.png'
        pdf_path = output_dir / 'pdf' / f'{base_name}.pdf'
        svg_path = output_dir / 'svg' / f'{base_name}.svg'
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        
        logger.info(f'Saved plots: {png_path}, {pdf_path}, {svg_path}')
        
        plt.close(fig)
    
    logger.info('\\n✓ Per-reference entropy gap scatter plot generation complete!')


if __name__ == '__main__':
    main()
