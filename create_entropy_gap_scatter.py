# create_entropy_gap_scatter.py: Generate scatter plots of self-preference vs entropy gap (replacing task accuracy)
# Created: January 18, 2026, 03:15 AM EST
# Last Modified: January 18, 2026, 03:15 AM EST

import json
import os
from pathlib import Path
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import numpy as np
import re


# ----------------------
# --- SETUP LOGGING ---
# ----------------------

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('file_logs/create_entropy_gap_scatter.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)


# ----------------------
# --- UTILITIES ---
# ----------------------

def extract_model_family(model_name):
    """
    Extract model family from model name.
    
    Args:
        model_name (str): Full model name
        
    Returns:
        str: Model family (llama, gemma, qwen, gpt, mistral, phi, deepseek, other)
    """
    families = {
        'meta-llama': 'llama',
        'llama': 'llama',
        'gemma': 'gemma',
        'gpt': 'gpt',
        'qwen': 'qwen',
        'mistral': 'mistral',
        'phi': 'phi',
        'deepseek': 'deepseek',
    }
    
    model_lower = model_name.lower()
    for key, family in families.items():
        if key in model_lower:
            return family
    return 'other'


def extract_model_size(model_name):
    """
    Extract model size in billions of parameters from model name.
    
    Args:
        model_name (str): Full model name
        
    Returns:
        int: Size in billions (3, 7, 9, 14, 32, 70, 72, or 16 if not parseable)
    """
    # Look for patterns like "3b", "7b", "70b", etc.
    pattern = r'(\d+)[bB](?:[^a-zA-Z]|$)'
    matches = re.findall(pattern, model_name)
    
    if matches:
        # Return the largest match (in case there are multiple)
        return int(matches[-1])
    
    # Default to medium if not found
    return 16


def size_to_category(size):
    """
    Categorize model size into size buckets.
    
    Args:
        size (int): Size in billions
        
    Returns:
        str: Size category (Small, Moderate, Medium, Large)
    """
    if size <= 3:
        return 'Small (3B)'
    elif size <= 9:
        return 'Moderate (7-9B)'
    elif size <= 32:
        return 'Medium (14-32B)'
    else:
        return 'Large (70-72B)'


def size_to_marker_size(size):
    """
    Convert size to matplotlib marker size.
    
    Args:
        size (int): Size in billions
        
    Returns:
        int: Marker size for scatter plot
    """
    if size <= 3:
        return 100
    elif size <= 9:
        return 200
    elif size <= 32:
        return 300
    else:
        return 500


def get_family_color(family):
    """
    Get color for model family.
    
    Args:
        family (str): Model family
        
    Returns:
        str: Color hex code
    """
    colors = {
        'llama': '#4A90E2',  # Blue
        'gemma': '#2ECC71',  # Green
        'qwen': '#E74C3C',   # Red
        'gpt': '#F39C12',    # Orange
        'mistral': '#9B59B6', # Purple
        'phi': '#1ABC9C',    # Teal
        'deepseek': '#E67E22', # Dark Orange
        'other': '#95A5A6'   # Gray
    }
    return colors.get(family, colors['other'])


# ----------------------
# --- DATA LOADING ---
# ----------------------

def load_spread_statistics():
    """
    Load spread statistics from JSON file.
    
    Returns:
        dict: Spread statistics by dataset and judge
    """
    with open('spread_statistics_analysis/all_spread_statistics.json', 'r') as f:
        data = json.load(f)
    return data


def load_self_preference_data(results_dir, dataset):
    """
    Load self-preference data from aggregated JSON.
    
    Args:
        results_dir (str): Results directory name
        dataset (str): Dataset name
        
    Returns:
        dict: Self-preference data by judge
    """
    file_path = Path(results_dir) / dataset / 'analysis' / 'aggregated_by_judge_reference.json'
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return {}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Aggregate by judge
    judges_data = defaultdict(list)
    for entry in data:
        judge = entry['judge']
        judges_data[judge].append(entry)
    
    # Calculate mean self-preference per judge
    result = {}
    for judge, entries in judges_data.items():
        original_sps = []
        updated_sps = []
        
        for entry in entries:
            # Get original mean_j_vs_r
            if 'tests' in entry and 'ilsp' in entry['tests']:
                ilsp = entry['tests']['ilsp']
            else:
                ilsp = entry
            
            mean_j_vs_r = ilsp.get('mean_j_vs_r', 0)
            mean_diff = ilsp.get('mean_diff', 0)
            
            original_sps.append(mean_j_vs_r)
            updated_sps.append(mean_diff)  # Just mean_diff, not sum
        
        result[judge] = {
            'original_sp': np.mean(original_sps) * 100 if original_sps else 0,
            'updated_sp': np.mean(updated_sps) * 100 if updated_sps else 0,
        }
    
    return result


# ----------------------
# --- DATA PREPARATION ---
# ----------------------

def prepare_scatter_data(spread_stats, sp_data, dataset):
    """
    Prepare data for scatter plot.
    
    Args:
        spread_stats (list): List of spread statistics entries
        sp_data (dict): Self-preference data
        dataset (str): Dataset name
        
    Returns:
        dict: {judge_name: {entropy_gap, original_sp, updated_sp, family, size}}
    """
    result = {}
    
    # spread_stats is a list of entries with 'dataset' and 'judge' fields
    for entry in spread_stats:
        if entry.get('dataset') != dataset:
            continue
        
        judge = entry.get('judge')
        if not judge:
            continue
        
        # Get entropy gap
        entropy_gap = entry.get('gap_entropy', None)
        if entropy_gap is None or np.isnan(entropy_gap):
            continue
        
        # Get self-preference data
        if judge not in sp_data:
            logger.warning(f"Judge {judge} not found in self-preference data")
            continue
        
        family = extract_model_family(judge)
        size = extract_model_size(judge)
        
        result[judge] = {
            'entropy_gap': entropy_gap,
            'original_sp': sp_data[judge]['original_sp'],
            'updated_sp': sp_data[judge]['updated_sp'],
            'family': family,
            'size': size,
            'size_category': size_to_category(size)
        }
    
    return result


# ----------------------
# --- PLOTTING ---
# ----------------------

def create_combined_scatter_plot(all_scatter_data):
    """
    Create combined scatter plot with subplots for all datasets.
    
    Args:
        all_scatter_data (dict): {dataset_name: {judge: metrics}}
        
    Returns:
        matplotlib figure
    """
    # Create subplots
    n_datasets = len(all_scatter_data)
    n_cols = min(4, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    fig.suptitle('Entropy Gap vs. Harmful Self-Preference Propensity', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Collect all families and sizes actually present in the data
    families_present = set()
    sizes_present = set()
    
    # Plot each dataset
    for idx, dataset_name in enumerate(sorted(all_scatter_data.keys())):
        ax = axes[idx]
        
        scatter_data = all_scatter_data[dataset_name]
        
        # Sort judges by size (descending) so larger nodes render first
        sorted_judges = sorted(scatter_data.items(), 
                              key=lambda item: item[1]['size'], 
                              reverse=True)
        
        # Plot for each judge (largest to smallest)
        for judge, metrics in sorted_judges:
            family = metrics['family']
            size = metrics['size']
            families_present.add(family)
            sizes_present.add(size)
            
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
        
        ax.set_xlabel('Entropy Gap (bits)', fontsize=10)
        ax.set_ylabel('Harmful Self-Preference Propensity', fontsize=10)
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.7, linestyle='-', linewidth=0.8, color='gray')
        ax.set_ylim(-0.5, 1)
    
    # Hide extra subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend with only present families and sizes
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    
    # Row 1: Family legend (only those present)
    family_handles = []
    for family in sorted(families_present):
        family_handles.append(
            mpatches.Patch(color=get_family_color(family), label=family.capitalize())
        )
    
    # Row 2: Size legend with ranges (only those present)
    size_ranges = [
        (3, '≤3B', 80),
        (7, '4-9B', 200),
        (14, '10-32B', 400),
        (70, '≥70B', 700)
    ]
    size_handles = []
    for ref_size, label, marker_size in size_ranges:
        # Only include if we have sizes in this range
        has_size_in_range = False
        if ref_size == 3 and any(s <= 3 for s in sizes_present):
            has_size_in_range = True
        elif ref_size == 7 and any(4 <= s <= 9 for s in sizes_present):
            has_size_in_range = True
        elif ref_size == 14 and any(10 <= s <= 32 for s in sizes_present):
            has_size_in_range = True
        elif ref_size == 70 and any(s >= 70 for s in sizes_present):
            has_size_in_range = True
        
        if has_size_in_range:
            size_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=np.sqrt(marker_size / np.pi),
                       label=label, markeredgecolor='black', markeredgewidth=0.5)
            )
    
    # Row 3: Alpha legend
    alpha_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=9, alpha=0.3, markeredgecolor='gray', 
               markeredgewidth=1.5, label='Original finding'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=9, alpha=1.0, markeredgecolor='black', 
               markeredgewidth=1, label='With judge swap baseline')
    ]
    
    # Create three-row legend layout
    n_families = len(family_handles)
    n_sizes = len(size_handles)
    
    # Row 1: Families
    legend1 = fig.legend(family_handles, [h.get_label() for h in family_handles],
                        loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                        ncol=n_families, fontsize=10, frameon=False)
    
    # Row 2: Sizes  
    legend2 = fig.legend(size_handles, [h.get_label() for h in size_handles],
                        loc='lower center', bbox_to_anchor=(0.5, -0.08),
                        ncol=n_sizes, fontsize=10, frameon=False)
    
    # Row 3: Alpha
    legend3 = fig.legend(alpha_handles, [h.get_label() for h in alpha_handles],
                        loc='lower center', bbox_to_anchor=(0.5, -0.14),
                        ncol=2, fontsize=10, frameon=False)
    
    # Add all legends to figure
    fig.add_artist(legend1)
    fig.add_artist(legend2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    
    return fig


# ----------------------
# --- MAIN EXECUTION ---
# ----------------------

def main():
    """
    Main function to generate scatter plots with entropy gap on x-axis.
    """
    # logger.info("Starting entropy gap scatter plot generation")
    # logger.info(f"Generated by: Dani")
    # logger.info(f"Timestamp: {pd.Timestamp.now(tz='US/Eastern')}")
    
    # # Load spread statistics
    # logger.info("Loading spread statistics...")
    spread_stats = load_spread_statistics()
    
    # Define datasets grouped by paper
    paper_groups = [
        ('panickserry', [
            ('panickserry_results/cnn_results', 'CNN', 'cnn'),
            ('panickserry_results/xsum_result', 'XSUM', 'xsum')
            
        ]),
        
    ]
    
    # Create output directory
    output_dir = Path('entropy_gap_scatter_plots')
    output_dir.mkdir(exist_ok=True)
    for subdir in ['png', 'pdf', 'svg']:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Process each paper group
    for paper_name, experiments in paper_groups:
        logger.info(f"\n=== PROCESSING {paper_name.upper()} ===")
        
        # Collect scatter data for this paper
        paper_scatter_data = {}
        
        for results_dir, display_name, dataset_key in experiments:
            logger.info(f"  Loading {display_name}...")
            
            # Load self-preference data
            sp_data = load_self_preference_data(results_dir, dataset_key)
            
            if not sp_data:
                logger.warning(f"  No self-preference data found for {dataset_key}")
                continue
            
            # Prepare scatter data
            scatter_data = prepare_scatter_data(spread_stats, sp_data, dataset_key)
            
            if not scatter_data:
                logger.warning(f"  No scatter data prepared for {dataset_key}")
                continue
            
            logger.info(f"  Found {len(scatter_data)} judges with entropy gap data")
            paper_scatter_data[display_name] = scatter_data
        
        if not paper_scatter_data:
            logger.warning(f"No data for {paper_name}, skipping...")
            continue
        
        # Create combined plot for this paper
        logger.info(f"  Generating plot for {paper_name}...")
        fig = create_combined_scatter_plot(paper_scatter_data)
        
        # Save in multiple formats
        png_path = output_dir / 'png' / f'{paper_name}.png'
        pdf_path = output_dir / 'pdf' / f'{paper_name}.pdf'
        svg_path = output_dir / 'svg' / f'{paper_name}.svg'
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        
        logger.info(f"  Saved: {png_path}, {pdf_path}, {svg_path}")
        
        plt.close(fig)
    
    logger.info("\n✓ Entropy gap scatter plot generation complete!")


if __name__ == '__main__':
    import pandas as pd
    main()
