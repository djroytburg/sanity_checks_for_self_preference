
import json
import os
from pathlib import Path
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
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
        logging.FileHandler('file_logs/create_entropy_gap_per_reference_unified.log'),
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
    # Normalize name
    model_lower = model_name.lower()
    
    # Map variations to canonical names
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


def extract_model_size(model_name):
    """
    Extract model size in billions of parameters from model name.
    
    Args:
        model_name (str): Full model name
        
    Returns:
        int: Size in billions
    """
    pattern = r'(\d+)[bB](?:[^a-zA-Z]|$)'
    matches = re.findall(pattern, model_name)
    
    if matches:
        return int(matches[-1])
    
    return 16  # Default medium size


def size_to_marker_size(size):
    """
    Convert model size to matplotlib marker size.
    
    Args:
        size (int): Size in billions
        
    Returns:
        int: Marker size for matplotlib
    """
    if size <= 3:
        return 80
    elif size <= 9:
        return 200
    elif size <= 32:
        return 400
    else:
        return 700


def get_family_color(family):
    """
    Get color for model family.
    
    Args:
        family (str): Model family name
        
    Returns:
        str: Hex color code
    """
    colors = {
        'llama': '#4A90E2',   # Blue
        'gemma': '#2ECC71',   # Green
        'qwen': '#E74C3C',    # Red
        'gpt': '#F39C12',     # Orange
        'mistral': '#9B59B6', # Purple
        'phi': '#1ABC9C',     # Teal
        'deepseek': '#E67E22',# Dark Orange
        'other': '#95A5A6'    # Gray
    }
    return colors.get(family, colors['other'])


def get_dataset_marker(dataset):
    """
    Get marker shape for dataset.

    Args:
        dataset (str): Dataset name

    Returns:
        str: Matplotlib marker code
    """
    markers = {
        'math500': 'o',      # Circle
        'mmlu': 's',         # Square
        'mbpp-plus': '^',    # Triangle up
        'translation': 'D',  # Diamond
        'truthfulness': 'v', # Triangle down
        'quality': 'P',      # Plus (filled)
        'alpaca_eval': '*',  # Star
        'cnn': 'X',          # X (filled)
        'xsum': 'h',         # Hexagon
    }
    return markers.get(dataset, 'o')


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
        dict: {(judge, reference): {original_sp}}
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
        
        # Get original mean_j_vs_r (just the original, not the updated)
        if 'tests' in entry and 'ilsp' in entry['tests']:
            ilsp = entry['tests']['ilsp']
        else:
            ilsp = entry
        
        mean_j_vs_r = ilsp.get('mean_j_vs_r', 0)
        
        result[(judge, reference)] = {
            'original_sp': mean_j_vs_r * 100,  # Convert to percentage
        }
    
    logger.info(f'Loaded {len(result)} (judge, reference) pairs from {file_path}')
    return result


def load_per_reference_entropy():
    """
    Load per-reference entropy data.

    Returns:
        dict: Per-reference entropy indexed by "{dataset}||{judge}||{reference}"
    """
    entropy_file = Path('per_reference_entropy.json')

    if not entropy_file.exists():
        logger.error(f"Per-reference entropy file not found: {entropy_file}")
        return {}

    with open(entropy_file, 'r') as f:
        entropy_data = json.load(f)

    logger.info(f"Loaded {len(entropy_data)} per-reference entropy entries")
    return entropy_data


def load_panickserry_per_reference(winrate_dir, dataset):
    """
    Load self-preference data per (judge, reference) pair for panickserry datasets (CNN/XSUM).

    This loads from the analyze_proxy_robustness.py output which has J vs R statistics.

    Args:
        winrate_dir (str): Winrate directory path
        dataset (str): Dataset name (cnn or xsum)

    Returns:
        dict: {(judge, reference): {original_sp}}
    """
    # For panickserry, we need to load from the analyze_proxy_robustness output
    # which has the ILSP statistics we need

    # Try loading from the test output directory first
    potential_dirs = [
        Path('test_jvr_kvr_cnn_xsum/csv'),
        Path('test_cnn_xsum_output/csv'),
        Path('proxy_robustness_output/csv'),
        Path('jvr_kvr_analysis/csv'),
    ]

    ref_stats_file = None
    for dir_path in potential_dirs:
        potential_file = dir_path / 'statistics_by_dataset_judge_reference.csv'
        if potential_file.exists():
            ref_stats_file = potential_file
            break

    if not ref_stats_file:
        logger.warning(f'No reference-level statistics file found for panickserry datasets')
        return {}

    import pandas as pd
    df = pd.read_csv(ref_stats_file)

    # Filter for this dataset
    df = df[df['dataset'] == dataset]

    result = {}
    for _, row in df.iterrows():
        judge = row['judge']
        reference = row['reference']

        # Get mean_diff which represents J - K (original self-preference)
        mean_diff = row['mean_diff']

        result[(judge, reference)] = {
            'original_sp': mean_diff * 100,  # Convert to percentage
        }

    logger.info(f'Loaded {len(result)} (judge, reference) pairs for {dataset} from {ref_stats_file}')
    return result


# ----------------------
# --- DATA PREPARATION ---
# ----------------------

def prepare_unified_scatter_data():
    """
    Prepare scatter plot data aggregated across all datasets.

    Returns:
        dict: {dataset: [(entropy_gap, hspp, family, size, judge, reference), ...]}
    """
    # Load per-reference entropy
    per_ref_entropy = load_per_reference_entropy()

    # Define datasets
    experiments = [
        ('judge_swap_null_verif_smoke2', 'math500'),
        ('judge_swap_null_verif_smoke2', 'mmlu'),
        ('judge_swap_null_verif_smoke2', 'mbpp-plus'),
        ('judge_swap_null_dbg_results', 'translation'),
        ('judge_swap_null_dbg_results', 'truthfulness'),
        ('judge_swap_null_author_obfuscation', 'quality'),
        ('judge_swap_null_dbg_results', 'alpaca_eval'),
    ]

    # Panickserry datasets
    panickserry_experiments = [
        ('CNN_and_XSUM results/cnn_winrates', 'cnn'),
        ('CNN_and_XSUM results/xsum_winrates', 'xsum'),
    ]

    all_data = defaultdict(list)

    # Process standard datasets
    for results_dir, dataset in experiments:
        logger.info(f"\nProcessing {dataset}...")

        # Load self-preference data
        sp_data = load_self_preference_per_reference(results_dir, dataset)

        if not sp_data:
            logger.warning(f"No self-preference data for {dataset}")
            continue

        # Create points for each (judge, reference)
        for (judge, reference), sp in sp_data.items():
            # Get per-reference entropy
            key = f"{dataset}||{judge}||{reference}"

            if key not in per_ref_entropy:
                logger.debug(f'No per-reference entropy for ({dataset}, {judge}, {reference})')
                continue

            entropy_gap = per_ref_entropy[key]['entropy_gap']
            hspp = sp['original_sp']  # Original SP (not updated)

            family = extract_model_family(judge)
            size = extract_model_size(judge)

            all_data[dataset].append({
                'entropy_gap': entropy_gap,
                'hspp': hspp,
                'family': family,
                'size': size,
                'judge': judge,
                'reference': reference
            })

        logger.info(f"  {len(all_data[dataset])} points for {dataset}")

    # Process panickserry datasets
    for winrate_dir, dataset in panickserry_experiments:
        logger.info(f"\nProcessing {dataset}...")

        # Load self-preference data for panickserry
        sp_data = load_panickserry_per_reference(winrate_dir, dataset)

        if not sp_data:
            logger.warning(f"No self-preference data for {dataset}")
            continue

        # Create points for each (judge, reference)
        for (judge, reference), sp in sp_data.items():
            # Get per-reference entropy
            key = f"{dataset}||{judge}||{reference}"

            if key not in per_ref_entropy:
                logger.debug(f'No per-reference entropy for ({dataset}, {judge}, {reference})')
                continue

            entropy_gap = per_ref_entropy[key]['entropy_gap']
            hspp = sp['original_sp']  # Original SP (not updated)

            family = extract_model_family(judge)
            size = extract_model_size(judge)

            all_data[dataset].append({
                'entropy_gap': entropy_gap,
                'hspp': hspp,
                'family': family,
                'size': size,
                'judge': judge,
                'reference': reference
            })

        logger.info(f"  {len(all_data[dataset])} points for {dataset}")

    return all_data


# ----------------------
# --- PLOTTING ---
# ----------------------

def create_unified_scatter_plot(all_data):
    """
    Create single unified scatter plot with all per-reference pairs.
    
    Args:
        all_data (dict): {dataset: [point_dicts, ...]}
        
    Returns:
        matplotlib figure, dict of family R² values
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data for regression per family
    family_data = defaultdict(lambda: {'x': [], 'y': []})
    
    # Track present families and sizes
    families_present = set()
    sizes_present = set()
    datasets_present = set()
    
    # Plot each dataset with its own marker
    for dataset, points in sorted(all_data.items()):
        # Sort by size (largest first) so larger points render in back
        sorted_points = sorted(points, key=lambda p: p['size'], reverse=True)
        
        datasets_present.add(dataset)
        
        for point in sorted_points:
            family = point['family']
            size = point['size']
            families_present.add(family)
            sizes_present.add(size)
            
            color = get_family_color(family)
            marker = get_dataset_marker(dataset)
            marker_size = size_to_marker_size(size)
            
            x = point['entropy_gap']
            y = point['hspp'] / 100  # Convert to 0-1 scale
            
            # Plot with full alpha (no updated points, just originals)
            ax.scatter(x, y,
                      color=color, 
                      s=marker_size, 
                      alpha=1.0,
                      marker=marker,
                      edgecolors='black', 
                      linewidth=0.8)
            
            # Collect for regression per family
            family_data[family]['x'].append(x)
            family_data[family]['y'].append(y)
    
    # Compute and plot line of best fit per family
    logger.info("\n=== FAMILY-CONDITIONAL REGRESSION ===")
    for family in sorted(families_present):
        if len(family_data[family]['x']) < 2:
            continue
        
        x_fam = family_data[family]['x']
        y_fam = family_data[family]['y']
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_fam, y_fam)
        r_squared = r_value ** 2
        
        # Generate line points
        x_line = np.linspace(min(x_fam), max(x_fam), 100)
        y_line = slope * x_line + intercept
        
        # Plot regression line with family color
        color = get_family_color(family)
        ax.plot(x_line, y_line, 
               color=color, 
               linestyle='--', 
               linewidth=2, 
               alpha=0.6,
               label=f'{family.capitalize()} (R² = {r_squared:.3f})')
        
        logger.info(f"{family.capitalize()}: R² = {r_squared:.3f}, p = {p_value:.3e}, n = {len(x_fam)}")
        logger.info(f"  slope = {slope:.4f}, intercept = {intercept:.4f}")
    
    # Add dotted gray line at y=0
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Entropy Gap (bits)', fontsize=14)
    ax.set_ylabel('Harmful Self-Preference Propensity', fontsize=14)
    ax.set_title('Per-Reference Entropy Gap vs. HSPP (All Datasets)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.7, linestyle='-', linewidth=0.8, color='gray')
    
    # Set y-limits to show negative region clearly and extend above 1.0
    ax.set_ylim(-0.6, 1.2)
    
    # Add R² to legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    
    # Create legend
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    
    # Row 1: Datasets (shapes)
    dataset_handles = []
    dataset_display_names = {
        'math500': 'MATH500',
        'mmlu': 'MMLU',
        'mbpp-plus': 'MBPP+',
        'translation': 'Translation',
        'truthfulness': 'Truthfulness',
        'quality': 'Quality',
        'alpaca_eval': 'Alpaca Eval',
        'cnn': 'CNN',
        'xsum': 'XSUM',
    }
    for dataset in sorted(datasets_present):
        dataset_handles.append(
            Line2D([0], [0], marker=get_dataset_marker(dataset), color='w',
                   markerfacecolor='gray', markersize=10,
                   label=dataset_display_names.get(dataset, dataset),
                   markeredgecolor='black', markeredgewidth=0.8)
        )
    
    # Row 2: Families (colors)
    family_handles = []
    for family in sorted(families_present):
        family_handles.append(
            mpatches.Patch(color=get_family_color(family), label=family.capitalize())
        )
    
    # Row 3: Sizes
    size_ranges = [
        (3, '≤3B', 80),
        (7, '4-9B', 200),
        (14, '10-32B', 400),
        (70, '≥70B', 700)
    ]
    size_handles = []
    for ref_size, label, marker_size in size_ranges:
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
    
    # Create three-row legend layout
    n_datasets = len(dataset_handles)
    n_families = len(family_handles)
    n_sizes = len(size_handles)
    
    # Row 1: Datasets
    legend1 = fig.legend(dataset_handles, [h.get_label() for h in dataset_handles],
                        loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                        ncol=n_datasets, fontsize=10, frameon=False,
                        title='Dataset (marker shape)')
    
    # Row 2: Families
    legend2 = fig.legend(family_handles, [h.get_label() for h in family_handles],
                        loc='lower center', bbox_to_anchor=(0.5, -0.08),
                        ncol=n_families, fontsize=10, frameon=False,
                        title='Model family (color)')
    
    # Row 3: Sizes
    legend3 = fig.legend(size_handles, [h.get_label() for h in size_handles],
                        loc='lower center', bbox_to_anchor=(0.5, -0.14),
                        ncol=n_sizes, fontsize=10, frameon=False,
                        title='Model size')
    
    # Add all legends to figure
    fig.add_artist(legend1)
    fig.add_artist(legend2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    
    # Return family-specific R² values
    family_r2 = {}
    for family in sorted(families_present):
        if len(family_data[family]['x']) >= 2:
            _, _, r_value, _, _ = stats.linregress(family_data[family]['x'], family_data[family]['y'])
            family_r2[family] = r_value ** 2
    
    return fig, family_r2


# ----------------------
# --- STATISTICS ---
# ----------------------

def generate_entropy_gap_statistics(all_data):
    """
    Generate statistics on entropy gap > 0 at different levels of precision.
    
    Args:
        all_data (dict): {dataset: [point_dicts, ...]}
    """
    # Load per-reference entropy
    per_ref_entropy = load_per_reference_entropy()
    
    # Statistics at (dataset, judge, reference) level
    per_ref_stats = []
    per_ref_positive = 0
    per_ref_total = 0
    
    for dataset, points in all_data.items():
        for point in points:
            judge = point['judge']
            reference = point['reference']
            entropy_gap = point['entropy_gap']
            
            per_ref_stats.append({
                'dataset': dataset,
                'judge': judge,
                'reference': reference,
                'entropy_gap': float(entropy_gap),
                'is_positive': bool(entropy_gap > 0)
            })
            
            per_ref_total += 1
            if entropy_gap > 0:
                per_ref_positive += 1
    
    # Statistics at (dataset, judge) level - aggregate across references
    judge_level_data = defaultdict(lambda: {'entropy_gaps': []})
    
    for dataset, points in all_data.items():
        for point in points:
            judge = point['judge']
            entropy_gap = point['entropy_gap']
            key = (dataset, judge)
            judge_level_data[key]['entropy_gaps'].append(entropy_gap)
    
    per_judge_stats = []
    per_judge_positive = 0
    per_judge_total = 0
    
    for (dataset, judge), data in judge_level_data.items():
        avg_entropy_gap = np.mean(data['entropy_gaps'])
        
        per_judge_stats.append({
            'dataset': dataset,
            'judge': judge,
            'avg_entropy_gap': float(avg_entropy_gap),
            'n_references': int(len(data['entropy_gaps'])),
            'is_positive': bool(avg_entropy_gap > 0)
        })
        
        per_judge_total += 1
        if avg_entropy_gap > 0:
            per_judge_positive += 1
    
    # Summary statistics
    summary = {
        'per_reference_level': {
            'total': per_ref_total,
            'entropy_gap_positive': per_ref_positive,
            'entropy_gap_negative': per_ref_total - per_ref_positive,
            'pct_positive': (per_ref_positive / per_ref_total * 100) if per_ref_total > 0 else 0
        },
        'per_judge_level': {
            'total': per_judge_total,
            'entropy_gap_positive': per_judge_positive,
            'entropy_gap_negative': per_judge_total - per_judge_positive,
            'pct_positive': (per_judge_positive / per_judge_total * 100) if per_judge_total > 0 else 0
        },
        'by_dataset_reference': {},
        'by_dataset_judge': {}
    }
    
    # Break down by dataset at reference level
    for dataset in all_data.keys():
        dataset_refs = [r for r in per_ref_stats if r['dataset'] == dataset]
        total = len(dataset_refs)
        positive = sum(1 for r in dataset_refs if r['is_positive'])
        
        summary['by_dataset_reference'][dataset] = {
            'total': total,
            'entropy_gap_positive': positive,
            'entropy_gap_negative': total - positive,
            'pct_positive': (positive / total * 100) if total > 0 else 0
        }
    
    # Break down by dataset at judge level
    for dataset in all_data.keys():
        dataset_judges = [j for j in per_judge_stats if j['dataset'] == dataset]
        total = len(dataset_judges)
        positive = sum(1 for j in dataset_judges if j['is_positive'])
        
        summary['by_dataset_judge'][dataset] = {
            'total': total,
            'entropy_gap_positive': positive,
            'entropy_gap_negative': total - positive,
            'pct_positive': (positive / total * 100) if total > 0 else 0
        }
    
    # Save JSON
    stats_dir = Path('entropy_gap_statistics')
    stats_dir.mkdir(exist_ok=True)
    
    with open(stats_dir / 'entropy_gap_statistics.json', 'w') as f:
        json.dump({
            'summary': summary,
            'per_reference_details': per_ref_stats,
            'per_judge_details': per_judge_stats
        }, f, indent=2)
    
    logger.info(f"Saved statistics to {stats_dir / 'entropy_gap_statistics.json'}")
    
    # Generate LaTeX table
    latex_output = []
    latex_output.append(r"\begin{table}[h]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{Entropy Gap Statistics}")
    latex_output.append(r"\begin{tabular}{lrrrr}")
    latex_output.append(r"\hline")
    latex_output.append(r"Level & Total & Positive & Negative & \% Positive \\")
    latex_output.append(r"\hline")
    
    # Overall statistics
    latex_output.append(r"\multicolumn{5}{l}{\textbf{Overall Statistics}} \\")
    latex_output.append(f"Per-Reference & {summary['per_reference_level']['total']} & "
                       f"{summary['per_reference_level']['entropy_gap_positive']} & "
                       f"{summary['per_reference_level']['entropy_gap_negative']} & "
                       f"{summary['per_reference_level']['pct_positive']:.1f}\\% \\\\")
    latex_output.append(f"Per-Judge & {summary['per_judge_level']['total']} & "
                       f"{summary['per_judge_level']['entropy_gap_positive']} & "
                       f"{summary['per_judge_level']['entropy_gap_negative']} & "
                       f"{summary['per_judge_level']['pct_positive']:.1f}\\% \\\\")
    latex_output.append(r"\hline")
    
    # By dataset (reference level)
    latex_output.append(r"\multicolumn{5}{l}{\textbf{By Dataset (Per-Reference)}} \\")
    for dataset, stats in sorted(summary['by_dataset_reference'].items()):
        latex_output.append(f"{dataset} & {stats['total']} & "
                           f"{stats['entropy_gap_positive']} & "
                           f"{stats['entropy_gap_negative']} & "
                           f"{stats['pct_positive']:.1f}\\% \\\\")
    latex_output.append(r"\hline")
    
    # By dataset (judge level)
    latex_output.append(r"\multicolumn{5}{l}{\textbf{By Dataset (Per-Judge)}} \\")
    for dataset, stats in sorted(summary['by_dataset_judge'].items()):
        latex_output.append(f"{dataset} & {stats['total']} & "
                           f"{stats['entropy_gap_positive']} & "
                           f"{stats['entropy_gap_negative']} & "
                           f"{stats['pct_positive']:.1f}\\% \\\\")
    
    latex_output.append(r"\hline")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"\end{table}")
    
    latex_str = "\n".join(latex_output)
    
    with open(stats_dir / 'entropy_gap_statistics.tex', 'w') as f:
        f.write(latex_str)
    
    logger.info(f"Saved LaTeX table to {stats_dir / 'entropy_gap_statistics.tex'}")
    
    # Log summary
    logger.info("\n=== ENTROPY GAP STATISTICS ===")
    logger.info(f"Per-Reference Level: {per_ref_positive}/{per_ref_total} ({summary['per_reference_level']['pct_positive']:.1f}%) positive")
    logger.info(f"Per-Judge Level: {per_judge_positive}/{per_judge_total} ({summary['per_judge_level']['pct_positive']:.1f}%) positive")


# ----------------------
# --- MAIN EXECUTION ---
# ----------------------

def main():
    """
    Main function to generate unified per-reference scatter plot.
    """
    logger.info("Starting unified per-reference entropy gap scatter plot generation")
    
    # Prepare data
    all_data = prepare_unified_scatter_data()
    
    if not all_data:
        logger.error("No data to plot!")
        return
    
    # Count total points
    total_points = sum(len(points) for points in all_data.values())
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Total points: {total_points}")
    for dataset, points in sorted(all_data.items()):
        logger.info(f"  {dataset}: {len(points)} points")
    
    # Create plot
    logger.info("\n=== GENERATING UNIFIED PLOT ===")
    fig, family_r2 = create_unified_scatter_plot(all_data)
    
    # Create output directory
    output_dir = Path('entropy_gap_per_reference_scatter_plots')
    output_dir.mkdir(exist_ok=True)
    for subdir in ['png', 'pdf', 'svg']:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    # Save in multiple formats
    png_path = output_dir / 'png' / 'per_reference_unified.png'
    pdf_path = output_dir / 'pdf' / 'per_reference_unified.pdf'
    svg_path = output_dir / 'svg' / 'per_reference_unified.svg'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    
    logger.info(f"Saved unified plots: {png_path}, {pdf_path}, {svg_path}")
    
    if family_r2:
        logger.info(f"\n=== FAMILY R² VALUES ===")
        for family, r2 in sorted(family_r2.items()):
            logger.info(f"  {family.capitalize()}: R² = {r2:.3f}")
    
    plt.close(fig)
    
    # Generate entropy gap statistics
    logger.info("\n=== GENERATING ENTROPY GAP STATISTICS ===")
    generate_entropy_gap_statistics(all_data)
    
    logger.info("\n✓ Unified per-reference entropy gap scatter plot generation complete!")


if __name__ == '__main__':
    main()
