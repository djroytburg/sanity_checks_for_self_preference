
import json
import os
from pathlib import Path
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import numpy as np
import re
from glob import glob

# ----------------------
# --- SETUP LOGGING ---
# ----------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_logs/analyze_judge_self_preference_scatter_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ----------------------
# --- UTILITIES ---
# ----------------------

def extract_model_family(model_name):
    """Extract model family from model name."""
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
    """Extract model size in billions of parameters."""
    pattern = r'(\d+)[bB](?:[^a-zA-Z]|$)'
    matches = re.findall(pattern, model_name)
    return int(matches[-1]) if matches else 16


def size_to_marker_size(size):
    """Convert size to matplotlib marker size with more dramatic differentiation."""
    if size <= 3:
        return 80
    elif size <= 9:
        return 200
    elif size <= 32:
        return 400
    else:
        return 700


def get_family_color(family):
    """Get color for model family."""
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

def load_author_obf_data(dataset_dir):
    """
    Load author obfuscation task accuracy data.
    
    Args:
        dataset_dir (str): Path to proxies directory
        
    Returns:
        dict: {(judge, reference): task_accuracy}
    """
    logger.info(f"Loading author_obf data from {dataset_dir}")
    
    judge_ref_accuracy = {}
    
    # Glob all JSON files
    json_files = glob(f"{dataset_dir}/*.json")
    logger.info(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            judge = data.get('reference_evaluator')
            reference = data.get('reference_evaluatee')
            
            if not judge or not reference:
                logger.warning(f"Missing judge/reference in {json_file}")
                continue
            
            # Get winrate from first proxy (should be same across all proxies)
            proxies = data.get('proxies', {})
            if not proxies:
                logger.warning(f"No proxies found in {json_file}")
                continue
            
            # Validate consistency across proxies
            winrates = []
            for proxy_name, proxy_data in proxies.items():
                wr = proxy_data.get('reference_winrate')
                if wr is not None:
                    winrates.append(wr)
            
            if not winrates:
                logger.warning(f"No reference_winrate found in proxies for {json_file}")
                continue
            
            # Assert all winrates are the same
            if not all(abs(w - winrates[0]) < 1e-6 for w in winrates):
                logger.error(f"Inconsistent winrates in {json_file}: {winrates}")
                raise ValueError(f"Inconsistent winrates for ({judge}, {reference})")
            
            task_accuracy = winrates[0]
            
            # Assert consistency if we've seen this (judge, ref) before
            key = (judge, reference)
            if key in judge_ref_accuracy:
                if abs(judge_ref_accuracy[key] - task_accuracy) > 1e-6:
                    logger.error(f"Inconsistent task_accuracy for {key}: {judge_ref_accuracy[key]} vs {task_accuracy}")
                    raise ValueError(f"Inconsistent accuracy for {key}")
            else:
                judge_ref_accuracy[key] = task_accuracy
                logger.debug(f"  ({judge}, {reference}): {task_accuracy:.4f}")
        
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            raise
    
    logger.info(f"Loaded {len(judge_ref_accuracy)} unique judge/reference pairs")
    return judge_ref_accuracy


def load_dbg_data(dataset_dir, dataset_name):
    """
    Load DBG task accuracy data.
    
    Args:
        dataset_dir (str): Path to judge_* files directory
        dataset_name (str): Name of dataset for logging
        
    Returns:
        dict: {(judge, reference): task_accuracy}
    """
    logger.info(f"Loading DBG data from {dataset_dir} ({dataset_name})")
    
    judge_ref_accuracy = {}
    
    # Glob all JSON files
    json_files = glob(f"{dataset_dir}/judge_*.json")
    logger.info(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            judge = data.get('judge_name')
            reference = data.get('reference_name')
            
            if not judge or not reference:
                logger.warning(f"Missing judge/reference in {json_file}")
                continue
            
            # Get winrate from first proxy
            proxies = data.get('proxies', {})
            if not proxies:
                logger.warning(f"No proxies found in {json_file}")
                continue
            
            # Validate consistency across proxies
            winrates = []
            for proxy_name, proxy_data in proxies.items():
                wr = proxy_data.get('judge_winrate')
                if wr is not None:
                    winrates.append(wr)
            
            if not winrates:
                logger.warning(f"No judge_winrate found in proxies for {json_file}")
                continue
            
            # Assert all winrates are the same
            if not all(abs(w - winrates[0]) < 1e-6 for w in winrates):
                logger.error(f"Inconsistent winrates in {json_file}: {winrates}")
                raise ValueError(f"Inconsistent winrates for ({judge}, {reference})")
            
            task_accuracy = winrates[0]
            
            # Assert consistency if we've seen this (judge, ref) before
            key = (judge, reference)
            if key in judge_ref_accuracy:
                if abs(judge_ref_accuracy[key] - task_accuracy) > 1e-6:
                    logger.error(f"Inconsistent task_accuracy for {key}: {judge_ref_accuracy[key]} vs {task_accuracy}")
                    raise ValueError(f"Inconsistent accuracy for {key}")
            else:
                judge_ref_accuracy[key] = task_accuracy
                logger.debug(f"  ({judge}, {reference}): {task_accuracy:.4f}")
        
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            raise
    
    logger.info(f"Loaded {len(judge_ref_accuracy)} unique judge/reference pairs")
    return judge_ref_accuracy


def load_verif_data(dataset_dir, dataset_name):
    """
    Load verification task accuracy data.
    
    Args:
        dataset_dir (str): Path to {family}/proxies directory
        dataset_name (str): Name of dataset
        
    Returns:
        dict: {(judge, reference): task_accuracy}
    """
    logger.info(f"Loading verif data from {dataset_dir} ({dataset_name})")
    
    judge_ref_accuracy = {}
    
    # Find all judge files
    judge_files = glob(f"{dataset_dir}/*.json")
    logger.info(f"Found {len(judge_files)} judge JSON files")
    
    for judge_file in judge_files:
        try:
            with open(judge_file, 'r') as f:
                data = json.load(f)
            
            # Filename is {judge_model_name}.json
            judge_name = Path(judge_file).stem
            logger.debug(f"Processing judge: {judge_name}")
            
            # Iterate through references
            for ref_name, ref_data in data.items():
                if not isinstance(ref_data, dict):
                    continue
                
                # Iterate through proxies
                winrates = []
                metadata_list = []
                
                for proxy_name, proxy_data in ref_data.items():
                    if isinstance(proxy_data, dict) and 'judge_winrate' in proxy_data:
                        metadata = proxy_data.get('metadata', {})
                        
                        # Validate metadata
                        assert metadata.get('judge_name') == judge_name, \
                            f"Judge name mismatch: {metadata.get('judge_name')} vs {judge_name}"
                        assert metadata.get('reference_name') == ref_name, \
                            f"Reference name mismatch: {metadata.get('reference_name')} vs {ref_name}"
                        
                        wr = proxy_data.get('judge_winrate')
                        winrates.append(wr)
                        metadata_list.append(metadata)
                
                if not winrates:
                    logger.debug(f"  No judge_winrate data for ({judge_name}, {ref_name})")
                    continue
                
                # Assert consistency across proxies
                if not all(abs(w - winrates[0]) < 1e-6 for w in winrates):
                    logger.error(f"Inconsistent winrates for ({judge_name}, {ref_name}): {winrates}")
                    raise ValueError(f"Inconsistent winrates for ({judge_name}, {ref_name})")
                
                task_accuracy = winrates[0]
                
                # Assert consistency if we've seen this (judge, ref) before
                key = (judge_name, ref_name)
                if key in judge_ref_accuracy:
                    if abs(judge_ref_accuracy[key] - task_accuracy) > 1e-6:
                        logger.error(f"Inconsistent task_accuracy for {key}: {judge_ref_accuracy[key]} vs {task_accuracy}")
                        raise ValueError(f"Inconsistent accuracy for {key}")
                else:
                    judge_ref_accuracy[key] = task_accuracy
                    logger.debug(f"  ({judge_name}, {ref_name}): {task_accuracy:.4f}")
        
        except Exception as e:
            logger.error(f"Error processing {judge_file}: {e}")
            raise
    
    logger.info(f"Loaded {len(judge_ref_accuracy)} unique judge/reference pairs from {dataset_name}")
    return judge_ref_accuracy


# ----------------------
# --- AGGREGATION ---
# ----------------------

def aggregate_judge_across_references(judge_ref_accuracy, judge_name):
    """
    Average task accuracy across all references for a judge.
    
    Args:
        judge_ref_accuracy (dict): {(judge, reference): task_accuracy}
        judge_name (str): Name of judge to aggregate
        
    Returns:
        float: Unweighted average task accuracy across all references
    """
    accuracies = [
        accuracy for (judge, ref), accuracy in judge_ref_accuracy.items()
        if judge == judge_name
    ]
    
    if not accuracies:
        logger.warning(f"No references found for judge {judge_name}")
        return 0.0
    
    avg = np.mean(accuracies)
    logger.debug(f"Judge {judge_name}: {len(accuracies)} refs, avg accuracy: {avg:.4f}")
    return avg


def prepare_scatter_data(judge_ref_accuracy, self_pref_data=None):
    """
    Prepare scatter plot data by aggregating per judge.
    
    Args:
        judge_ref_accuracy (dict): {(judge, reference): task_accuracy}
        self_pref_data (dict): {judge_name: {original_sp, updated_sp}} or None
        
    Returns:
        dict: {judge_name: {task_accuracy, original_sp, updated_sp, family, size}}
    """
    judges = set(judge for judge, ref in judge_ref_accuracy.keys())
    result = {}
    
    # Extract judge map if it exists
    judge_map = {}
    if self_pref_data and '_judge_map' in self_pref_data:
        judge_map = self_pref_data.pop('_judge_map')
    
    for judge in judges:
        task_accuracy = aggregate_judge_across_references(judge_ref_accuracy, judge)
        family = extract_model_family(judge)
        size = extract_model_size(judge)
        
        result[judge] = {
            'task_accuracy': task_accuracy * 100,  # Convert to percentage
            'family': family,
            'size': size,
        }
        
        # Add self-preference data if available (with case-insensitive matching)
        sp_data = None
        if self_pref_data:
            # Try direct match first
            if judge in self_pref_data:
                sp_data = self_pref_data[judge]
            # Try case-insensitive match
            elif judge_map and judge.lower() in judge_map:
                canonical_judge = judge_map[judge.lower()]
                if canonical_judge in self_pref_data:
                    sp_data = self_pref_data[canonical_judge]
                    logger.debug(f"Matched judge '{judge}' to '{canonical_judge}' (case-insensitive)")
        
        if sp_data:
            result[judge]['original_sp'] = sp_data.get('original_sp')
            result[judge]['updated_sp'] = sp_data.get('updated_sp')
    
    # Filter out judges without valid self-preference data
    result = {k: v for k, v in result.items() if v.get('original_sp') is not None}
    
    return result


# ----------------------
# --- PLOTTING ---
# ----------------------

def load_self_pref_data(dataset_name):
    """
    Load self-preference data from aggregated results.
    
    Args:
        dataset_name (str): Name of dataset (e.g., 'math500', 'quality')
        
    Returns:
        dict: {judge_name: {original_sp, updated_sp}} or None if not found
    """
    # Try to find aggregated data file
    candidates = [
        f"judge_swap_null_verif_smoke2/{dataset_name}/analysis/aggregated_by_judge_reference.json",
        f"judge_swap_null_author_obfuscation/{dataset_name}/analysis/aggregated_by_judge_reference.json",
        f"judge_swap_null_dbg_results/{dataset_name}/analysis/aggregated_by_judge_reference.json",
    ]
    
    for candidate in candidates:
        if Path(candidate).exists():
            logger.info(f"Loading self-preference data from {candidate}")
            try:
                with open(candidate, 'r') as f:
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
                    judge_sp[judge]['updated_sp'].append(mean_diff * 100)  # Just mean_diff, not sum
                
                # Average across references
                result = {}
                for judge, sp_data in judge_sp.items():
                    result[judge] = {
                        'original_sp': np.mean(sp_data['original_sp']),
                        'updated_sp': np.mean(sp_data['updated_sp']),
                    }
                
                # Create case-insensitive mapping for matching judge names from different sources
                judge_map = {judge.lower(): judge for judge in result.keys()}
                
                logger.info(f"Loaded self-preference data for {len(result)} judges")
                logger.debug(f"Judge mapping (sample): {list(judge_map.items())[:3]}")
                
                # Return both the original and a case-insensitive lookup function
                result['_judge_map'] = judge_map
                return result
            except Exception as e:
                logger.warning(f"Failed to load self-preference from {candidate}: {e}")
    
    logger.warning(f"No self-preference data found for {dataset_name}")
    return None


def create_scatter_plot_for_paper(paper_dir, all_scatter_data):
    """
    Create scatter plot for a single paper_dir.
    
    Args:
        paper_dir (str): Name of the paper directory
        all_scatter_data (dict): {dataset_name: {judge: metrics}}
    """
    # Create subplots for datasets in this paper_dir
    n_datasets = len(all_scatter_data)
    n_cols = min(4, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    fig.suptitle(f'{paper_dir}: Updated Self-Preference Biases against Baselines', 
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
            # Only plot if we have valid SP data
            if metrics.get('original_sp') is None or metrics.get('updated_sp') is None:
                continue
                
            family = metrics['family']
            size = metrics['size']
            families_present.add(family)
            sizes_present.add(size)
            
            color = get_family_color(family)
            marker_size = size_to_marker_size(size)
            
            x = metrics['task_accuracy']
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
        
        ax.set_xlabel('Task Accuracy (%)', fontsize=10)
        ax.set_ylabel('Harmful Self-Preference Propensity', fontsize=10)
        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.7, linestyle='-', linewidth=0.8, color='gray')
        ax.set_xlim(30, 100)
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
    """Main function to generate scatter plots with correct task accuracy."""
    logger.info("Starting scatter plot generation with correct task accuracy")
    
    all_scatter_data_by_dir = {
        'author_obfuscation': {},
        'dbg-score-paper': {},
        'llm-sp-verif': {}
    }
    
    # Load author obfuscation (quality)
    logger.info("\n=== AUTHOR OBFUSCATION (QUALITY) ===")
    try:
        quality_dir = Path('~/project/author_obfuscation/data/quality/proxies').expanduser()
        judge_ref_accuracy = load_author_obf_data(str(quality_dir))
        self_pref = load_self_pref_data('quality')
        scatter_data = prepare_scatter_data(judge_ref_accuracy, self_pref)
        all_scatter_data_by_dir['author_obfuscation']['QUALITY'] = scatter_data
        logger.info(f"Loaded {len(scatter_data)} judges for Quality")
    except Exception as e:
        logger.error(f"Failed to load quality data: {e}")
    
    # Load DBG data (alpaca_eval, translation, truthfulness)
    dbg_datasets = ['alpaca_eval', 'translation', 'truthfulness']
    for dataset in dbg_datasets:
        logger.info(f"\n=== DBG ({dataset.upper()}) ===")
        try:
            dbg_dir = Path(f'~/project/dbg-score-paper/proxy_preference_data/{dataset}').expanduser()
            judge_ref_accuracy = load_dbg_data(str(dbg_dir), dataset)
            self_pref = load_self_pref_data(dataset)
            scatter_data = prepare_scatter_data(judge_ref_accuracy, self_pref)
            all_scatter_data_by_dir['dbg-score-paper'][dataset.upper()] = scatter_data
            logger.info(f"Loaded {len(scatter_data)} judges for {dataset}")
        except Exception as e:
            logger.error(f"Failed to load DBG {dataset} data: {e}")
    
    # Load verif data (math500, mmlu, mbpp-plus)
    verif_datasets = ['math500', 'mmlu', 'mbpp-plus']
    for dataset in verif_datasets:
        logger.info(f"\n=== VERIF ({dataset.upper()}) ===")
        try:
            # Need to glob family directories
            verif_base = Path(f'~/project/llm-sp-verif/{dataset}').expanduser()
            family_dirs = glob(f"{verif_base}/*/proxies")
            
            all_judge_ref_accuracy = {}
            for family_dir in family_dirs:
                judge_ref_accuracy = load_verif_data(family_dir, dataset)
                all_judge_ref_accuracy.update(judge_ref_accuracy)
            
            self_pref = load_self_pref_data(dataset)
            scatter_data = prepare_scatter_data(all_judge_ref_accuracy, self_pref)
            all_scatter_data_by_dir['llm-sp-verif'][dataset.upper()] = scatter_data
            logger.info(f"Loaded {len(scatter_data)} judges for {dataset}")
        except Exception as e:
            logger.error(f"Failed to load verif {dataset} data: {e}")
    
    logger.info(f"\n=== SUMMARY ===")
    for paper_dir, datasets in all_scatter_data_by_dir.items():
        logger.info(f"{paper_dir}:")
        for dataset, judges in datasets.items():
            valid_judges = {k: v for k, v in judges.items() if v.get('original_sp') is not None}
            logger.info(f"  {dataset}: {len(valid_judges)} judges with valid SP data")
    
    # Print sample data for verification
    logger.info("\n=== SAMPLE DATA ===")
    for paper_dir, datasets in all_scatter_data_by_dir.items():
        logger.info(f"{paper_dir}:")
        for dataset, judges in datasets.items():
            logger.info(f"  {dataset}:")
            for judge, metrics in list(judges.items())[:2]:
                if metrics.get('original_sp') is not None:
                    logger.info(f"    {judge}: accuracy={metrics['task_accuracy']:.1f}%, "
                               f"original_sp={metrics['original_sp']:.1f}%, updated_sp={metrics['updated_sp']:.1f}%, "
                               f"family={metrics['family']}, size={metrics['size']}B")
    
    # Generate scatter plots for each paper_dir
    logger.info("\n=== GENERATING PLOTS ===")
    output_dir = Path('scatter_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different formats
    png_dir = output_dir / 'png'
    pdf_dir = output_dir / 'pdf'
    svg_dir = output_dir / 'svg'
    png_dir.mkdir(exist_ok=True)
    pdf_dir.mkdir(exist_ok=True)
    svg_dir.mkdir(exist_ok=True)
    
    for paper_dir, all_scatter_data in all_scatter_data_by_dir.items():
        if not all_scatter_data:
            continue
            
        try:
            logger.info(f"Generating plot for {paper_dir}...")
            fig = create_scatter_plot_for_paper(paper_dir, all_scatter_data)
            
            # Sanitize paper_dir name for filename
            safe_name = paper_dir.replace('/', '_').replace(' ', '_')
            
            # Save PNG
            png_file = png_dir / f'{safe_name}_scatter.png'
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PNG to: {png_file}")
            
            # Save PDF
            pdf_file = pdf_dir / f'{safe_name}_scatter.pdf'
            plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PDF to: {pdf_file}")
            
            # Save SVG
            svg_file = svg_dir / f'{safe_name}_scatter.svg'
            plt.savefig(svg_file, format='svg', bbox_inches='tight')
            logger.info(f"Saved SVG to: {svg_file}")
            
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate plot for {paper_dir}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
