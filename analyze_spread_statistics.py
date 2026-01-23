# analyze_spread_statistics.py: ANALYZE LSP VS ILSP SPREAD STATISTICS
# Created: 2026-01-18, 01:20 EST
# Last Modified: 2026-01-18, 01:20 EST

import json
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# --- LOGGING SETUP ---
# ----------------------

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('file_logs/analyze_spread_statistics.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)


# ----------------------
# --- DATA LOADING ---
# ----------------------

def load_jvsr_from_cache(cache_dir, judge, reference):
    """
    Load J_vs_R data from cache files and calculate p_self for each example.
    
    Args:
        cache_dir (Path): Base cache directory
        judge (str): Judge name
        reference (str): Reference name
        
    Returns:
        dict: {example_id: {'p_self': float, 'category': str}}
    """
    data = defaultdict(lambda: {'game1': [], 'game2': []})
    
    # Load game1 and game2
    for game in ['J_vs_R_game1', 'J_vs_R_game2']:
        cache_file = cache_dir / judge / reference / f'{game}.jsonl'
        
        if not cache_file.exists():
            continue
        
        with open(cache_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                example_id = entry.get('example_id')
                category = entry.get('category', 'unknown')
                
                # prob_response1 for game1, prob_response2 for game2
                if game == 'J_vs_R_game1':
                    prob = entry.get('prob_response1')
                else:
                    prob = entry.get('prob_response2')
                
                if prob is not None:
                    if game == 'J_vs_R_game1':
                        data[example_id]['game1'].append({'prob': prob, 'category': category})
                    else:
                        data[example_id]['game2'].append({'prob': prob, 'category': category})
    
    # Calculate p_self for each example
    result = {}
    for example_id, games in data.items():
        if not (games['game1'] and games['game2']):
            continue
        
        p_game1 = np.mean([e['prob'] for e in games['game1']])
        p_game2 = np.mean([e['prob'] for e in games['game2']])
        p_self = (p_game1 + p_game2) / 2
        
        category = games['game1'][0]['category']
        
        result[example_id] = {
            'p_self': p_self,
            'category': category
        }
    
    return result


def load_author_obf_jvsr(judge, reference=None):
    """Load J_vs_R data from author_obfuscation JSON files."""
    judge_dir = Path(f'judge_swap_null_author_obfuscation/quality/{judge}')
    data = defaultdict(lambda: {'game1': [], 'game2': []})

    if not judge_dir.exists():
        return {}

    for file_path in judge_dir.glob('judge_swap_*.json'):
        with open(file_path, 'r') as f:
            cache_file = json.load(f)

        metadata = cache_file.get('metadata', {})
        file_judge = metadata.get('judge')
        file_reference = metadata.get('reference')

        if file_judge != judge:
            continue
        if reference is not None and file_reference != reference:
            continue

        for game_key in ['J_vs_R_game1', 'J_vs_R_game2']:
            game_data = cache_file.get(game_key, [])

            for entry in game_data:
                example_id = entry.get('example_id')
                category = entry.get('category', 'unknown')
                prob = entry.get('prob_A') if game_key == 'J_vs_R_game1' else entry.get('prob_B')

                if prob is not None:
                    if game_key == 'J_vs_R_game1':
                        data[example_id]['game1'].append({'prob': prob, 'category': category})
                    else:
                        data[example_id]['game2'].append({'prob': prob, 'category': category})

    # Calculate p_self
    result = {}
    for example_id, games in data.items():
        if not (games['game1'] and games['game2']):
            continue

        p_game1 = np.mean([e['prob'] for e in games['game1']])
        p_game2 = np.mean([e['prob'] for e in games['game2']])
        p_self = (p_game1 + p_game2) / 2

        category = games['game1'][0]['category']

        result[example_id] = {
            'p_self': p_self,
            'category': category
        }

    return result


def load_panickserry_results_jvsr(dataset, judge, reference=None):
    """
    Load J_vs_R data from Panickserry results cache files.

    Args:
        dataset (str): Either 'cnn' or 'xsum'
        judge (str): Judge name
        reference (str, optional): Reference name. If None, aggregate across all references.

    Returns:
        dict: {example_id: {'p_self': float, 'category': str}}
    """
    if dataset == 'cnn':
        cache_base = Path('panickserry_results/cnn_results/cnn/cache')
    elif dataset == 'xsum':
        cache_base = Path('panickserry_results/xsum_result/xsum/cache')
    else:
        logger.warning(f"Unknown dataset: {dataset}")
        return {}

    data = defaultdict(lambda: {'game1': [], 'game2': []})

    judge_dir = cache_base / judge
    if not judge_dir.exists():
        return {}

    # Get all reference directories
    if reference is not None:
        reference_dirs = [judge_dir / reference] if (judge_dir / reference).exists() else []
    else:
        reference_dirs = [d for d in judge_dir.iterdir() if d.is_dir()]

    for ref_dir in reference_dirs:
        # Load game1 and game2
        for game in ['J_vs_R_game1', 'J_vs_R_game2']:
            cache_file = ref_dir / f'{game}.jsonl'

            if not cache_file.exists():
                continue

            with open(cache_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    example_id = entry.get('example_id')
                    category = entry.get('category', 'unknown')

                    # prob_response1 for game1, prob_response2 for game2
                    if game == 'J_vs_R_game1':
                        prob = entry.get('prob_response1')
                    else:
                        prob = entry.get('prob_response2')

                    if prob is not None:
                        if game == 'J_vs_R_game1':
                            data[example_id]['game1'].append({'prob': prob, 'category': category})
                        else:
                            data[example_id]['game2'].append({'prob': prob, 'category': category})

    # Calculate p_self for each example
    result = {}
    for example_id, games in data.items():
        if not (games['game1'] and games['game2']):
            continue

        p_game1 = np.mean([e['prob'] for e in games['game1']])
        p_game2 = np.mean([e['prob'] for e in games['game2']])
        p_self = (p_game1 + p_game2) / 2

        category = games['game1'][0]['category']

        result[example_id] = {
            'p_self': p_self,
            'category': category
        }

    return result


# ----------------------
# --- STATISTICS ---
# ----------------------

def calculate_spread_stats(dataset, judge, data_by_example):
    """
    Calculate spread statistics for LSP and ILSP.
    
    Args:
        dataset (str): Dataset name
        judge (str): Judge name
        data_by_example (dict): Example data with p_self and category
        
    Returns:
        dict: Statistics for this dataset-judge combination
    """
    lsp_values = []
    ilsp_values = []
    
    for example_id, data in data_by_example.items():
        p_self = data['p_self']
        category = data['category']
        
        if category == 'lsp':
            lsp_values.append(p_self)
        elif category == 'ilsp':
            ilsp_values.append(p_self)
    
    stats = {
        'dataset': dataset,
        'judge': judge,
        'n_lsp': len(lsp_values),
        'n_ilsp': len(ilsp_values)
    }
    
    # Helper function to calculate entropy
    def calculate_entropy(probs):
        # Binary entropy for self-preference probabilities
        probs = np.array(probs)
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        entropy_values = -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)
        return np.mean(entropy_values)
    
    if lsp_values:
        stats['mean_lsp'] = np.mean(lsp_values)
        stats['var_lsp'] = np.var(lsp_values)
        stats['std_lsp'] = np.std(lsp_values)
        stats['entropy_lsp'] = calculate_entropy(lsp_values)
    else:
        stats['mean_lsp'] = np.nan
        stats['var_lsp'] = np.nan
        stats['std_lsp'] = np.nan
        stats['entropy_lsp'] = np.nan
    
    if ilsp_values:
        stats['mean_ilsp'] = np.mean(ilsp_values)
        stats['var_ilsp'] = np.var(ilsp_values)
        stats['std_ilsp'] = np.std(ilsp_values)
        stats['entropy_ilsp'] = calculate_entropy(ilsp_values)
    else:
        stats['mean_ilsp'] = np.nan
        stats['var_ilsp'] = np.nan
        stats['std_ilsp'] = np.nan
        stats['entropy_ilsp'] = np.nan
    
    # Calculate gaps
    # gap_mean = mean_lsp - (1 - mean_ilsp) = mean_lsp + mean_ilsp - 1
    if not np.isnan(stats['mean_lsp']) and not np.isnan(stats['mean_ilsp']):
        stats['gap_mean'] = stats['mean_lsp'] - (1 - stats['mean_ilsp'])
    else:
        stats['gap_mean'] = np.nan
    
    if not np.isnan(stats['var_lsp']) and not np.isnan(stats['var_ilsp']):
        stats['gap_var'] = stats['var_ilsp'] - stats['var_lsp']
    else:
        stats['gap_var'] = np.nan
    
    if not np.isnan(stats['std_lsp']) and not np.isnan(stats['std_ilsp']):
        stats['gap_std'] = stats['std_ilsp'] - stats['std_lsp']
    else:
        stats['gap_std'] = np.nan
    
    if not np.isnan(stats['entropy_lsp']) and not np.isnan(stats['entropy_ilsp']):
        stats['gap_entropy'] = stats['entropy_ilsp'] - stats['entropy_lsp']
    else:
        stats['gap_entropy'] = np.nan
    
    return stats


# ----------------------
# --- LATEX TABLES ---
# ----------------------

def generate_latex_table(stats_list, dataset_name):
    """Generate LaTeX table for a dataset."""
    
    # Sort by judge name
    stats_list = sorted(stats_list, key=lambda x: x['judge'])
    
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Spread Statistics: " + dataset_name.replace('_', '\\_') + "}")
    latex.append("\\label{tab:spread_" + dataset_name + "}")
    latex.append("\\begin{tabular}{l c c c c c c}")
    latex.append("\\hline")
    latex.append("Judge & $\\mu_{LSP}$ & $\\sigma^2_{LSP}$ & $\\mu_{ILSP}$ & $\\sigma^2_{ILSP}$ & $\\Delta\\mu^*$ & $\\Delta\\sigma^2$ \\\\")
    latex.append("\\hline")
    
    # Add rows
    for stats in stats_list:
        judge = stats['judge'].replace('_', '\\_')
        
        mean_lsp = f"{stats['mean_lsp']:.3f}" if not np.isnan(stats['mean_lsp']) else "---"
        var_lsp = f"{stats['var_lsp']:.3f}" if not np.isnan(stats['var_lsp']) else "---"
        mean_ilsp = f"{stats['mean_ilsp']:.3f}" if not np.isnan(stats['mean_ilsp']) else "---"
        var_ilsp = f"{stats['var_ilsp']:.3f}" if not np.isnan(stats['var_ilsp']) else "---"
        gap_mean = f"{stats['gap_mean']:.3f}" if not np.isnan(stats['gap_mean']) else "---"
        gap_var = f"{stats['gap_var']:.3f}" if not np.isnan(stats['gap_var']) else "---"
        
        latex.append(f"{judge} & {mean_lsp} & {var_lsp} & {mean_ilsp} & {var_ilsp} & {gap_mean} & {gap_var} \\\\")
    
    # Calculate macro average
    valid_mean_lsp = [s['mean_lsp'] for s in stats_list if not np.isnan(s['mean_lsp'])]
    valid_var_lsp = [s['var_lsp'] for s in stats_list if not np.isnan(s['var_lsp'])]
    valid_mean_ilsp = [s['mean_ilsp'] for s in stats_list if not np.isnan(s['mean_ilsp'])]
    valid_var_ilsp = [s['var_ilsp'] for s in stats_list if not np.isnan(s['var_ilsp'])]
    valid_gap_mean = [s['gap_mean'] for s in stats_list if not np.isnan(s['gap_mean'])]
    valid_gap_var = [s['gap_var'] for s in stats_list if not np.isnan(s['gap_var'])]
    
    macro_mean_lsp = f"{np.mean(valid_mean_lsp):.3f}" if valid_mean_lsp else "---"
    macro_var_lsp = f"{np.mean(valid_var_lsp):.3f}" if valid_var_lsp else "---"
    macro_mean_ilsp = f"{np.mean(valid_mean_ilsp):.3f}" if valid_mean_ilsp else "---"
    macro_var_ilsp = f"{np.mean(valid_var_ilsp):.3f}" if valid_var_ilsp else "---"
    macro_gap_mean = f"{np.mean(valid_gap_mean):.3f}" if valid_gap_mean else "---"
    macro_gap_var = f"{np.mean(valid_gap_var):.3f}" if valid_gap_var else "---"
    
    latex.append("\\hline")
    latex.append(f"Macro Avg & {macro_mean_lsp} & {macro_var_lsp} & {macro_mean_ilsp} & {macro_var_ilsp} & {macro_gap_mean} & {macro_gap_var} \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


# ----------------------
# --- VISUALIZATIONS ---
# ----------------------

def create_spread_visualization(all_stats, output_dir):
    """Create visualizations of spread statistics."""
    
    # Prepare data for plotting
    datasets = sorted(set(s['dataset'] for s in all_stats))
    judges = sorted(set(s['judge'] for s in all_stats))
    judge_colors = dict(zip(judges, plt.cm.tab20(np.linspace(0, 1, len(judges)))))
    
    # 1. Entropy gap scatter plot, colored by judge, sized by sample count
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for dataset_idx, dataset in enumerate(datasets):
        dataset_stats = [s for s in all_stats if s['dataset'] == dataset]
        
        for stats in dataset_stats:
            if not np.isnan(stats['gap_entropy']):
                # Size proportional to total examples
                size = (stats['n_lsp'] + stats['n_ilsp']) / 5  # Scale for visibility
                color = judge_colors[stats['judge']]
                ax.scatter(stats['gap_entropy'], dataset_idx, s=size, alpha=0.7, color=color)
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel('Entropy Gap (H(ILSP) - H(LSP))', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title('Entropy Gap Distribution by Dataset (colored by judge, sized by sample count)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend for judges
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=judge_colors[judge], label=judge) for judge in judges]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        output_file = output_dir / f'spread_entropy_gap_scatter.{fmt}'
        if fmt == 'svg':
            plt.savefig(output_file, format='svg', bbox_inches='tight')
        else:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    plt.close()
    # logger.info(f"Saved entropy gap scatter plot")
    
    # 2. Heatmap of gap means by dataset and judge
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create matrix
    judges = sorted(set(s['judge'] for s in all_stats))
    gap_mean_matrix = np.full((len(datasets), len(judges)), np.nan)
    
    for i, dataset in enumerate(datasets):
        for j, judge in enumerate(judges):
            matching = [s for s in all_stats if s['dataset'] == dataset and s['judge'] == judge]
            if matching and not np.isnan(matching[0]['gap_mean']):
                gap_mean_matrix[i, j] = matching[0]['gap_mean']
    
    sns.heatmap(gap_mean_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=[j.replace('-', '\n') for j in judges],
                yticklabels=datasets, ax=ax, cbar_kws={'label': '$(1-\\mu_{ILSP}) - (1-\\mu_{LSP})$'})
    ax.set_title('Mean Gap Heatmap (1-transformed)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Judge', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    
    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        output_file = output_dir / f'spread_gap_mean_heatmap.{fmt}'
        if fmt == 'svg':
            plt.savefig(output_file, format='svg', bbox_inches='tight')
        else:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    # logger.info(f"Saved gap mean heatmap")
    
    # 3. Bar chart of mean gap by dataset
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dataset_gap_means = []
    dataset_gap_stds = []
    
    for dataset in datasets:
        dataset_stats = [s for s in all_stats if s['dataset'] == dataset]
        valid_gaps = [s['gap_mean'] for s in dataset_stats if not np.isnan(s['gap_mean'])]
        
        if valid_gaps:
            dataset_gap_means.append(np.mean(valid_gaps))
            dataset_gap_stds.append(np.std(valid_gaps))
        else:
            dataset_gap_means.append(0)
            dataset_gap_stds.append(0)
    
    x_pos = np.arange(len(datasets))
    bars = ax.bar(x_pos, dataset_gap_means, yerr=dataset_gap_stds, capsize=5, alpha=0.7)
    
    # Color bars based on sign
    for i, bar in enumerate(bars):
        if dataset_gap_means[i] > 0:
            bar.set_color('blue')  # Positive means LSP prefers self less in 1-space
        else:
            bar.set_color('red')  # Negative means ILSP prefers self less in 1-space
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Mean Gap ($(1-\\mu_{ILSP}) - (1-\\mu_{LSP})$)', fontsize=12)
    ax.set_title('Average Mean Gap by Dataset (1-transformed)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    for fmt in ['png', 'pdf', 'svg']:
        output_file = output_dir / f'spread_gap_mean_by_dataset.{fmt}'
        if fmt == 'svg':
            plt.savefig(output_file, format='svg', bbox_inches='tight')
        else:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    # logger.info(f"Saved gap mean by dataset bar chart")


# ----------------------
# --- MAIN EXECUTION ---
# ----------------------

def main():
    """Main execution function."""
    # logger.info("Starting spread statistics analysis")
    
    # Create output directories
    output_dir = Path('spread_statistics_analysis')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'latex').mkdir(exist_ok=True)
    
    all_stats = []
    
    
    # logger.info("\n=== Processing verif_smoke2 ===")
    verif_dir = Path('judge_swap_null_verif_cot')
    for dataset_dir in verif_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset = dataset_dir.name
        # logger.info(f"Processing dataset: {dataset}")
        
        cache_base = dataset_dir / 'cache'
        if not cache_base.exists():
            continue
        
        for judge_dir in cache_base.iterdir():
            if not judge_dir.is_dir():
                continue
            
            judge = judge_dir.name
            
            # Aggregate across all references
            all_examples = {}
            references = [d.name for d in judge_dir.iterdir() if d.is_dir()]
            
            for reference in references:
                data = load_jvsr_from_cache(cache_base, judge, reference)
                all_examples.update(data)
            
            if all_examples:
                stats = calculate_spread_stats(dataset, judge, all_examples)
                all_stats.append(stats)
                # logger.info(f"  {judge}: n_lsp={stats['n_lsp']}, n_ilsp={stats['n_ilsp']}, gap_mean={stats['gap_mean']:.4f}")
    
    
    # logger.info("\n=== Processing dbg_results ===")
    # dbg_dir = Path('judge_swap_null_dbg_results')
    # for dataset_dir in dbg_dir.iterdir():
    #     if not dataset_dir.is_dir():
    #         continue
        
    #     dataset = dataset_dir.name
    #     logger.info(f"Processing dataset: {dataset}")
        
    #     cache_base = dataset_dir / 'cache'
    #     if not cache_base.exists():
    #         continue
        
    #     for judge_dir in cache_base.iterdir():
    #         if not judge_dir.is_dir():
    #             continue
            
    #         judge = judge_dir.name
            
    #         all_examples = {}
    #         references = [d.name for d in judge_dir.iterdir() if d.is_dir()]
            
    #         for reference in references:
    #             data = load_jvsr_from_cache(cache_base, judge, reference)
    #             all_examples.update(data)
            
    #         if all_examples:
    #             stats = calculate_spread_stats(dataset, judge, all_examples)
    #             all_stats.append(stats)
    #             logger.info(f"  {judge}: n_lsp={stats['n_lsp']}, n_ilsp={stats['n_ilsp']}, gap_mean={stats['gap_mean']:.4f}")
    
    # Process author_obfuscation
    # logger.info("\n=== Processing author_obfuscation ===")
    # author_obf_dir = Path('judge_swap_null_author_obfuscation/quality')
    # dataset = 'quality'

    # for judge_dir in author_obf_dir.iterdir():
    #     if not judge_dir.is_dir() or judge_dir.name == 'analysis':
    #         continue

    #     judge = judge_dir.name

    #     all_examples = load_author_obf_jvsr(judge, reference=None)

    #     if all_examples:
    #         stats = calculate_spread_stats(dataset, judge, all_examples)
    #         all_stats.append(stats)
    #         logger.info(f"  {judge}: n_lsp={stats['n_lsp']}, n_ilsp={stats['n_ilsp']}, gap_mean={stats['gap_mean']:.4f}")

    # Process Panickserry CNN results
    # logger.info("\n=== Processing Panickserry CNN results ===")
    # cnn_cache_dir = Path('panickserry_results/cnn_results/cnn/cache')
    # dataset = 'cnn'

    # if cnn_cache_dir.exists():
    #     for judge_dir in cnn_cache_dir.iterdir():
    #         if not judge_dir.is_dir():
    #             continue

    #         judge = judge_dir.name

    #         all_examples = load_panickserry_results_jvsr(dataset, judge, reference=None)

    #         if all_examples:
    #             stats = calculate_spread_stats(dataset, judge, all_examples)
    #             all_stats.append(stats)
    #             # logger.info(f"  {judge}: n_lsp={stats['n_lsp']}, n_ilsp={stats['n_ilsp']}, gap_mean={stats['gap_mean']:.4f}")

    # # Process Panickserry XSUM results
    # # logger.info("\n=== Processing Panickserry XSUM results ===")
    # xsum_cache_dir = Path('panickserry_results/xsum_result/xsum/cache')
    # dataset = 'xsum'

    # if xsum_cache_dir.exists():
    #     for judge_dir in xsum_cache_dir.iterdir():
    #         if not judge_dir.is_dir():
    #             continue

    #         judge = judge_dir.name

    #         all_examples = load_panickserry_results_jvsr(dataset, judge, reference=None)

    #         if all_examples:
    #             stats = calculate_spread_stats(dataset, judge, all_examples)
    #             all_stats.append(stats)
    #             # logger.info(f"  {judge}: n_lsp={stats['n_lsp']}, n_ilsp={stats['n_ilsp']}, gap_mean={stats['gap_mean']:.4f}")

    # Generate LaTeX tables by dataset
    # logger.info("\n=== Generating LaTeX tables ===")
    datasets = sorted(set(s['dataset'] for s in all_stats))
    
    for dataset in datasets:
        dataset_stats = [s for s in all_stats if s['dataset'] == dataset]
        latex_table = generate_latex_table(dataset_stats, dataset)
        
        output_file = output_dir / 'latex' / f'spread_stats_{dataset}.tex'
        with open(output_file, 'w') as f:
            f.write(latex_table)
        # logger.info(f"Saved LaTeX table for {dataset}")
    
    # Save all stats to JSON
    output_file = output_dir / 'all_spread_statistics.json'
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    # logger.info(f"Saved all statistics to JSON")
    
    # # Create visualizations
    # logger.info("\n=== Creating visualizations ===")
    create_spread_visualization(all_stats, output_dir)
    
    # logger.info("\n=== Spread statistics analysis complete ===")
    # logger.info(f"Output directory: {output_dir}")
    # logger.info(f"Total dataset-judge combinations: {len(all_stats)}")


if __name__ == "__main__":
    main()
