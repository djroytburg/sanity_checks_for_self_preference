
import json
import csv
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# ----------------------
# --- SETUP LOGGING ---
# ----------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_logs/create_entropy_gap_tables.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ----------------------
# --- DATA LOADING ---
# ----------------------

def load_all_data():
    """
    Load spread statistics and self-preference data for all datasets.
    
    Returns:
        list: List of dicts with all metrics per (dataset, judge, reference)
    """
    logger.info("Loading spread statistics...")
    with open('spread_statistics_analysis/all_spread_statistics.json', 'r') as f:
        spread_stats = json.load(f)
    
    # Convert to dict keyed by (dataset, judge) for easy lookup
    spread_dict = {}
    for entry in spread_stats:
        key = (entry['dataset'], entry['judge'])
        spread_dict[key] = entry
    
    logger.info(f"Loaded {len(spread_dict)} spread statistics entries")
    
    # Load self-preference data from aggregated files
    experiments = [
        ('judge_swap_null_verif_smoke2', ['math500', 'mmlu', 'mbpp-plus']),
        ('judge_swap_null_dbg_results', ['translation', 'truthfulness', 'alpaca_eval']),
        ('judge_swap_null_author_obfuscation', ['quality']),
    ]
    
    all_rows = []
    
    for exp_dir, datasets in experiments:
        for dataset in datasets:
            agg_file = Path(exp_dir) / dataset / 'analysis' / 'aggregated_by_judge_reference.json'
            
            if not agg_file.exists():
                logger.warning(f"File not found: {agg_file}")
                continue
            
            logger.info(f"Loading {agg_file}...")
            with open(agg_file, 'r') as f:
                agg_data = json.load(f)
            
            for entry in agg_data:
                judge = entry['judge']
                reference = entry['reference']
                
                # Get ILSP test statistics
                if 'tests' in entry and 'ilsp' in entry['tests']:
                    ilsp = entry['tests']['ilsp']
                else:
                    ilsp = entry
                
                mean_j_vs_r = ilsp.get('mean_j_vs_r', None)
                mean_diff = ilsp.get('mean_diff', None)
                
                if mean_j_vs_r is None or mean_diff is None:
                    continue
                
                # Get spread statistics
                spread_key = (dataset, judge)
                if spread_key not in spread_dict:
                    logger.debug(f"No spread stats for ({dataset}, {judge})")
                    continue
                
                spread_entry = spread_dict[spread_key]
                
                row = {
                    'dataset': dataset,
                    'judge': judge,
                    'reference': reference,
                    'entropy_gap': spread_entry.get('gap_entropy'),
                    'entropy_lsp': spread_entry.get('entropy_lsp'),
                    'entropy_ilsp': spread_entry.get('entropy_ilsp'),
                    'mean_gap': spread_entry.get('gap_mean'),
                    'mean_lsp': spread_entry.get('mean_lsp'),
                    'mean_ilsp': spread_entry.get('mean_ilsp'),
                    'n_lsp': spread_entry.get('n_lsp'),
                    'n_ilsp': spread_entry.get('n_ilsp'),
                    'original_sp': mean_j_vs_r * 100,  # Convert to percentage
                    'updated_sp': mean_diff * 100,  # Just mean_diff, not sum
                    'mean_diff': mean_diff * 100,  # Just the correction
                }
                
                all_rows.append(row)
    
    logger.info(f"Loaded {len(all_rows)} total (dataset, judge, reference) combinations")
    return all_rows


# ----------------------
# --- TABLE GENERATION ---
# ----------------------

def create_csv_table(data, output_path):
    """Create CSV table."""
    logger.info(f"Creating CSV table: {output_path}")
    
    df = pd.DataFrame(data)
    
    # Reorder columns
    cols = ['dataset', 'judge', 'reference', 'entropy_gap', 'entropy_lsp', 'entropy_ilsp',
            'mean_gap', 'mean_lsp', 'mean_ilsp', 'original_sp', 'updated_sp', 'mean_diff',
            'n_lsp', 'n_ilsp']
    df = df[cols]
    
    # Sort by dataset, judge, reference
    df = df.sort_values(['dataset', 'judge', 'reference'])
    
    df.to_csv(output_path, index=False, float_format='%.4f')
    logger.info(f"Saved CSV with {len(df)} rows")


def create_json_table(data, output_path):
    """Create JSON table."""
    logger.info(f"Creating JSON table: {output_path}")
    
    # Sort by dataset, judge, reference
    sorted_data = sorted(data, key=lambda x: (x['dataset'], x['judge'], x['reference']))
    
    with open(output_path, 'w') as f:
        json.dump(sorted_data, f, indent=2)
    
    logger.info(f"Saved JSON with {len(sorted_data)} entries")


def create_latex_table_per_dataset(data, output_dir):
    """Create LaTeX tables, one per dataset."""
    logger.info(f"Creating LaTeX tables in: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by dataset
    datasets = {}
    for row in data:
        dataset = row['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(row)
    
    for dataset, rows in datasets.items():
        # Sort by judge, reference
        rows = sorted(rows, key=lambda x: (x['judge'], x['reference']))
        
        latex_file = output_dir / f'entropy_gap_table_{dataset}.tex'
        
        with open(latex_file, 'w') as f:
            # Write table header
            f.write('\\begin{table}[h]\n')
            f.write('\\centering\n')
            f.write('\\small\n')
            f.write(f'\\caption{{Entropy Gap vs Self-Preference Metrics for {dataset.upper()}}}\n')
            f.write(f'\\label{{tab:entropy_gap_{dataset}}}\n')
            f.write('\\begin{tabular}{llrrrrrr}\n')
            f.write('\\toprule\n')
            f.write('Judge & Reference & $\\Delta H$ & $H_{LSP}$ & $H_{ILSP}$ & Original SP & Updated SP & $\\Delta$ SP \\\\\n')
            f.write('\\midrule\n')
            
            # Write rows
            for row in rows:
                judge = row['judge'].replace('_', '\\_')
                reference = row['reference'].replace('_', '\\_')
                entropy_gap = row['entropy_gap']
                entropy_lsp = row['entropy_lsp']
                entropy_ilsp = row['entropy_ilsp']
                original_sp = row['original_sp']
                updated_sp = row['updated_sp']
                mean_diff = row['mean_diff']
                
                f.write(f'{judge} & {reference} & {entropy_gap:.3f} & {entropy_lsp:.3f} & {entropy_ilsp:.3f} & ')
                f.write(f'{original_sp:.1f} & {updated_sp:.1f} & {mean_diff:.1f} \\\\\n')
            
            f.write('\\bottomrule\n')
            f.write('\\end{tabular}\n')
            f.write('\\end{table}\n')
        
        logger.info(f"Saved LaTeX table for {dataset} with {len(rows)} rows")


def create_aggregated_latex_table(data, output_path):
    """Create aggregated LaTeX table (judge-level averages)."""
    logger.info(f"Creating aggregated LaTeX table: {output_path}")
    
    # Group by (dataset, judge)
    aggregated = {}
    for row in data:
        key = (row['dataset'], row['judge'])
        if key not in aggregated:
            aggregated[key] = {
                'dataset': row['dataset'],
                'judge': row['judge'],
                'entropy_gap': row['entropy_gap'],  # Same for all refs
                'entropy_lsp': row['entropy_lsp'],
                'entropy_ilsp': row['entropy_ilsp'],
                'mean_gap': row['mean_gap'],
                'original_sp_list': [],
                'updated_sp_list': [],
                'mean_diff_list': [],
                'n_refs': 0
            }
        
        aggregated[key]['original_sp_list'].append(row['original_sp'])
        aggregated[key]['updated_sp_list'].append(row['updated_sp'])
        aggregated[key]['mean_diff_list'].append(row['mean_diff'])
        aggregated[key]['n_refs'] += 1
    
    # Compute averages
    agg_rows = []
    for key, data_dict in aggregated.items():
        agg_rows.append({
            'dataset': data_dict['dataset'],
            'judge': data_dict['judge'],
            'entropy_gap': data_dict['entropy_gap'],
            'entropy_lsp': data_dict['entropy_lsp'],
            'entropy_ilsp': data_dict['entropy_ilsp'],
            'mean_gap': data_dict['mean_gap'],
            'original_sp': np.mean(data_dict['original_sp_list']),
            'updated_sp': np.mean(data_dict['updated_sp_list']),
            'mean_diff': np.mean(data_dict['mean_diff_list']),
            'n_refs': data_dict['n_refs']
        })
    
    # Sort by dataset, judge
    agg_rows = sorted(agg_rows, key=lambda x: (x['dataset'], x['judge']))
    
    with open(output_path, 'w') as f:
        # Write table header
        f.write('\\begin{table}[h]\n')
        f.write('\\centering\n')
        f.write('\\small\n')
        f.write('\\caption{Entropy Gap vs Self-Preference Metrics (Aggregated by Judge)}\n')
        f.write('\\label{tab:entropy_gap_aggregated}\n')
        f.write('\\begin{tabular}{llrrrrrrrr}\n')
        f.write('\\toprule\n')
        f.write('Dataset & Judge & $\\Delta H$ & $H_{LSP}$ & $H_{ILSP}$ & $\\Delta\\mu^*$ & Orig SP & Upd SP & $\\Delta$ SP & $n_{refs}$ \\\\\n')
        f.write('\\midrule\n')
        
        current_dataset = None
        for row in agg_rows:
            if row['dataset'] != current_dataset:
                if current_dataset is not None:
                    f.write('\\midrule\n')
                current_dataset = row['dataset']
            
            dataset = row['dataset'].replace('_', '\\_')
            judge = row['judge'].replace('_', '\\_')
            entropy_gap = row['entropy_gap']
            entropy_lsp = row['entropy_lsp']
            entropy_ilsp = row['entropy_ilsp']
            mean_gap = row['mean_gap']
            original_sp = row['original_sp']
            updated_sp = row['updated_sp']
            mean_diff = row['mean_diff']
            n_refs = row['n_refs']
            
            f.write(f'{dataset} & {judge} & {entropy_gap:.3f} & {entropy_lsp:.3f} & {entropy_ilsp:.3f} & ')
            f.write(f'{mean_gap:.3f} & {original_sp:.1f} & {updated_sp:.1f} & {mean_diff:.1f} & {n_refs} \\\\\n')
        
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    
    logger.info(f"Saved aggregated LaTeX table with {len(agg_rows)} rows")


# ----------------------
# --- MAIN EXECUTION ---
# ----------------------

def main():
    """Main function."""
    logger.info("Starting entropy gap table generation")
    
    # Load all data
    data = load_all_data()
    
    if not data:
        logger.error("No data loaded, exiting")
        return
    
    # Create output directory
    output_dir = Path('entropy_gap_tables')
    output_dir.mkdir(exist_ok=True)
    
    # Create CSV table
    csv_path = output_dir / 'entropy_gap_full.csv'
    create_csv_table(data, csv_path)
    
    # Create JSON table
    json_path = output_dir / 'entropy_gap_full.json'
    create_json_table(data, json_path)
    
    # Create LaTeX tables per dataset
    latex_dir = output_dir / 'latex'
    create_latex_table_per_dataset(data, latex_dir)
    
    # Create aggregated LaTeX table
    agg_latex_path = output_dir / 'latex' / 'entropy_gap_aggregated.tex'
    create_aggregated_latex_table(data, agg_latex_path)
    
    logger.info("\n✓ Entropy gap table generation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  - CSV: entropy_gap_full.csv")
    logger.info(f"  - JSON: entropy_gap_full.json")
    logger.info(f"  - LaTeX: latex/*.tex")


if __name__ == '__main__':
    main()
