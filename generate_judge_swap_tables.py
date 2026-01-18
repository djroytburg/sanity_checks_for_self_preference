# generate_judge_swap_tables.py: Generate TeX tables for judge swap experiments
# Written by: Dani
# Created: January 17, 2026, 10:30 AM EST
# Last Modified: January 17, 2026, 10:30 AM EST

import json
import os
from pathlib import Path
from collections import defaultdict
import logging

# ----------------------
# --- SETUP LOGGING ---
# ----------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_logs/generate_judge_swap_tables.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ----------------------
# --- UTILITIES ---
# ----------------------

def extract_model_family(model_name):
    """
    Extract model family from model name for grouping.
    
    Args:
        model_name (str): Full model name
        
    Returns:
        str: Model family prefix
    """
    families = {
        'gemma': 'gemma',
        'gpt': 'gpt',
        'llama': 'llama',
        'mistral': 'mistral',
        'phi': 'phi',
        'qwen': 'qwen',
        'deepseek': 'deepseek',
        'meta-llama': 'llama'
    }
    
    model_lower = model_name.lower()
    for key, family in families.items():
        if model_lower.startswith(key):
            return family
    return 'other'


def sort_models_by_family(models):
    """
    Sort models by family and then alphabetically within family.
    
    Args:
        models (list): List of model names
        
    Returns:
        list: Sorted model names
    """
    model_tuples = [(extract_model_family(m), m) for m in models]
    model_tuples.sort(key=lambda x: (x[0], x[1]))
    return [m for _, m in model_tuples]


def format_percentage(value):
    """
    Format value as percentage with 0.1% precision.
    
    Args:
        value (float): Value to format (assumed to be in [0, 1] range)
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.1f}"


def should_bold(entry):
    """
    Check if entry should be bolded based on statistical tests.
    
    Args:
        entry (dict): Entry containing test results
        
    Returns:
        bool: True if should be bolded
    """
    # Check if data is nested under 'tests' key
    if 'tests' in entry and 'ilsp' in entry['tests']:
        ilsp_tests = entry['tests']['ilsp']
    else:
        # Data is directly in the entry (quality dataset format)
        ilsp_tests = entry
    
    ks_pvalue = ilsp_tests.get('ks_pvalue', 0)
    ttest_pvalue = ilsp_tests.get('ttest_one_sided_p', 0)
    
    # Bold if EITHER p-value > 0.05
    return ks_pvalue > 0.05 or ttest_pvalue > 0.05


def load_aggregated_data(file_path):
    """
    Load aggregated data from JSON file.
    
    Args:
        file_path (str): Path to aggregated_by_judge_reference.json
        
    Returns:
        list: List of entries from the JSON file
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# ----------------------
# --- CORE LOGIC ---
# ----------------------

def create_table_data(data):
    """
    Create table data structure from aggregated results.
    
    Args:
        data (list): List of judge/reference entries
        
    Returns:
        tuple: (judges, references, table_dict) where table_dict maps (judge, ref) to (value, should_bold)
    """
    judges = set()
    references = set()
    table_dict = {}
    
    for entry in data:
        judge = entry['judge']
        reference = entry['reference']
        judges.add(judge)
        references.add(reference)
        
        # Extract ILSP test results - handle both nested and flat formats
        if 'tests' in entry and 'ilsp' in entry['tests']:
            # Nested format (math500, mmlu, etc.)
            ilsp = entry['tests']['ilsp']
        else:
            # Flat format (quality dataset)
            ilsp = entry
        
        mean_diff = ilsp.get('mean_diff', 0)
        mean_j_vs_r = ilsp.get('mean_j_vs_r', 0)
        
        # Format as "mean_diff (mean_j_vs_r)"
        value_str = f"{format_percentage(mean_diff)} ({format_percentage(mean_j_vs_r)})"
        bold = should_bold(entry)
        
        table_dict[(judge, reference)] = (value_str, bold)
    
    # Sort judges and references by family
    judges = sort_models_by_family(list(judges))
    references = sort_models_by_family(list(references))
    
    return judges, references, table_dict


def generate_latex_table(judges, references, table_dict, dataset_name, paper_name):
    """
    Generate LaTeX table code.
    
    Args:
        judges (list): Sorted list of judge models
        references (list): Sorted list of reference models
        table_dict (dict): Map of (judge, ref) to (value, should_bold)
        dataset_name (str): Name of dataset
        paper_name (str): Name of paper/experiment
        
    Returns:
        str: LaTeX table code
    """
    # Create short names for column headers
    ref_short_names = {ref: ref.replace('-', '-\\allowbreak ') for ref in references}
    
    # Start table
    lines = []
    lines.append(f"% Table for {dataset_name} ({paper_name})")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Judge Swap Results: {dataset_name} ({paper_name})}}")
    lines.append(f"\\label{{tab:{paper_name.replace('_', '-')}_{dataset_name}}}")
    
    # Column specification: first column for judges, then one column per reference
    num_cols = len(references) + 1
    col_spec = "l" + "c" * len(references)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    
    # Header row
    header = "Judge & " + " & ".join([ref_short_names[r] for r in references]) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    
    # Data rows
    for judge in judges:
        row_values = [judge.replace('_', '\\_')]
        for ref in references:
            if (judge, ref) in table_dict:
                value_str, bold = table_dict[(judge, ref)]
                if bold:
                    cell = f"\\textbf{{{value_str}}}"
                else:
                    cell = value_str
                row_values.append(cell)
            else:
                row_values.append("--")
        
        row = " & ".join(row_values) + " \\\\"
        lines.append(row)
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


# ----------------------
# --- MAIN EXECUTION ---
# ----------------------

def main():
    """
    Main function to generate all judge swap tables.
    """
    logger.info("Starting judge swap table generation")
    
    # Define directories and datasets
    experiments = [
        ('judge_swap_null_verif_smoke2', ['mbpp-plus', 'math500', 'mmlu']),
        ('judge_swap_null_author_obfuscation', ['quality']),
        ('judge_swap_null_dbg_results', ['alpaca_eval', 'translation', 'truthfulness'])
    ]
    
    # Create output directory
    output_dir = Path('judge_swap_tables')
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Process each experiment and dataset
    for paper_dir, datasets in experiments:
        paper_name = paper_dir
        logger.info(f"Processing paper: {paper_name}")
        
        for dataset in datasets:
            logger.info(f"  Processing dataset: {dataset}")
            
            # Construct path to aggregated data
            data_file = Path(paper_dir) / dataset / 'analysis' / 'aggregated_by_judge_reference.json'
            
            if not data_file.exists():
                logger.warning(f"    File not found: {data_file}")
                continue
            
            # Load data
            data = load_aggregated_data(data_file)
            logger.info(f"    Loaded {len(data)} judge/reference pairs")
            
            # Create table data
            judges, references, table_dict = create_table_data(data)
            logger.info(f"    Found {len(judges)} judges and {len(references)} references")
            
            # Generate LaTeX table
            latex_table = generate_latex_table(judges, references, table_dict, dataset, paper_name)
            
            # Save to file
            output_file = output_dir / f"{paper_name}_{dataset}.tex"
            with open(output_file, 'w') as f:
                f.write(latex_table)
            logger.info(f"    Saved table to: {output_file}")
    
    logger.info("Table generation complete")


if __name__ == "__main__":
    main()
