#!/usr/bin/env python3
# compare_dataset_self_preference.py: Compare self-preference across datasets
# Loads statistics from CNN and XSum analysis and creates comparison visualizations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap


# ----------------------
# --- LOGGING SETUP  ---
# ----------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging."""
    log_dir = Path("file_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"compare_datasets_{timestamp}.log"

    logger = logging.getLogger("compare_datasets")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("DATASET COMPARISON ANALYSIS STARTED")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")

    return logger


# ----------------------
# --- DATA LOADING   ---
# ----------------------

def load_statistics(stats_file: Path, logger: logging.Logger) -> Dict:
    """Load self-preference statistics from JSON file."""
    logger.info(f"Loading statistics from {stats_file}")

    if not stats_file.exists():
        logger.error(f"Statistics file not found: {stats_file}")
        return None

    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    logger.info(f"  Model: {stats.get('model_name', 'Unknown')}")
    logger.info(f"  LSP mean: {stats['probability_stats']['lsp_mean']:.4f}")
    logger.info(f"  ILSP mean: {stats['probability_stats']['ilsp_mean']:.4f}")
    logger.info(f"  LSP count: {stats['probability_stats']['lsp_count_balanced']}")
    logger.info(f"  ILSP count: {stats['probability_stats']['ilsp_count_balanced']}")

    return stats


# ----------------------
# --- VISUALIZATION  ---
# ----------------------

def create_comparison_barplot(datasets: List[Dict], output_dir: Path, logger: logging.Logger):
    """
    Create bar plot comparing LSP and ILSP means across datasets.
    """
    logger.info("Creating comparison bar plot...")

    # Set up matplotlib style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })

    # Extract data
    dataset_names = [d['name'] for d in datasets]
    lsp_means = [d['stats']['probability_stats']['lsp_mean'] for d in datasets]
    ilsp_means = [d['stats']['probability_stats']['ilsp_mean'] for d in datasets]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(dataset_names))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, lsp_means, width, label='LSP (Human Correct)',
                   color='green', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, ilsp_means, width, label='ILSP (Human Incorrect)',
                   color='red', alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    # Customize plot
    ax.set_ylabel('Self-Preference Probability', fontweight='bold')
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_title('Self-Preference Comparison Across Datasets', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    output_file = output_dir / "dataset_comparison_barplot.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    logger.info(f"Saved bar plot to {output_file}")

    pdf_file = output_dir / "dataset_comparison_barplot.pdf"
    plt.savefig(pdf_file, bbox_inches='tight', dpi=300)
    logger.info(f"Saved bar plot to {pdf_file}")

    plt.close()


def create_detailed_comparison_table(datasets: List[Dict], output_dir: Path, logger: logging.Logger):
    """
    Create detailed comparison table with all statistics.
    """
    logger.info("Creating detailed comparison table...")

    # Prepare table data
    table_data = []

    for d in datasets:
        stats = d['stats']['probability_stats']
        table_data.append({
            'Dataset': d['name'],
            'LSP Mean': f"{stats['lsp_mean']:.4f}",
            'ILSP Mean': f"{stats['ilsp_mean']:.4f}",
            'All Mean': f"{stats['all_mean']:.4f}",
            'LSP Count': stats['lsp_count_balanced'],
            'ILSP Count': stats['ilsp_count_balanced'],
            'LSP Original': stats.get('lsp_count_original', stats['lsp_count_balanced']),
            'ILSP Original': stats.get('ilsp_count_original', stats['ilsp_count_balanced']),
            'Difference (LSP - ILSP)': f"{stats['lsp_mean'] - stats['ilsp_mean']:.4f}"
        })

    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=[[row[k] for k in row.keys()] for row in table_data],
                    colLabels=list(table_data[0].keys()),
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Detailed Dataset Comparison', fontweight='bold', fontsize=14, pad=20)
    plt.tight_layout()

    # Save
    output_file = output_dir / "dataset_comparison_table.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    logger.info(f"Saved comparison table to {output_file}")

    plt.close()


def create_summary_report(datasets: List[Dict], output_dir: Path, logger: logging.Logger):
    """Create text summary report."""
    logger.info("Creating summary report...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SELF-PREFERENCE COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    for d in datasets:
        stats = d['stats']
        prob_stats = stats['probability_stats']

        report_lines.append("-" * 80)
        report_lines.append(f"Dataset: {d['name']}")
        report_lines.append("-" * 80)
        report_lines.append(f"Model: {stats.get('model_name', 'Unknown')}")
        report_lines.append(f"Timestamp: {stats.get('timestamp', 'Unknown')}")
        report_lines.append("")
        report_lines.append("Self-Preference Statistics:")
        report_lines.append(f"  LSP Mean (Human Correct):      {prob_stats['lsp_mean']:.4f}")
        report_lines.append(f"  ILSP Mean (Human Incorrect):   {prob_stats['ilsp_mean']:.4f}")
        report_lines.append(f"  All Mean (Balanced):           {prob_stats['all_mean']:.4f}")
        report_lines.append(f"  Difference (LSP - ILSP):       {prob_stats['lsp_mean'] - prob_stats['ilsp_mean']:.4f}")
        report_lines.append("")
        report_lines.append("Sample Counts:")
        report_lines.append(f"  LSP Count (Balanced):          {prob_stats['lsp_count_balanced']}")
        report_lines.append(f"  ILSP Count (Balanced):         {prob_stats['ilsp_count_balanced']}")
        report_lines.append(f"  LSP Count (Original):          {prob_stats.get('lsp_count_original', 'N/A')}")
        report_lines.append(f"  ILSP Count (Original):         {prob_stats.get('ilsp_count_original', 'N/A')}")
        report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("CROSS-DATASET OBSERVATIONS")
    report_lines.append("=" * 80)

    # Compute some cross-dataset statistics
    all_lsp_means = [d['stats']['probability_stats']['lsp_mean'] for d in datasets]
    all_ilsp_means = [d['stats']['probability_stats']['ilsp_mean'] for d in datasets]
    all_diffs = [lsp - ilsp for lsp, ilsp in zip(all_lsp_means, all_ilsp_means)]

    report_lines.append(f"LSP Mean Range:   {min(all_lsp_means):.4f} - {max(all_lsp_means):.4f}")
    report_lines.append(f"ILSP Mean Range:  {min(all_ilsp_means):.4f} - {max(all_ilsp_means):.4f}")
    report_lines.append(f"Difference Range: {min(all_diffs):.4f} - {max(all_diffs):.4f}")
    report_lines.append(f"Average LSP:      {np.mean(all_lsp_means):.4f}")
    report_lines.append(f"Average ILSP:     {np.mean(all_ilsp_means):.4f}")
    report_lines.append(f"Average Diff:     {np.mean(all_diffs):.4f}")
    report_lines.append("")
    report_lines.append("=" * 80)

    # Write report
    report_text = "\n".join(report_lines)

    # Print to logger
    for line in report_lines:
        logger.info(line)

    # Save to file
    output_file = output_dir / "comparison_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"Saved text report to {output_file}")


# ----------------------
# --- MAIN ENTRY     ---
# ----------------------

def main():
    """Main entry point for dataset comparison."""
    parser = argparse.ArgumentParser(
        description="Compare self-preference metrics across datasets"
    )
    parser.add_argument("--stats_files", type=str, nargs='+', required=True,
                       help="Paths to statistics JSON files from analyze_cnn_self_preference.py")
    parser.add_argument("--dataset_names", type=str, nargs='+', required=True,
                       help="Names for each dataset (same order as stats_files)")
    parser.add_argument("--output_dir", type=str, default="comparison_plots",
                       help="Output directory for comparison plots")

    args = parser.parse_args()

    if len(args.stats_files) != len(args.dataset_names):
        print("ERROR: Number of stats files must match number of dataset names")
        sys.exit(1)

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(output_dir)

    logger.info(f"Comparing {len(args.stats_files)} datasets")

    # Load all statistics
    datasets = []
    for stats_file, name in zip(args.stats_files, args.dataset_names):
        stats = load_statistics(Path(stats_file), logger)
        if stats is None:
            logger.error(f"Failed to load statistics for {name}")
            continue
        datasets.append({
            'name': name,
            'stats': stats,
            'file': stats_file
        })

    if len(datasets) == 0:
        logger.error("No datasets loaded successfully. Exiting.")
        sys.exit(1)

    logger.info(f"\nSuccessfully loaded {len(datasets)} datasets")

    # Create visualizations
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    create_comparison_barplot(datasets, output_dir, logger)
    create_detailed_comparison_table(datasets, output_dir, logger)
    create_summary_report(datasets, output_dir, logger)

    logger.info("=" * 80)
    logger.info("COMPARISON ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
