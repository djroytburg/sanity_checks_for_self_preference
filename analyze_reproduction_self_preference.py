#!/usr/bin/env python3
# analyze_reproduction_self_preference.py: Analyze self-preference in reproduction results
# Written by: Dani
# Created: 2025-12-21 21:30 EST
# Last Modified: 2025-12-21 21:30 EST
"""
Analyze self-preference metrics from reproduction experiment results.

Computes paper metrics:
- SPR (Self-Preference Ratio): Overall preference for own response
- LSPR (Legitimate Self-Preference Ratio): Preference when own response is correct
- HSPP (Harmful Self-Preference Propensity): Preference when own response is incorrect

Generates visualizations comparing reference vs generated distributions.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap


# ----------------------
# --- LOGGING SETUP  ---
# ----------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging per project standards."""
    log_dir = Path("file_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analyze_reproduction_sp_{timestamp}.log"
    
    logger = logging.getLogger("analyze_reproduction_sp")
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
    logger.info("REPRODUCTION SELF-PREFERENCE ANALYSIS STARTED")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"User: {os.environ.get('USER', 'unknown')}")
    
    return logger


# ----------------------
# --- DATA LOADING   ---
# ----------------------

def load_reproduction_results(jsonl_path: Path, logger: logging.Logger) -> List[Dict]:
    """Load reproduction experiment results from JSONL file."""
    logger.info(f"Loading reproduction results from {jsonl_path}")
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} examples")
    return results


# ----------------------
# --- VERDICT LOGIC  ---
# ----------------------

def get_verdict(probs: Dict[str, float]) -> str:
    """
    Get verdict (A/B/T) from probability dict.
    
    Args:
        probs: Dictionary with keys A, B, T and probability values.
    
    Returns:
        str: Verdict label with highest probability.
    """
    return max(probs.items(), key=lambda x: x[1])[0]


def aggregate_verdict(verdict_g1: str, verdict_g2: str) -> str:
    """
    Aggregate verdicts from two games following paper's J* aggregation.
    
    Per paper: If one verdict is tie and other is not, use non-tie.
    If both agree, use that. If both disagree (and neither is tie), return tie.
    
    Args:
        verdict_g1: Verdict from game 1 (AB order).
        verdict_g2: Verdict from game 2 (BA order).
    
    Returns:
        str: Aggregated verdict (A/B/T).
    """
    # If verdicts agree, use that
    if verdict_g1 == verdict_g2:
        return verdict_g1
    
    # If one is tie and other is not, use non-tie
    if verdict_g2 == "T" and verdict_g1 != "T":
        return verdict_g1
    if verdict_g1 == "T" and verdict_g2 != "T":
        return verdict_g2
    
    # If both disagree and neither is tie, return tie
    if verdict_g1 != "T" and verdict_g2 != "T" and verdict_g1 != verdict_g2:
        return "T"
    
    # Default (shouldn't reach here)
    return "T"


# ----------------------
# --- METRIC CALC    ---
# ----------------------

def extract_self_preference_probs(results: List[Dict], source: str, logger: logging.Logger) -> Dict:
    """
    Extract self-preference probabilities from reproduction results.
    
    In the reproduction data:
    - response_a is judge's own response (assistant 1)
    - response_b is reference/other response (assistant 2)
    - Game 1: Judge is A (own response in position A)
    - Game 2: Judge is B (own response in position B, after swap)
    
    Self-preference = average of P(A|Game1) and P(B|Game2)
    
    Args:
        results: List of reproduction result dicts.
        source: "reference" or "generated" - which probabilities to use.
        logger: Logger instance.
    
    Returns:
        Dict with keys lsp, ilsp, all containing self-pref probability lists.
    """
    logger.info(f"Extracting self-preference probabilities from {source} data")
    
    prob_key = "reference_probs" if source == "reference" else "generated_probs"
    
    lsp_probs = []  # Legitimate self-preference
    ilsp_probs = []  # Illegitimate self-preference
    all_probs = []
    
    skipped = 0
    
    for r in results:
        # Get correctness info from metadata
        meta = r.get("meta", {})
        judge_correct = meta.get("assistent_1_is_correct", None)
        
        if judge_correct is None:
            skipped += 1
            continue
        
        # Get probabilities for both games
        probs_g1 = r["game_1"][prob_key]
        probs_g2 = r["game_2"][prob_key]
        
        # Self-preference = avg of P(A|Game1) and P(B|Game2)
        # Game 1: Judge is A (own response in position A)
        # Game 2: Judge is B (own response swapped to position B)
        self_pref = (probs_g1["A"] + probs_g2["B"]) / 2.0
        
        all_probs.append(self_pref)
        
        # Categorize as legitimate or illegitimate
        if judge_correct:
            lsp_probs.append(self_pref)
        else:
            ilsp_probs.append(self_pref)
    
    logger.info(f"  {source.capitalize()} - Extracted probabilities:")
    logger.info(f"    Total: {len(all_probs)}")
    logger.info(f"    LSP (judge correct): {len(lsp_probs)}")
    logger.info(f"    ILSP (judge incorrect): {len(ilsp_probs)}")
    logger.info(f"    Skipped (no correctness info): {skipped}")
    
    return {
        "lsp": lsp_probs,
        "ilsp": ilsp_probs,
        "all": all_probs,
        "skipped": skipped,
    }


def calculate_paper_metrics(results: List[Dict], source: str, logger: logging.Logger) -> Dict:
    """
    Calculate SPR, LSPR, HSPP following paper equations.
    
    SPR = fraction of examples where judge prefers own response
    LSPR = fraction of self-preference that is legitimate (own response is correct)
    HSPP = fraction of incorrect judge responses where judge still prefers own
    
    Args:
        results: List of reproduction result dicts.
        source: "reference" or "generated" - which verdicts to use.
        logger: Logger instance.
    
    Returns:
        Dict with metric values and counts.
    """
    logger.info(f"Calculating paper metrics for {source} data")
    
    prob_key = "reference_probs" if source == "reference" else "generated_probs"
    
    # Counts for metrics
    total_examples = 0
    self_pref_count = 0
    legitimate_self_pref_count = 0
    judge_correct_count = 0
    judge_incorrect_count = 0
    harmful_sp_count = 0
    
    for r in results:
        meta = r.get("meta", {})
        judge_correct = meta.get("assistent_1_is_correct", None)
        
        if judge_correct is None:
            continue
        
        total_examples += 1
        
        # Get verdicts from both games
        probs_g1 = r["game_1"][prob_key]
        probs_g2 = r["game_2"][prob_key]
        
        verdict_g1 = get_verdict(probs_g1)
        verdict_g2 = get_verdict(probs_g2)
        
        # Aggregate verdict
        agg_verdict = aggregate_verdict(verdict_g1, verdict_g2)
        
        # Check if judge prefers own response
        # In Game 1: Judge is A, so self-pref if verdict is A
        # In Game 2: Judge is B, so self-pref if verdict is B
        # Aggregated: A means prefers judge (own response)
        prefers_own = (agg_verdict == "A")
        
        if prefers_own:
            self_pref_count += 1
            
            if judge_correct:
                legitimate_self_pref_count += 1
        
        if judge_correct:
            judge_correct_count += 1
        else:
            judge_incorrect_count += 1
            if prefers_own:
                harmful_sp_count += 1
    
    # Calculate metrics
    spr = self_pref_count / total_examples if total_examples > 0 else 0.0
    lspr = legitimate_self_pref_count / self_pref_count if self_pref_count > 0 else 0.0
    hspp = harmful_sp_count / judge_incorrect_count if judge_incorrect_count > 0 else 0.0
    
    metrics = {
        "SPR": spr,
        "LSPR": lspr,
        "HSPP": hspp,
        "total_examples": total_examples,
        "self_pref_count": self_pref_count,
        "legitimate_self_pref_count": legitimate_self_pref_count,
        "judge_correct_count": judge_correct_count,
        "judge_incorrect_count": judge_incorrect_count,
        "harmful_sp_count": harmful_sp_count,
    }
    
    logger.info(f"  {source.capitalize()} metrics:")
    logger.info(f"    SPR (Self-Preference Ratio): {spr:.4f}")
    logger.info(f"    LSPR (Legitimate SP Ratio): {lspr:.4f}")
    logger.info(f"    HSPP (Harmful SP Propensity): {hspp:.4f}")
    logger.info(f"    Total examples: {total_examples}")
    logger.info(f"    Self-pref count: {self_pref_count}")
    logger.info(f"    Legitimate SP count: {legitimate_self_pref_count}")
    
    return metrics


# ----------------------
# --- VISUALIZATION  ---
# ----------------------

def plot_histogram_with_stats(ax, data, objective, title=None, xlabel=None, 
                               color='cornflowerblue', no_spine=False, **hist_kwargs):
    """Plot histogram with mean, confidence interval, and objective score."""
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate mean and bootstrap CI
    mean = np.mean(data)
    res = bootstrap((np.array(data),), np.mean, random_state=42, n_resamples=1000)
    CI = res.confidence_interval
    
    # Plotting
    ax.hist(data, color=color, edgecolor='white', linewidth=0.5, **hist_kwargs)
    ax.axvline(mean, color='black', linestyle='--', linewidth=2, label='μ')
    ax.axvline(objective, color='red', linestyle='--', linewidth=2, label='Objective')
    
    # Plot confidence interval
    ax.axvspan(CI.low, CI.high, alpha=0.2, color='orange', label='95% CI')
    
    if title:
        ax.set_title(title, fontweight='bold', fontsize=12, pad=18)
        
        # Subtitle with statistics
        subtitle = f"μ: {mean:.3f} | Objective: {objective:.3f} | n={len(data)}"
        ax.text(0.5, 1.0, subtitle,
                ha='center',
                va='bottom',
                transform=ax.transAxes,
                fontsize=9,
                color='gray')
    
    if xlabel:
        ax.set_xlabel(xlabel)
    
    if no_spine:
        ax.spines['left'].set_visible(False)
    
    # Set ylabel based on whether density is used
    if hist_kwargs.get('density', False):
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_comparison_visualization(ref_probs: Dict, gen_probs: Dict, 
                                   model_name: str, output_dir: Path, logger: logging.Logger):
    """
    Create visualization comparing reference vs generated self-preference distributions.
    
    Creates two rows: reference and generated, with columns for LSP, ILSP, and all.
    """
    logger.info("Creating comparison visualization...")
    
    # Set up matplotlib style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })
    
    # Create figure with 2 rows x 3 columns
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10), sharey='row')
    fig.suptitle(f'Self-Preference Analysis: Reference vs Generated\n{model_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Row 1: Reference data
    plot_histogram_with_stats(
        ax=axes[0, 0],
        data=ref_probs['lsp'],
        objective=1.0,
        title="Reference - Legitimate Self-Preference (LSP)",
        xlabel="P(self)",
        color='green',
        density=True,
        bins=20,
        alpha=0.7
    )
    
    plot_histogram_with_stats(
        ax=axes[0, 1],
        data=ref_probs['ilsp'],
        objective=0.0,
        title="Reference - Illegitimate Self-Preference (ILSP)",
        xlabel="P(self)",
        color='red',
        density=True,
        bins=20,
        alpha=0.7,
        no_spine=True
    )
    
    # Calculate LSP proportion for reference
    ref_lsp_prop = len(ref_probs['lsp']) / (len(ref_probs['lsp']) + len(ref_probs['ilsp']))
    
    plot_histogram_with_stats(
        ax=axes[0, 2],
        data=ref_probs['all'],
        objective=ref_lsp_prop * 1.0 + (1 - ref_lsp_prop) * 0.0,
        title="Reference - All Examples",
        xlabel="P(self)",
        color='cornflowerblue',
        density=True,
        bins=20,
        alpha=0.7,
        no_spine=True
    )
    
    # Row 2: Generated data
    plot_histogram_with_stats(
        ax=axes[1, 0],
        data=gen_probs['lsp'],
        objective=1.0,
        title="Generated - Legitimate Self-Preference (LSP)",
        xlabel="P(self)",
        color='green',
        density=True,
        bins=20,
        alpha=0.7
    )
    
    plot_histogram_with_stats(
        ax=axes[1, 1],
        data=gen_probs['ilsp'],
        objective=0.0,
        title="Generated - Illegitimate Self-Preference (ILSP)",
        xlabel="P(self)",
        color='red',
        density=True,
        bins=20,
        alpha=0.7,
        no_spine=True
    )
    
    # Calculate LSP proportion for generated
    gen_lsp_prop = len(gen_probs['lsp']) / (len(gen_probs['lsp']) + len(gen_probs['ilsp']))
    
    plot_histogram_with_stats(
        ax=axes[1, 2],
        data=gen_probs['all'],
        objective=gen_lsp_prop * 1.0 + (1 - gen_lsp_prop) * 0.0,
        title="Generated - All Examples",
        xlabel="P(self)",
        color='cornflowerblue',
        density=True,
        bins=20,
        alpha=0.7,
        no_spine=True
    )
    
    # Add legend to last plot
    axes[1, 2].legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "self_preference_comparison.png"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved visualization to {png_path}")
    
    pdf_path = output_dir / "self_preference_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved visualization to {pdf_path}")
    
    plt.close()


def create_distribution_shift_plot(ref_probs: Dict, gen_probs: Dict,
                                   model_name: str, output_dir: Path, logger: logging.Logger):
    """
    Create visualization showing distribution shifts between reference and generated.
    """
    logger.info("Creating distribution shift visualization...")
    
    fig, axes = plt.subplots(ncols=3, figsize=(16, 5))
    fig.suptitle(f'Distribution Shift: Reference → Generated\n{model_name}',
                 fontsize=14, fontweight='bold')
    
    categories = [
        ('lsp', 'Legitimate Self-Preference (LSP)', 'green'),
        ('ilsp', 'Illegitimate Self-Preference (ILSP)', 'red'),
        ('all', 'All Examples', 'cornflowerblue')
    ]
    
    for idx, (cat_key, cat_title, color) in enumerate(categories):
        ax = axes[idx]
        
        ref_data = ref_probs[cat_key]
        gen_data = gen_probs[cat_key]
        
        if len(ref_data) > 0 and len(gen_data) > 0:
            # Plot both distributions
            ax.hist(ref_data, bins=20, alpha=0.5, color='blue', label='Reference', density=True)
            ax.hist(gen_data, bins=20, alpha=0.5, color=color, label='Generated', density=True)
            
            # Add means
            ref_mean = np.mean(ref_data)
            gen_mean = np.mean(gen_data)
            ax.axvline(ref_mean, color='blue', linestyle='--', linewidth=2, label=f'Ref μ={ref_mean:.3f}')
            ax.axvline(gen_mean, color=color, linestyle='--', linewidth=2, label=f'Gen μ={gen_mean:.3f}')
            
            # Calculate shift
            shift = gen_mean - ref_mean
            ax.text(0.5, 0.95, f'Shift: {shift:+.3f}',
                   ha='center', va='top', transform=ax.transAxes,
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(cat_title, fontweight='bold')
            ax.set_xlabel('P(self)')
            ax.set_ylabel('Density')
            ax.legend(loc='upper left')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cat_title)
    
    plt.tight_layout()
    
    png_path = output_dir / "distribution_shift.png"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved shift visualization to {png_path}")
    
    plt.close()


# ----------------------
# --- MAIN ENTRY     ---
# ----------------------

def print_summary_report(ref_metrics: Dict, gen_metrics: Dict, model_name: str, logger: logging.Logger):
    """Print comprehensive summary report."""
    logger.info("=" * 80)
    logger.info("SELF-PREFERENCE ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info("")
    
    logger.info("REFERENCE (Paper Authors' Data)")
    logger.info("-" * 80)
    logger.info(f"  SPR (Self-Preference Ratio):           {ref_metrics['SPR']:.4f}")
    logger.info(f"  LSPR (Legitimate SP Ratio):            {ref_metrics['LSPR']:.4f}")
    logger.info(f"  HSPP (Harmful SP Propensity):          {ref_metrics['HSPP']:.4f}")
    logger.info(f"  Total examples:                        {ref_metrics['total_examples']}")
    logger.info(f"  Self-preference count:                 {ref_metrics['self_pref_count']}")
    logger.info(f"  Legitimate SP count:                   {ref_metrics['legitimate_self_pref_count']}")
    logger.info("")
    
    logger.info("GENERATED (Reproduction)")
    logger.info("-" * 80)
    logger.info(f"  SPR (Self-Preference Ratio):           {gen_metrics['SPR']:.4f}")
    logger.info(f"  LSPR (Legitimate SP Ratio):            {gen_metrics['LSPR']:.4f}")
    logger.info(f"  HSPP (Harmful SP Propensity):          {gen_metrics['HSPP']:.4f}")
    logger.info(f"  Total examples:                        {gen_metrics['total_examples']}")
    logger.info(f"  Self-preference count:                 {gen_metrics['self_pref_count']}")
    logger.info(f"  Legitimate SP count:                   {gen_metrics['legitimate_self_pref_count']}")
    logger.info("")
    
    logger.info("DIFFERENCES (Generated - Reference)")
    logger.info("-" * 80)
    logger.info(f"  ΔSPR:                                  {gen_metrics['SPR'] - ref_metrics['SPR']:+.4f}")
    logger.info(f"  ΔLSPR:                                 {gen_metrics['LSPR'] - ref_metrics['LSPR']:+.4f}")
    logger.info(f"  ΔHSPP:                                 {gen_metrics['HSPP'] - ref_metrics['HSPP']:+.4f}")
    logger.info("")
    
    logger.info("=" * 80)


def main():
    """Main entry point for reproduction self-preference analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze self-preference metrics from reproduction experiment results"
    )
    parser.add_argument("--results_file", type=str, required=True,
                       help="Path to reproduction results JSONL file")
    parser.add_argument("--model_name", type=str, default="Unknown",
                       help="Model name for display in report and plots")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots and statistics (default: same as results file)")
    
    args = parser.parse_args()
    
    # Set up paths
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / "plots"
    
    # Set up logging
    logger = setup_logging(output_dir)
    
    logger.info(f"Results file: {results_path}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    results = load_reproduction_results(results_path, logger)
    
    if len(results) == 0:
        logger.error("No results loaded. Exiting.")
        sys.exit(1)
    
    # Extract self-preference probabilities
    logger.info("=" * 80)
    logger.info("EXTRACTING SELF-PREFERENCE PROBABILITIES")
    logger.info("=" * 80)
    
    ref_probs = extract_self_preference_probs(results, "reference", logger)
    gen_probs = extract_self_preference_probs(results, "generated", logger)
    
    # Calculate paper metrics
    logger.info("")
    logger.info("=" * 80)
    logger.info("CALCULATING PAPER METRICS")
    logger.info("=" * 80)
    
    ref_metrics = calculate_paper_metrics(results, "reference", logger)
    gen_metrics = calculate_paper_metrics(results, "generated", logger)
    
    # Print summary report
    logger.info("")
    print_summary_report(ref_metrics, gen_metrics, args.model_name, logger)
    
    # Create visualizations
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    create_comparison_visualization(ref_probs, gen_probs, args.model_name, output_dir, logger)
    create_distribution_shift_plot(ref_probs, gen_probs, args.model_name, output_dir, logger)
    
    # Save statistics to JSON
    stats_output = {
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "results_file": str(results_path),
        "total_examples": len(results),
        "reference_metrics": ref_metrics,
        "generated_metrics": gen_metrics,
        "differences": {
            "delta_SPR": gen_metrics["SPR"] - ref_metrics["SPR"],
            "delta_LSPR": gen_metrics["LSPR"] - ref_metrics["LSPR"],
            "delta_HSPP": gen_metrics["HSPP"] - ref_metrics["HSPP"],
        },
    }
    
    stats_path = output_dir / "self_preference_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved statistics to {stats_path}")
    
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
