#!/usr/bin/env python3
"""
Batch reproduction validation script that runs reproduce_paper_experiments.py
across multiple judge/evaluatee pairs.

Usage:
    # Run all pairs for a specific judge
    python batch_reproduction.py --judge llama-3.1-8b --n_samples 200

    # Run specific benchmark only
    python batch_reproduction.py --judge llama-3.1-8b --benchmark math500

    # Run all available judges
    python batch_reproduction.py --all_judges --n_samples 200

"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd

# Import the core reproduction function
from reproduce_paper_experiments import (
    run_reproduction_experiment,
    setup_logging,
    CONFIG,
    load_jsonl,
)

# ============================
# CONFIGURATION
# ============================

# Map judge short names to HuggingFace model IDs
JUDGE_TO_MODEL = {
    # Llama family
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Qwen family
    "qwen-2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen-2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    # Gemma family
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-2-27b": "google/gemma-2-27b-it",
}

# Map judge short names to family names (for directory structure)
JUDGE_TO_FAMILY = {
    "llama-3.1-8b": "llama",
    "llama-3.2-3b": "llama",
    "llama-3.1-70b": "llama",
    "llama-3.3-70b": "llama",
    "qwen-2.5-3b": "qwen",
    "qwen-2.5-7b": "qwen",
    "qwen-2.5-14b": "qwen",
    "qwen-2.5-32b": "qwen",
    "qwen-2.5-72b": "qwen",
    "gemma-2-9b": "gemma",
    "gemma-2-27b": "gemma",
}

# Default judges to run 
DEFAULT_JUDGES = [
    "llama-3.1-8b",
    "llama-3.2-3b",
    "qwen-2.5-7b",
    "gemma-2-9b",
]

# All benchmarks
BENCHMARKS = ["math500", "mmlu", "mbpp-plus"]

# Default settings
DEFAULT_N_SAMPLES = 200
DEFAULT_SEED = 42


# ============================
# DISCOVERY FUNCTIONS
# ============================

def discover_jr_pairs(
    base_path: Path = Path("llm-sp/sp"),
    benchmark_filter: Optional[str] = None,
    judge_filter: Optional[str] = None,
    evaluatee_filter: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Discover all available J/R pairs from the llm-sp/sp directory.
    
    Args:
        base_path: Path to llm-sp/sp directory
        benchmark_filter: Only include this benchmark (e.g., "math500")
        judge_filter: Only include this judge (e.g., "llama-3.1-8b")
        evaluatee_filter: Only include this evaluatee (e.g., "mistral-7b-v0.3")
    
    Returns:
        List of dicts with keys: benchmark, judge_family, judge_short, evaluatee_short, data_path
    """
    pairs = []
    
    benchmarks = [benchmark_filter] if benchmark_filter else BENCHMARKS
    
    for benchmark in benchmarks:
        benchmark_path = base_path / benchmark
        if not benchmark_path.exists():
            continue
        
        for family_dir in benchmark_path.iterdir():
            if not family_dir.is_dir():
                continue
            
            judge_family = family_dir.name
            
            for eval_file in family_dir.glob("*_eval.jsonl"):
                # Parse filename: {judge_short}_{evaluatee_short}_eval.jsonl
                filename = eval_file.stem  # Remove .jsonl
                filename = filename.replace("_eval", "")  # Remove _eval suffix
                
                # Split by underscore, but handle multi-part names
                # Pattern: judge_evaluatee where both can have underscores
                # We need to match against known judge names
                parts = filename.split("_")
                
                # Try to find the split point by matching known judge patterns
                judge_short = None
                evaluatee_short = None
                
                # Build up judge name from left, check if remainder is valid evaluatee
                for i in range(1, len(parts)):
                    potential_judge = "_".join(parts[:i])
                    potential_evaluatee = "_".join(parts[i:])
                    
                    # Check if this looks like a valid split
                    # Judge names typically have version numbers
                    if any(potential_judge.startswith(j.split("-")[0]) for j in JUDGE_TO_MODEL.keys()):
                        judge_short = potential_judge
                        evaluatee_short = potential_evaluatee
                
                if not judge_short or not evaluatee_short:
                    # Fallback: assume last part is evaluatee
                    # This handles cases like "llama-3.1-8b_gemma-2-2b"
                    # Try common patterns
                    for known_judge in JUDGE_TO_MODEL.keys():
                        if filename.startswith(known_judge + "_"):
                            judge_short = known_judge
                            evaluatee_short = filename[len(known_judge) + 1:]
                            break
                
                if not judge_short or not evaluatee_short:
                    # Last resort: split in middle
                    mid = len(parts) // 2
                    judge_short = "_".join(parts[:mid])
                    evaluatee_short = "_".join(parts[mid:])
                
                # Apply filters
                if judge_filter and judge_short != judge_filter:
                    continue
                if evaluatee_filter and evaluatee_short != evaluatee_filter:
                    continue
                
                pairs.append({
                    "benchmark": benchmark,
                    "judge_family": judge_family,
                    "judge_short": judge_short,
                    "evaluatee_short": evaluatee_short,
                    "data_path": str(eval_file),
                })
    
    return pairs


def get_output_path(pair: Dict[str, str]) -> Path:
    """Get the expected output path for a J/R pair."""
    return Path("llm-sp-reprod") / pair["benchmark"] / pair["judge_family"] / f"{pair['judge_short']}_{pair['evaluatee_short']}_reprod.jsonl"


def is_completed(pair: Dict[str, str]) -> bool:
    """Check if a J/R pair has already been processed."""
    output_path = get_output_path(pair)
    return output_path.exists()


# ============================
# BATCH EXECUTION
# ============================

def run_batch(
    pairs: List[Dict[str, str]],
    n_samples: int,
    seed: int,
    resume: bool = True,
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """
    Run reproduction experiments for a batch of J/R pairs.
    
    Groups pairs by judge to minimize model loading.
    
    Args:
        pairs: List of J/R pair dicts
        n_samples: Number of samples per pair
        seed: Random seed
        resume: Skip completed pairs if True
        logger: Logger instance
    
    Returns:
        List of result summaries for each pair
    """
    if logger is None:
        logger = logging.getLogger("batch_reproduction")
    
    results = []
    
    # Group pairs by judge for efficient model loading
    pairs_by_judge = {}
    for pair in pairs:
        judge = pair["judge_short"]
        if judge not in pairs_by_judge:
            pairs_by_judge[judge] = []
        pairs_by_judge[judge].append(pair)
    
    total_pairs = len(pairs)
    completed = 0
    skipped = 0
    failed = 0
    
    logger.info("=" * 80)
    logger.info(f"BATCH REPRODUCTION: {total_pairs} pairs across {len(pairs_by_judge)} judges")
    logger.info("=" * 80)
    
    for judge_short, judge_pairs in pairs_by_judge.items():
        # Check if we have the model mapping
        if judge_short not in JUDGE_TO_MODEL:
            logger.warning(f"No model mapping for judge '{judge_short}', skipping {len(judge_pairs)} pairs")
            skipped += len(judge_pairs)
            continue
        
        model_id = JUDGE_TO_MODEL[judge_short]
        judge_family = JUDGE_TO_FAMILY.get(judge_short, "unknown")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"JUDGE: {judge_short} ({model_id})")
        logger.info(f"Processing {len(judge_pairs)} pairs")
        logger.info(f"{'='*60}")
        
        for i, pair in enumerate(judge_pairs):
            pair_id = f"{pair['benchmark']}/{pair['judge_short']}_{pair['evaluatee_short']}"
            
            # Check if already completed
            if resume and is_completed(pair):
                logger.info(f"[{completed+skipped+1}/{total_pairs}] SKIP (exists): {pair_id}")
                skipped += 1
                
                # Still collect results from existing file
                try:
                    output_path = get_output_path(pair)
                    existing_results = load_jsonl(output_path)
                    summary = summarize_results(existing_results, pair)
                    summary["status"] = "skipped"
                    results.append(summary)
                except Exception as e:
                    logger.warning(f"Could not load existing results: {e}")
                
                continue
            
            logger.info(f"\n[{completed+skipped+failed+1}/{total_pairs}] RUNNING: {pair_id}")
            
            try:
                start_time = time.time()
                
                # Run the reproduction experiment
                run_reproduction_experiment(
                    judge_model=model_id,
                    data_path=Path(pair["data_path"]),
                    output_dir=Path("reproduction_results"),
                    benchmark=pair["benchmark"],
                    judge_family=pair["judge_family"],
                    judge_short=pair["judge_short"],
                    evaluatee_short=pair["evaluatee_short"],
                    reasoning_mode="none",
                    n_samples=n_samples,
                    seed=seed,
                    config=CONFIG,
                    logger=logger,
                )
                
                elapsed = time.time() - start_time
                
                # Load and summarize results
                output_path = get_output_path(pair)
                if output_path.exists():
                    run_results = load_jsonl(output_path)
                    summary = summarize_results(run_results, pair)
                    summary["status"] = "completed"
                    summary["runtime_seconds"] = elapsed
                    results.append(summary)
                    
                    logger.info(f"  ✓ Completed in {elapsed:.1f}s")
                    logger.info(f"    Avg diff G1: {summary['avg_diff_g1']:.4f}")
                    logger.info(f"    Avg diff G2: {summary['avg_diff_g2']:.4f}")
                else:
                    logger.error(f"  ✗ Output file not created")
                    failed += 1
                    results.append({
                        **pair,
                        "status": "failed",
                        "error": "Output file not created",
                    })
                    continue
                
                completed += 1
                
            except Exception as e:
                logger.error(f"  ✗ FAILED: {e}")
                failed += 1
                results.append({
                    **pair,
                    "status": "failed",
                    "error": str(e),
                })
    
    logger.info("\n" + "=" * 80)
    logger.info("BATCH COMPLETE")
    logger.info(f"  Completed: {completed}")
    logger.info(f"  Skipped (resume): {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info("=" * 80)
    
    return results


def summarize_results(results: List[Dict], pair: Dict[str, str]) -> Dict[str, Any]:
    """
    Summarize reproduction results for a single J/R pair.
    
    Args:
        results: List of per-example results from reproduction
        pair: J/R pair metadata
    
    Returns:
        Summary dict with statistics
    """
    if not results:
        return {
            **pair,
            "n_examples": 0,
            "avg_diff_g1": float("nan"),
            "avg_diff_g2": float("nan"),
            "median_diff_g1": float("nan"),
            "median_diff_g2": float("nan"),
        }
    
    diffs_g1 = [r["game_1"]["max_diff"] for r in results]
    diffs_g2 = [r["game_2"]["max_diff"] for r in results]
    
    # Calculate flip rates
    flips_g1 = sum(
        1 for r in results
        if max(r["game_1"]["generated_probs"], key=r["game_1"]["generated_probs"].get) !=
           max(r["game_1"]["reference_probs"], key=r["game_1"]["reference_probs"].get)
    )
    flips_g2 = sum(
        1 for r in results
        if max(r["game_2"]["generated_probs"], key=r["game_2"]["generated_probs"].get) !=
           max(r["game_2"]["reference_probs"], key=r["game_2"]["reference_probs"].get)
    )
    
    return {
        **pair,
        "n_examples": len(results),
        "avg_diff_g1": np.mean(diffs_g1),
        "avg_diff_g2": np.mean(diffs_g2),
        "median_diff_g1": np.median(diffs_g1),
        "median_diff_g2": np.median(diffs_g2),
        "p95_diff_g1": np.percentile(diffs_g1, 95),
        "p95_diff_g2": np.percentile(diffs_g2, 95),
        "max_diff_g1": np.max(diffs_g1),
        "max_diff_g2": np.max(diffs_g2),
        "flip_rate_g1": flips_g1 / len(results),
        "flip_rate_g2": flips_g2 / len(results),
    }


# ============================
# SUMMARY AND VISUALIZATION
# ============================

def generate_summary(
    results: List[Dict[str, Any]],
    output_dir: Path,
    logger: logging.Logger = None,
) -> None:
    """
    Generate aggregate summary statistics and visualizations.
    
    Args:
        results: List of per-pair summary dicts
        output_dir: Directory for output files
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger("batch_reproduction")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to completed/skipped (not failed)
    valid_results = [r for r in results if r.get("status") in ("completed", "skipped") and "avg_diff_g1" in r]
    
    if not valid_results:
        logger.warning("No valid results to summarize")
        return
    
    # Create DataFrame
    df = pd.DataFrame(valid_results)
    
    # Base directory for batch reproduction outputs
    batch_base = output_dir / "batch_reproduction"
    batch_base.mkdir(parents=True, exist_ok=True)
    
    # Save JSON per judge
    for judge in df["judge_short"].unique():
        judge_results = [r for r in valid_results if r.get("judge_short") == judge]
        judge_dir = batch_base / judge
        judge_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = judge_dir / "batch_summary.json"
        with open(json_path, 'w') as f:
            json.dump(judge_results, f, indent=2, default=str)
        logger.info(f"Saved summary JSON: {json_path}")
    
    # Print aggregate statistics
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATE STATISTICS")
    logger.info("=" * 80)
    
    logger.info(f"\nOverall ({len(valid_results)} pairs):")
    logger.info(f"  Avg diff Game 1: {df['avg_diff_g1'].mean():.4f} (±{df['avg_diff_g1'].std():.4f})")
    logger.info(f"  Avg diff Game 2: {df['avg_diff_g2'].mean():.4f} (±{df['avg_diff_g2'].std():.4f})")
    logger.info(f"  Median diff Game 1: {df['median_diff_g1'].mean():.4f}")
    logger.info(f"  Median diff Game 2: {df['median_diff_g2'].mean():.4f}")
    
    # Per-benchmark stats
    logger.info("\nPer-benchmark:")
    for benchmark in df["benchmark"].unique():
        bm_df = df[df["benchmark"] == benchmark]
        logger.info(f"  {benchmark} ({len(bm_df)} pairs):")
        logger.info(f"    Avg diff G1: {bm_df['avg_diff_g1'].mean():.4f}")
        logger.info(f"    Avg diff G2: {bm_df['avg_diff_g2'].mean():.4f}")
    
    # Per-judge stats
    logger.info("\nPer-judge:")
    for judge in df["judge_short"].unique():
        j_df = df[df["judge_short"] == judge]
        logger.info(f"  {judge} ({len(j_df)} pairs):")
        logger.info(f"    Avg diff G1: {j_df['avg_diff_g1'].mean():.4f}")
        logger.info(f"    Avg diff G2: {j_df['avg_diff_g2'].mean():.4f}")
    
    # Flag problematic pairs (avg_diff > 0.3)
    problematic = df[(df["avg_diff_g1"] > 0.3) | (df["avg_diff_g2"] > 0.3)]
    if len(problematic) > 0:
        logger.warning(f"\n⚠ PROBLEMATIC PAIRS (avg_diff > 0.3): {len(problematic)}")
        for _, row in problematic.iterrows():
            logger.warning(f"  {row['benchmark']}/{row['judge_short']}_{row['evaluatee_short']}: "
                         f"G1={row['avg_diff_g1']:.4f}, G2={row['avg_diff_g2']:.4f}")
    
    # Generate visualizations
    try:
        import matplotlib.pyplot as plt
        
        # Base plot directory for batch reproduction
        batch_plot_base = output_dir / "batch_reproduction"
        batch_plot_base.mkdir(parents=True, exist_ok=True)
        
        # Generate plots per judge, per benchmark
        for judge in df["judge_short"].unique():
            judge_df = df[df["judge_short"] == judge]
            judge_dir = batch_plot_base / judge
            judge_dir.mkdir(parents=True, exist_ok=True)
            
            # Overall bar chart for this judge (all benchmarks combined)
            _generate_plots_for_subset(judge_df, judge_dir, f"{judge} (all benchmarks)", logger, filename="diff_overall.png")
            
            # Per-benchmark bar charts
            for benchmark in df["benchmark"].unique():
                subset = judge_df[judge_df["benchmark"] == benchmark]
                if len(subset) == 0:
                    continue
                
                # Create directory: batch_reproduction/{judge}/{benchmark}/
                plot_dir = judge_dir / benchmark
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                _generate_plots_for_subset(subset, plot_dir, f"{judge} / {benchmark}", logger)
        
        # Also generate overall summary plots (all judges, all benchmarks)
        overall_plot_dir = batch_plot_base / "overall"
        overall_plot_dir.mkdir(parents=True, exist_ok=True)
        _generate_plots_for_subset(df, overall_plot_dir, "All Judges / All Benchmarks", logger, filename="diff_overall.png")
        
        logger.info("All visualizations generated")
        
    except ImportError:
        logger.warning("matplotlib not available, skipping visualizations")
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def _generate_plots_for_subset(
    df: pd.DataFrame,
    plot_dir: Path,
    title_prefix: str,
    logger: logging.Logger,
    filename: str = "diff_bar_chart.png",
) -> None:
    """
    Generate plots for a subset of results (e.g., single judge + benchmark).
    
    Args:
        df: DataFrame with results to plot
        plot_dir: Directory to save plots
        title_prefix: Prefix for plot titles
        logger: Logger instance
        filename: Output filename for the bar chart (default: diff_bar_chart.png)
    """
    import matplotlib.pyplot as plt
    
    if len(df) == 0:
        logger.warning(f"No data to plot for {title_prefix}")
        return
    
    # Plot 1: Color-coded bar chart of differences per J/R pair
    # Sort by benchmark first, then alphabetically by evaluatee within each benchmark
    df_sorted = df.copy()
    df_sorted["avg_diff_combined"] = (df_sorted["avg_diff_g1"] + df_sorted["avg_diff_g2"]) / 2
    df_sorted = df_sorted.sort_values(["benchmark", "evaluatee_short"], ascending=[True, True])
    
    # Create pair labels (just evaluatee if single judge, else full pair)
    if df_sorted["judge_short"].nunique() == 1:
        df_sorted["pair_label"] = df_sorted["evaluatee_short"]
    else:
        df_sorted["pair_label"] = df_sorted["judge_short"] + " → " + df_sorted["evaluatee_short"]
    
    # Color by benchmark
    BENCHMARK_COLORS = {
        "math500": "#3498db",    # Blue
        "mmlu": "#e74c3c",       # Red
        "mbpp-plus": "#2ecc71",  # Green
    }
    
    # Get colors based on benchmark
    colors = [BENCHMARK_COLORS.get(b, "#95a5a6") for b in df_sorted["benchmark"]]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(df_sorted) * 0.4)))
    fig.suptitle(f'{title_prefix}', fontsize=14, fontweight='bold')
    
    # Game 1 bar chart
    y_pos = np.arange(len(df_sorted))
    axes[0].barh(y_pos, df_sorted["avg_diff_g1"], color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_sorted["pair_label"], fontsize=9)
    axes[0].set_xlabel('Average Difference from Reference')
    axes[0].set_title(f'Game 1 - Mean: {df["avg_diff_g1"].mean():.4f}')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].set_xlim(0, max(0.8, df_sorted["avg_diff_g1"].max() * 1.1))
    
    # Game 2 bar chart
    axes[1].barh(y_pos, df_sorted["avg_diff_g2"], color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(df_sorted["pair_label"], fontsize=9)
    axes[1].set_xlabel('Average Difference from Reference')
    axes[1].set_title(f'Game 2 - Mean: {df["avg_diff_g2"].mean():.4f}')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_xlim(0, max(0.8, df_sorted["avg_diff_g2"].max() * 1.1))
    
    # Add legend for benchmarks (only if multiple benchmarks present)
    unique_benchmarks = df_sorted["benchmark"].unique()
    if len(unique_benchmarks) > 1:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=BENCHMARK_COLORS.get(b, "#95a5a6"), 
                                  edgecolor='black', label=b) 
                          for b in sorted(unique_benchmarks)]
        axes[1].legend(handles=legend_elements, loc='lower right', title='Benchmark')
    
    plt.tight_layout()
    plt.savefig(plot_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot: {plot_dir / filename}")


def collect_existing_results(
    pairs: List[Dict[str, str]],
    logger: logging.Logger = None,
) -> List[Dict[str, Any]]:
    """
    Collect results from existing reproduction files (for summary_only mode).
    
    Args:
        pairs: List of J/R pair dicts
        logger: Logger instance
    
    Returns:
        List of summary dicts for pairs that have results
    """
    if logger is None:
        logger = logging.getLogger("batch_reproduction")
    
    results = []
    
    for pair in pairs:
        output_path = get_output_path(pair)
        
        if output_path.exists():
            try:
                run_results = load_jsonl(output_path)
                summary = summarize_results(run_results, pair)
                summary["status"] = "collected"
                results.append(summary)
            except Exception as e:
                logger.warning(f"Could not load {output_path}: {e}")
    
    logger.info(f"Collected results from {len(results)} existing files")
    return results


# ============================
# MAIN
# ============================

def setup_batch_logging(output_dir: Path) -> logging.Logger:
    """Set up logging for batch reproduction."""
    log_dir = Path("file_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_reproduction_{timestamp}.log"
    
    logger = logging.getLogger("batch_reproduction")
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
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
    logger.info("BATCH REPRODUCTION PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    
    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Batch reproduction validation across J/R pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all pairs for Llama-3.1-8B judge
  python batch_reproduction.py --judge llama-3.1-8b --n_samples 200

  # Run only math500 benchmark
  python batch_reproduction.py --judge llama-3.1-8b --benchmark math500

  # Run with multiple judges
  python batch_reproduction.py --judges llama-3.1-8b,qwen-2.5-7b


        """
    )
    
    # Judge selection
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument(
        "--judge",
        type=str,
        help="Single judge to run (e.g., llama-3.1-8b)"
    )
    judge_group.add_argument(
        "--judges",
        type=str,
        help="Comma-separated list of judges (e.g., llama-3.1-8b,qwen-2.5-7b)"
    )
    judge_group.add_argument(
        "--all_judges",
        action="store_true",
        help="Run all available judges (warning: long runtime)"
    )
    
    # Filtering
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=BENCHMARKS,
        help="Only run this benchmark"
    )
    parser.add_argument(
        "--evaluatee",
        type=str,
        help="Only run this evaluatee"
    )
    
    # Execution options
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of samples per pair (default: {DEFAULT_N_SAMPLES})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip completed pairs (default: True)"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Re-run all pairs even if completed"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="reproduction_results",
        help="Output directory for summary (default: reproduction_results)"
    )
    
    # Special modes
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only generate summary from existing results, don't run new experiments"
    )
    parser.add_argument(
        "--list_pairs",
        action="store_true",
        help="List all available J/R pairs and exit"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_batch_logging(output_dir)
    
    # Determine which judges to run
    if args.judge:
        judges_to_run = [args.judge]
    elif args.judges:
        judges_to_run = [j.strip() for j in args.judges.split(",")]
    elif args.all_judges:
        judges_to_run = list(JUDGE_TO_MODEL.keys())
    else:
        judges_to_run = DEFAULT_JUDGES
    
    logger.info(f"Judges to run: {judges_to_run}")
    logger.info(f"Benchmark filter: {args.benchmark or 'all'}")
    logger.info(f"Evaluatee filter: {args.evaluatee or 'all'}")
    logger.info(f"N samples: {args.n_samples}")
    logger.info(f"Resume mode: {not args.no_resume}")
    
    # Discover pairs
    all_pairs = []
    for judge in judges_to_run:
        pairs = discover_jr_pairs(
            benchmark_filter=args.benchmark,
            judge_filter=judge,
            evaluatee_filter=args.evaluatee,
        )
        all_pairs.extend(pairs)
    
    logger.info(f"Discovered {len(all_pairs)} J/R pairs")
    
    # List pairs mode
    if args.list_pairs:
        logger.info("\nAvailable J/R pairs:")
        for pair in all_pairs:
            status = "✓" if is_completed(pair) else "○"
            model_available = "✓" if pair["judge_short"] in JUDGE_TO_MODEL else "✗"
            logger.info(f"  {status} [{model_available}] {pair['benchmark']}/{pair['judge_short']}_{pair['evaluatee_short']}")
        
        completed_count = sum(1 for p in all_pairs if is_completed(p))
        runnable_count = sum(1 for p in all_pairs if p["judge_short"] in JUDGE_TO_MODEL)
        logger.info(f"\nTotal: {len(all_pairs)} pairs")
        logger.info(f"Completed: {completed_count}")
        logger.info(f"Runnable (model available): {runnable_count}")
        return
    
    # Summary only mode
    if args.summary_only:
        logger.info("Summary only mode - collecting existing results")
        results = collect_existing_results(all_pairs, logger)
        generate_summary(results, output_dir, logger)
        return
    
    # Run batch
    resume = not args.no_resume
    results = run_batch(all_pairs, args.n_samples, args.seed, resume=resume, logger=logger)
    
    # Generate summary
    generate_summary(results, output_dir, logger)
    
    logger.info("\n" + "=" * 80)
    logger.info("BATCH REPRODUCTION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
