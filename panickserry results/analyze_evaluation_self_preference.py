#!/usr/bin/env python3
# analyze_evaluation_self_preference.py: Analyze self-preference in summary evaluation results
# Created for analyzing evaluation_results.json with GPT-3.5 logprobs and GPT-4o ground truth
"""
Analyze self-preference metrics from summary evaluation results.

Computes paper metrics:
- SPR (Self-Preference Ratio): Overall preference for own response
- LSPR (Legitimate Self-Preference Ratio): Preference when own response is correct
- HSPP (Harmful Self-Preference Propensity): Preference when own response is incorrect

Uses probabilities from tokens "1" and "2" in logprobs and judge correctness from ground truth.
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
    log_file = log_dir / f"analyze_evaluation_sp_{timestamp}.log"

    logger = logging.getLogger("analyze_evaluation_sp")
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
    logger.info("EVALUATION SELF-PREFERENCE ANALYSIS STARTED")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"User: {os.environ.get('USER', 'unknown')}")

    return logger


# ----------------------
# --- DATA LOADING   ---
# ----------------------

def load_evaluation_results(json_path: Path, logger: logging.Logger) -> List[Dict]:
    """Load evaluation results from JSON file."""
    logger.info(f"Loading evaluation results from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} examples")
    return results


def load_ground_truth_judge(json_path: Path, logger: logging.Logger) -> List[Dict]:
    """Load ground truth judge results from JSON file."""
    logger.info(f"Loading ground truth judge results from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} ground truth judgments")
    return results


# ----------------------
# --- VERDICT LOGIC  ---
# ----------------------

def get_probabilities_from_logprobs(logprobs_list: List[Dict]) -> Dict[str, float]:
    """
    Extract probabilities for tokens "1" and "2" from logprobs.

    Args:
        logprobs_list: List of dicts with token, logprob, and probability.

    Returns:
        Dict with keys "1" and "2" mapping to their probabilities.
    """
    probs = {"1": 0.0, "2": 0.0}

    for item in logprobs_list:
        token = item["token"]
        if token in ["1", "2"]:
            probs[token] = item["probability"]

    return probs


def aggregate_verdict_from_answers(answer_original: str, answer_flipped: str) -> str:
    """
    Aggregate verdicts from original and flipped orderings.

    - original_order: summary1 is first (position 1)
    - flipped_order: summary2 is first (position 1), summary1 is second (position 2)

    If both agree on the same summary (accounting for position swap), return that.
    Otherwise, return tie.

    Args:
        answer_original: Answer when summary1 is in position 1 ("1" or "2")
        answer_flipped: Answer when summary2 is in position 1 ("1" or "2")

    Returns:
        "1" if prefers summary1, "2" if prefers summary2, "T" for tie
    """
    # Map answers to which summary they prefer
    # original_order: "1" = summary1, "2" = summary2
    # flipped_order: "1" = summary2, "2" = summary1

    if answer_original == "1" and answer_flipped == "2":
        # Both prefer summary1
        return "1"
    elif answer_original == "2" and answer_flipped == "1":
        # Both prefer summary2
        return "2"
    else:
        # Disagreement = tie
        return "T"


def get_judge_correct_from_ground_truth(gt_original: str, gt_flipped: str) -> bool:
    """
    Determine if summary1 (judge's own response) is correct based on ground truth.

    Args:
        gt_original: Ground truth answer when summary1 is in position 1
        gt_flipped: Ground truth answer when summary2 is in position 1

    Returns:
        True if summary1 is preferred by ground truth, False otherwise
    """
    verdict = aggregate_verdict_from_answers(gt_original, gt_flipped)
    return verdict == "1"


# ----------------------
# --- METRIC CALC    ---
# ----------------------

def extract_self_preference_probs(eval_results: List[Dict], gt_results: List[Dict],
                                  logger: logging.Logger) -> Dict:
    """
    Extract self-preference probabilities from evaluation results.

    In the evaluation data:
    - summary1 is the judge's own response (what we're measuring self-preference for)
    - summary2 is the other response
    - original_order: summary1 in position 1
    - flipped_order: summary2 in position 1 (summary1 in position 2)

    Self-preference = average of P(1|original) and P(2|flipped)
    (probability of choosing summary1 in both orderings)

    Args:
        eval_results: List of evaluation result dicts with logprobs.
        gt_results: List of ground truth judge result dicts.
        logger: Logger instance.

    Returns:
        Dict with keys lsp, ilsp, all containing self-pref probability lists.
    """
    logger.info("Extracting self-preference probabilities from evaluation data")

    lsp_probs = []  # Legitimate self-preference
    ilsp_probs = []  # Illegitimate self-preference
    all_probs = []

    skipped = 0

    # Build ground truth mapping by article_index
    gt_map = {gt["article_index"]: gt for gt in gt_results}

    for eval_item in eval_results:
        article_idx = eval_item["article_index"]

        # Get corresponding ground truth
        if article_idx not in gt_map:
            skipped += 1
            continue

        gt_item = gt_map[article_idx]

        # Determine if judge (summary1) is correct based on ground truth
        judge_correct = get_judge_correct_from_ground_truth(
            gt_item["original_order"]["answer"],
            gt_item["flipped_order"]["answer"]
        )

        # Get probabilities from logprobs
        probs_original = get_probabilities_from_logprobs(
            eval_item["original_order"]["top_logprobs"]
        )
        probs_flipped = get_probabilities_from_logprobs(
            eval_item["flipped_order"]["top_logprobs"]
        )

        # Self-preference = avg of P(1|original) and P(2|flipped)
        # This is the probability of choosing summary1 in both orderings
        self_pref = (probs_original["1"] + probs_flipped["2"]) / 2.0

        all_probs.append(self_pref)

        # Categorize as legitimate or illegitimate
        if judge_correct:
            lsp_probs.append(self_pref)
        else:
            ilsp_probs.append(self_pref)

    logger.info(f"  Extracted probabilities:")
    logger.info(f"    Total: {len(all_probs)}")
    logger.info(f"    LSP (judge correct): {len(lsp_probs)}")
    logger.info(f"    ILSP (judge incorrect): {len(ilsp_probs)}")
    logger.info(f"    Skipped (no ground truth): {skipped}")

    return {
        "lsp": lsp_probs,
        "ilsp": ilsp_probs,
        "all": all_probs,
        "skipped": skipped,
    }


def calculate_paper_metrics(eval_results: List[Dict], gt_results: List[Dict],
                            logger: logging.Logger) -> Dict:
    """
    Calculate SPR, LSPR, HSPP following paper equations.

    SPR = fraction of examples where judge prefers own response
    LSPR = fraction of self-preference that is legitimate (own response is correct)
    HSPP = fraction of incorrect judge responses where judge still prefers own

    Args:
        eval_results: List of evaluation result dicts.
        gt_results: List of ground truth judge result dicts.
        logger: Logger instance.

    Returns:
        Dict with metric values and counts.
    """
    logger.info("Calculating paper metrics")

    # Counts for metrics
    total_examples = 0
    self_pref_count = 0
    legitimate_self_pref_count = 0
    judge_correct_count = 0
    judge_incorrect_count = 0
    harmful_sp_count = 0

    # Build ground truth mapping by article_index
    gt_map = {gt["article_index"]: gt for gt in gt_results}

    for eval_item in eval_results:
        article_idx = eval_item["article_index"]

        # Get corresponding ground truth
        if article_idx not in gt_map:
            continue

        gt_item = gt_map[article_idx]

        total_examples += 1

        # Determine if judge (summary1) is correct based on ground truth
        judge_correct = get_judge_correct_from_ground_truth(
            gt_item["original_order"]["answer"],
            gt_item["flipped_order"]["answer"]
        )

        # Get judge's verdict from evaluation results
        eval_verdict = aggregate_verdict_from_answers(
            eval_item["original_order"]["answer"],
            eval_item["flipped_order"]["answer"]
        )

        # Check if judge prefers own response (summary1)
        prefers_own = (eval_verdict == "1")

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

    logger.info(f"  Metrics:")
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
        if title:
            ax.set_title(title, fontweight='bold', fontsize=12)
        return

    # Calculate mean
    mean = np.mean(data)

    # Calculate bootstrap CI only if we have at least 2 observations
    has_ci = len(data) >= 2
    if has_ci:
        res = bootstrap((np.array(data),), np.mean, random_state=42, n_resamples=1000)
        CI = res.confidence_interval

    # Plotting
    ax.hist(data, color=color, edgecolor='white', linewidth=0.5, **hist_kwargs)
    ax.axvline(mean, color='black', linestyle='--', linewidth=2, label='μ')
    ax.axvline(objective, color='red', linestyle='--', linewidth=2, label='Objective')

    # Plot confidence interval only if available
    if has_ci:
        ax.axvspan(CI.low, CI.high, alpha=0.2, color='orange', label='95% CI')

    if title:
        ax.set_title(title, fontweight='bold', fontsize=12, pad=18)

        # Subtitle with statistics
        if has_ci:
            subtitle = f"μ: {mean:.3f} | Objective: {objective:.3f} | n={len(data)}"
        else:
            subtitle = f"μ: {mean:.3f} | Objective: {objective:.3f} | n={len(data)} (insufficient for CI)"
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
    ax.legend(loc='upper left')


def create_visualization(probs: Dict, model_name: str, output_dir: Path, logger: logging.Logger):
    """
    Create visualization showing self-preference distributions.
    """
    logger.info("Creating visualization...")

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

    # Create figure with 1 row x 3 columns
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    fig.suptitle(f'Self-Preference Analysis\n{model_name}',
                 fontsize=14, fontweight='bold', y=0.98)

    # Plot LSP
    plot_histogram_with_stats(
        ax=axes[0],
        data=probs['lsp'],
        objective=1.0,
        title="Legitimate Self-Preference (LSP)",
        xlabel="P(self)",
        color='green',
        density=True,
        bins=20,
        alpha=0.7
    )

    # Plot ILSP
    plot_histogram_with_stats(
        ax=axes[1],
        data=probs['ilsp'],
        objective=0.0,
        title="Illegitimate Self-Preference (ILSP)",
        xlabel="P(self)",
        color='red',
        density=True,
        bins=20,
        alpha=0.7,
        no_spine=True
    )

    # Calculate LSP proportion
    lsp_prop = len(probs['lsp']) / (len(probs['lsp']) + len(probs['ilsp'])) if (len(probs['lsp']) + len(probs['ilsp'])) > 0 else 0

    # Plot all
    plot_histogram_with_stats(
        ax=axes[2],
        data=probs['all'],
        objective=lsp_prop * 1.0 + (1 - lsp_prop) * 0.0,
        title="All Examples",
        xlabel="P(self)",
        color='cornflowerblue',
        density=True,
        bins=20,
        alpha=0.7,
        no_spine=True
    )

    plt.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "self_preference_analysis.png"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved visualization to {png_path}")

    pdf_path = output_dir / "self_preference_analysis.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved visualization to {pdf_path}")

    plt.close()


# ----------------------
# --- MAIN ENTRY     ---
# ----------------------

def print_summary_report(metrics: Dict, model_name: str, logger: logging.Logger):
    """Print comprehensive summary report."""
    logger.info("=" * 80)
    logger.info("SELF-PREFERENCE ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info("")

    logger.info("METRICS")
    logger.info("-" * 80)
    logger.info(f"  SPR (Self-Preference Ratio):           {metrics['SPR']:.4f}")
    logger.info(f"  LSPR (Legitimate SP Ratio):            {metrics['LSPR']:.4f}")
    logger.info(f"  HSPP (Harmful SP Propensity):          {metrics['HSPP']:.4f}")
    logger.info(f"  Total examples:                        {metrics['total_examples']}")
    logger.info(f"  Self-preference count:                 {metrics['self_pref_count']}")
    logger.info(f"  Legitimate SP count:                   {metrics['legitimate_self_pref_count']}")
    logger.info(f"  Judge correct count:                   {metrics['judge_correct_count']}")
    logger.info(f"  Judge incorrect count:                 {metrics['judge_incorrect_count']}")
    logger.info(f"  Harmful SP count:                      {metrics['harmful_sp_count']}")
    logger.info("")
    logger.info("=" * 80)


def main():
    """Main entry point for evaluation self-preference analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze self-preference metrics from summary evaluation results"
    )
    parser.add_argument("--eval_results", type=str, required=True,
                       help="Path to evaluation results JSON file (with logprobs)")
    parser.add_argument("--ground_truth", type=str, required=True,
                       help="Path to ground truth judge results JSON file (GPT-4o)")
    parser.add_argument("--model_name", type=str, default="Unknown",
                       help="Model name for display in report and plots")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots and statistics (default: same as eval results file)")

    args = parser.parse_args()

    # Set up paths
    eval_path = Path(args.eval_results)
    gt_path = Path(args.ground_truth)

    if not eval_path.exists():
        print(f"Error: Evaluation results file not found: {eval_path}")
        sys.exit(1)

    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = eval_path.parent / "plots"

    # Set up logging
    logger = setup_logging(output_dir)

    logger.info(f"Evaluation results file: {eval_path}")
    logger.info(f"Ground truth file: {gt_path}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Output directory: {output_dir}")

    # Load data
    eval_results = load_evaluation_results(eval_path, logger)
    gt_results = load_ground_truth_judge(gt_path, logger)

    if len(eval_results) == 0:
        logger.error("No evaluation results loaded. Exiting.")
        sys.exit(1)

    if len(gt_results) == 0:
        logger.error("No ground truth results loaded. Exiting.")
        sys.exit(1)

    # Extract self-preference probabilities
    logger.info("=" * 80)
    logger.info("EXTRACTING SELF-PREFERENCE PROBABILITIES")
    logger.info("=" * 80)

    probs = extract_self_preference_probs(eval_results, gt_results, logger)

    # Calculate paper metrics
    logger.info("")
    logger.info("=" * 80)
    logger.info("CALCULATING PAPER METRICS")
    logger.info("=" * 80)

    metrics = calculate_paper_metrics(eval_results, gt_results, logger)

    # Print summary report
    logger.info("")
    print_summary_report(metrics, args.model_name, logger)

    # Create visualizations
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    create_visualization(probs, args.model_name, output_dir, logger)

    # Save statistics to JSON
    stats_output = {
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "eval_results_file": str(eval_path),
        "ground_truth_file": str(gt_path),
        "total_examples": len(eval_results),
        "metrics": metrics,
        "probability_stats": {
            "lsp_mean": float(np.mean(probs["lsp"])) if len(probs["lsp"]) > 0 else 0.0,
            "ilsp_mean": float(np.mean(probs["ilsp"])) if len(probs["ilsp"]) > 0 else 0.0,
            "all_mean": float(np.mean(probs["all"])) if len(probs["all"]) > 0 else 0.0,
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
