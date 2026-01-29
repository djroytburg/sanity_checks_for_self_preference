#!/usr/bin/env python3
# analyze_cnn_self_preference.py: Analyze self-preference in CNN dataset results
# Created for analyzing gpt35_results.json with human model entries only

#DO NOT USE, NOT PART OF FINAL PAPER!!!!!!!
"""
Analyze self-preference metrics from CNN dataset evaluation results.

Computes distributions for:
- LSP (Legitimate Self-Preference): When human's response is objectively better
- ILSP (Illegitimate Self-Preference): When human's response is objectively worse

Uses forward_comparison_probability and backward_comparison_probability from the dataset.
In this dataset, we need ground truth about which response is objectively better.
"""

import argparse
import json
import logging
import os
import random
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
    log_file = log_dir / f"analyze_cnn_sp_{timestamp}.log"

    logger = logging.getLogger("analyze_cnn_sp")
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
    logger.info("CNN DATASET SELF-PREFERENCE ANALYSIS STARTED")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"User: {os.environ.get('USER', 'unknown')}")

    return logger


# ----------------------
# --- DATA LOADING   ---
# ----------------------

def load_cnn_results(json_path: Path, logger: logging.Logger) -> List[Dict]:
    """Load CNN evaluation results from JSON file."""
    logger.info(f"Loading CNN results from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} total examples")

    # Filter for human model only if 'model' field exists
    if results and 'model' in results[0]:
        human_results = [r for r in results if r.get("model") == "gpt35"]
        logger.info(f"Filtered to {len(human_results)} human examples")
    else:
        # If no 'model' field, assume all results are for human comparisons
        logger.info(f"No 'model' field found, using all {len(results)} examples")
        human_results = results

    return human_results


def load_ground_truth_labels(json_path: Path, logger: logging.Logger) -> List[Dict]:
    """
    Load ground truth labels from JSON file.

    Returns:
        List of ground truth dicts with article_index and verdict info.
    """
    logger.info(f"Loading ground truth labels from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    logger.info(f"Loaded {len(labels)} ground truth labels")
    return labels


def aggregate_verdict_from_answers(answer_original: str, answer_flipped: str) -> str:
    """
    Aggregate verdicts from original and flipped orderings.

    Args:
        answer_original: Answer when summary1 is in position 1 ("1" or "2")
        answer_flipped: Answer when summary2 is in position 1 ("1" or "2")

    Returns:
        "1" if prefers summary1, "2" if prefers summary2, "T" for tie
    """
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


def determine_human_is_correct(gt_item: Dict, cnn_item: Dict, logger: logging.Logger) -> tuple:
    """
    Determine if human response is correct based on ground truth verdict.

    The CNN results tell us which summary is human vs GPT-3.5.
    The ground truth tells us which summary is objectively better.

    Returns:
        Tuple of (human_correct: bool, is_tie: bool) or (None, False) for unclear detection
    """
    # Get ground truth verdict
    gt_verdict = aggregate_verdict_from_answers(
        gt_item["original_order"]["answer"],
        gt_item["flipped_order"]["answer"]
    )

    # Check if ground truth is a tie
    if gt_verdict == "T":
        return (None, True)

    # Determine which position human is in from CNN detection results
    # In CNN data:
    # - forward_detection tells us what was detected in position 1 (original order)
    # - "1" means position 1 is human, "2" means position 1 is GPT-3.5

    # Get the detection results to know which is human
    forward_detection = cnn_item.get("forward_detection")
    backward_detection = cnn_item.get("backward_detection")

    # Aggregate detection to determine which summary is human
    # This is similar to verdict aggregation
    if forward_detection == "1" and backward_detection == "2":
        # Both say summary1 is human
        human_position = "1"
    elif forward_detection == "2" and backward_detection == "1":
        # Both say summary2 is human
        human_position = "2"
    else:
        # Unclear detection - skip this example
        return (None, False)

    # Human is correct if ground truth verdict matches human position
    return (gt_verdict == human_position, False)


# ----------------------
# --- METRIC CALC    ---
# ----------------------

def extract_self_preference_probs(cnn_results: List[Dict],
                                  ground_truth: List[Dict],
                                  logger: logging.Logger) -> Dict:
    """
    Extract self-preference probabilities from CNN results.

    In the CNN data:
    - forward_comparison_probability: P(choosing position 1 | forward order)
    - backward_comparison_probability: P(choosing position 2 | backward order)

    We need to determine which position the human is in, then compute:
    Self-preference = probability of choosing human in both orderings

    Args:
        cnn_results: List of human model result dicts.
        ground_truth: List of ground truth dicts with verdicts.
        logger: Logger instance.

    Returns:
        Dict with keys lsp, ilsp, all containing self-pref probability lists.
    """
    logger.info("Extracting self-preference probabilities from CNN data")

    lsp_probs = []  # Legitimate self-preference (human correct)
    ilsp_probs = []  # Illegitimate self-preference (human incorrect)
    all_probs = []

    skipped_no_gt = 0
    skipped_missing_fields = 0
    skipped_unclear_detection = 0
    skipped_tie = 0

    # Build ground truth mapping by article_index
    gt_map = {gt["article_index"]: gt for gt in ground_truth}

    for cnn_item in cnn_results:
        key = cnn_item.get("key")

        # Check if we have ground truth for this key
        if key not in gt_map:
            skipped_no_gt += 1
            continue

        gt_item = gt_map[key]

        # Determine if human is correct
        human_correct, is_tie = determine_human_is_correct(gt_item, cnn_item, logger)

        if is_tie:
            skipped_tie += 1
            continue

        if human_correct is None:
            skipped_unclear_detection += 1
            continue

        # Get self-preference probability
        # Try pre-computed field first, then compute from individual probabilities
        if "self_preference" in cnn_item:
            # Use pre-computed self-preference field (xsum format)
            self_pref = cnn_item["self_preference"]
        elif ("forward_comparison_probability" in cnn_item and
              "backward_comparison_probability" in cnn_item):
            # Compute from individual probabilities (CNN format)
            forward_prob = cnn_item["forward_comparison_probability"]
            backward_prob = cnn_item["backward_comparison_probability"]
            # Self-preference = avg of P(1|forward) and P(2|backward)
            self_pref = (forward_prob + backward_prob) / 2.0
        else:
            # Missing required fields
            skipped_missing_fields += 1
            continue

        all_probs.append(self_pref)

        # Categorize as legitimate or illegitimate based on ground truth
        if human_correct:
            lsp_probs.append(self_pref)
        else:
            ilsp_probs.append(self_pref)

    logger.info(f"  Extracted probabilities:")
    logger.info(f"    Total: {len(all_probs)}")
    logger.info(f"    LSP (human correct): {len(lsp_probs)}")
    logger.info(f"    ILSP (human incorrect): {len(ilsp_probs)}")
    logger.info(f"    Skipped (no ground truth): {skipped_no_gt}")
    logger.info(f"    Skipped (missing fields): {skipped_missing_fields}")
    logger.info(f"    Skipped (unclear detection): {skipped_unclear_detection}")
    logger.info(f"    Skipped (ties): {skipped_tie}")

    # Balance LSP and ILSP to avoid skew in final distribution
    min_count = min(len(lsp_probs), len(ilsp_probs))
    if min_count > 0:
        logger.info(f"  Balancing LSP and ILSP to {min_count} examples each")
        # Set random seed for reproducibility
        random.seed(42)
        lsp_probs_balanced = random.sample(lsp_probs, min_count)
        ilsp_probs_balanced = random.sample(ilsp_probs, min_count)
        all_probs_balanced = lsp_probs_balanced + ilsp_probs_balanced
    else:
        logger.warning("  Cannot balance: one category has no examples")
        lsp_probs_balanced = lsp_probs
        ilsp_probs_balanced = ilsp_probs
        all_probs_balanced = all_probs

    return {
        "lsp": lsp_probs_balanced,
        "ilsp": ilsp_probs_balanced,
        "all": all_probs_balanced,
        "lsp_original_count": len(lsp_probs),
        "ilsp_original_count": len(ilsp_probs),
        "skipped_no_gt": skipped_no_gt,
        "skipped_missing_fields": skipped_missing_fields,
        "skipped_unclear_detection": skipped_unclear_detection,
        "skipped_ties": skipped_tie,
    }


def extract_self_preference_examples(cnn_results: List[Dict], ground_truth: List[Dict],
                                     logger: logging.Logger) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract examples of legitimate and harmful self-preference.

    Args:
        cnn_results: List of CNN result dicts.
        ground_truth: List of ground truth dicts with verdicts.
        logger: Logger instance.

    Returns:
        Tuple of (legitimate_examples, harmful_examples) lists
    """
    logger.info("Extracting self-preference examples")

    legitimate_examples = []
    harmful_examples = []

    # Build ground truth mapping by article_index
    gt_map = {gt["article_index"]: gt for gt in ground_truth}

    for cnn_item in cnn_results:
        key = cnn_item.get("key")

        # Check if we have ground truth for this key
        if key not in gt_map:
            continue

        gt_item = gt_map[key]

        # Determine if human is correct
        human_correct, is_tie = determine_human_is_correct(gt_item, cnn_item, logger)

        # Skip ties and unclear detections
        if is_tie or human_correct is None:
            continue

        # Get self-preference probability
        if "self_preference" in cnn_item:
            self_pref = cnn_item["self_preference"]
        elif ("forward_comparison_probability" in cnn_item and
              "backward_comparison_probability" in cnn_item):
            forward_prob = cnn_item["forward_comparison_probability"]
            backward_prob = cnn_item["backward_comparison_probability"]
            self_pref = (forward_prob + backward_prob) / 2.0
        else:
            continue

        # Determine if the model prefers the human response (self-preference)
        # We consider it self-preference if the probability is > 0.5
        prefers_self = self_pref > 0.5

        if prefers_self:
            example = {
                "key": key,
                "article": cnn_item.get("article", ""),
                "summary1": cnn_item.get("summary1", ""),
                "summary2": cnn_item.get("summary2", ""),
                "self_preference": float(self_pref),
                "forward_comparison_probability": cnn_item.get("forward_comparison_probability"),
                "backward_comparison_probability": cnn_item.get("backward_comparison_probability"),
                "forward_detection": cnn_item.get("forward_detection"),
                "backward_detection": cnn_item.get("backward_detection"),
                "human_correct": human_correct,
                "ground_truth_verdict": aggregate_verdict_from_answers(
                    gt_item["original_order"]["answer"],
                    gt_item["flipped_order"]["answer"]
                ),
            }

            if human_correct:
                legitimate_examples.append(example)
            else:
                harmful_examples.append(example)

    logger.info(f"  Extracted {len(legitimate_examples)} legitimate self-preference examples")
    logger.info(f"  Extracted {len(harmful_examples)} harmful self-preference examples")

    return legitimate_examples, harmful_examples


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
    fig.suptitle(f'Self-Preference Analysis - CNN Dataset\n{model_name}',
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

    # Calculate LSP proportion (should be 0.5 after balancing)
    lsp_prop = len(probs['lsp']) / (len(probs['lsp']) + len(probs['ilsp'])) if (len(probs['lsp']) + len(probs['ilsp'])) > 0 else 0

    # Plot all (balanced)
    plot_histogram_with_stats(
        ax=axes[2],
        data=probs['all'],
        objective=lsp_prop * 1.0 + (1 - lsp_prop) * 0.0,
        title="All Examples (Balanced)",
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
    png_path = output_dir / "cnn_self_preference_analysis.png"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved visualization to {png_path}")

    pdf_path = output_dir / "cnn_self_preference_analysis.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved visualization to {pdf_path}")

    plt.close()


# ----------------------
# --- MAIN ENTRY     ---
# ----------------------

def get_probabilities_from_cnn_data(cnn_item: Dict) -> Dict[str, float]:
    """
    Extract probabilities for positions "1" and "2" from CNN data.

    Returns:
        Dict with keys "1" and "2" mapping to their probabilities.
    """
    # Try pre-computed field first
    if "forward_comparison_probability" in cnn_item and "backward_comparison_probability" in cnn_item:
        # In CNN data:
        # forward_comparison_probability = P(choosing position 1 | forward order)
        # backward_comparison_probability = P(choosing position 2 | backward order)
        return {
            "1": cnn_item["forward_comparison_probability"],
            "2": 1.0 - cnn_item["forward_comparison_probability"]
        }
    else:
        # If no probabilities, return equal probability
        return {"1": 0.5, "2": 0.5}


def create_game_jsonl_files(cnn_results: List[Dict], ground_truth: List[Dict],
                            judge_name: str, reference_name: str, comparison_type: str,
                            dataset_name: str, output_dir: Path, logger: logging.Logger):
    """
    Create J_vs_R game1.jsonl and game2.jsonl files for CNN dataset.

    Game 1: Judge response is first (response1)
    Game 2: Reference response is first (response1)

    Args:
        cnn_results: List of CNN result dicts.
        ground_truth: List of ground truth dicts with verdicts.
        judge_name: Name of the judge model.
        reference_name: Name of the reference model.
        comparison_type: Type of comparison (should be "J_vs_R").
        dataset_name: Name of the dataset.
        output_dir: Output directory for the JSONL files.
        logger: Logger instance.
    """
    logger.info(f"Creating game JSONL files for {comparison_type}...")

    # Build ground truth mapping by key
    gt_map = {gt["article_index"]: gt for gt in ground_truth}

    game1_records = []
    game2_records = []

    for cnn_item in cnn_results:
        key = cnn_item.get("key")

        # Get corresponding ground truth
        if key not in gt_map:
            continue

        gt_item = gt_map[key]

        # Determine if human (judge) is correct based on ground truth
        human_correct, is_tie = determine_human_is_correct(gt_item, cnn_item, logger)

        # Skip ties and unclear detections
        if is_tie or human_correct is None:
            continue

        # Determine category (lsp = legitimate self-preference, ilsp = illegitimate)
        category = "lsp" if human_correct else "ilsp"

        # Get probabilities
        # For game1 (original order): judge is position 1
        if "forward_comparison_probability" in cnn_item:
            prob_judge_original = cnn_item["forward_comparison_probability"]
            prob_ref_original = 1.0 - prob_judge_original
        else:
            prob_judge_original = 0.5
            prob_ref_original = 0.5

        # For game2 (flipped order): reference is position 1, judge is position 2
        if "backward_comparison_probability" in cnn_item:
            prob_ref_flipped = cnn_item.get("backward_comparison_probability", 0.5)
            prob_judge_flipped = 1.0 - prob_ref_flipped
        else:
            prob_ref_flipped = 0.5
            prob_judge_flipped = 0.5

        # Get the judge's answer from CNN data
        forward_answer = cnn_item.get("forward_comparison", "T")
        backward_answer = cnn_item.get("backward_comparison", "T")

        # Game 1: Judge is response1 (original order)
        game1_record = {
            "cache_key": key,
            "example_id": key,
            "judge": judge_name,
            "reference": reference_name,
            "comparison_type": f"{comparison_type}_game1",
            "response1_key": "judge",
            "response2_key": "reference",
            "prob_response1": prob_judge_original,
            "prob_response2": prob_ref_original,
            "generated_text": forward_answer if forward_answer in ["1", "2"] else "T",
            "raw_probs": [prob_judge_original, prob_ref_original],
            "normalized_sum": prob_judge_original + prob_ref_original,
            "category": category,
            "dataset": dataset_name
        }
        game1_records.append(game1_record)

        # Game 2: Reference is response1 (flipped order)
        game2_record = {
            "cache_key": key,
            "example_id": key,
            "judge": judge_name,
            "reference": reference_name,
            "comparison_type": f"{comparison_type}_game2",
            "response1_key": "reference",
            "response2_key": "judge",
            "prob_response1": prob_ref_flipped,
            "prob_response2": prob_judge_flipped,
            "generated_text": backward_answer if backward_answer in ["1", "2"] else "T",
            "raw_probs": [prob_ref_flipped, prob_judge_flipped],
            "normalized_sum": prob_ref_flipped + prob_judge_flipped,
            "category": category,
            "dataset": dataset_name
        }
        game2_records.append(game2_record)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write game1 JSONL file
    game1_path = output_dir / f"{comparison_type}_game1.jsonl"
    with open(game1_path, 'w', encoding='utf-8') as f:
        for record in game1_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(game1_records)} records to {game1_path}")

    # Write game2 JSONL file
    game2_path = output_dir / f"{comparison_type}_game2.jsonl"
    with open(game2_path, 'w', encoding='utf-8') as f:
        for record in game2_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(game2_records)} records to {game2_path}")

    # Return the output directory
    return output_dir


def print_summary_report(probs: Dict, model_name: str, logger: logging.Logger):
    """Print comprehensive summary report."""
    logger.info("=" * 80)
    logger.info("CNN SELF-PREFERENCE ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info("")

    logger.info("STATISTICS")
    logger.info("-" * 80)
    logger.info(f"  LSP mean:                              {np.mean(probs['lsp']):.4f}")
    logger.info(f"  ILSP mean:                             {np.mean(probs['ilsp']):.4f}")
    logger.info(f"  All mean (balanced):                   {np.mean(probs['all']):.4f}")
    logger.info(f"  LSP count (balanced):                  {len(probs['lsp'])}")
    logger.info(f"  ILSP count (balanced):                 {len(probs['ilsp'])}")
    logger.info(f"  LSP count (original):                  {probs['lsp_original_count']}")
    logger.info(f"  ILSP count (original):                 {probs['ilsp_original_count']}")
    logger.info("")
    logger.info("=" * 80)


def main():
    """Main entry point for CNN self-preference analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze self-preference metrics from CNN dataset evaluation results"
    )
    parser.add_argument("--cnn_results", type=str, required=True,
                       help="Path to CNN results JSON file (e.g., gpt35_results.json)")
    parser.add_argument("--ground_truth", type=str, required=True,
                       help="Path to ground truth judge results JSON file (same format as evaluation results)")
    parser.add_argument("--model_name", type=str, default="Human (CNN Dataset)",
                       help="Model name for display in report and plots")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots and statistics (default: same as results file)")
    parser.add_argument("--judge_name", type=str, default=None,
                       help="Name of the judge model for JSONL output (default: same as model_name)")
    parser.add_argument("--reference_name", type=str, default="reference",
                       help="Name of the reference model for JSONL output (default: 'reference')")
    parser.add_argument("--comparison_type", type=str, default="J_vs_R",
                       help="Type of comparison for JSONL output (default: 'J_vs_R')")
    parser.add_argument("--dataset", type=str, default="cnn_dailymail",
                       help="Name of the dataset for JSONL output (default: 'cnn_dailymail')")

    args = parser.parse_args()

    # Set up paths
    cnn_path = Path(args.cnn_results)
    gt_path = Path(args.ground_truth)

    if not cnn_path.exists():
        print(f"Error: CNN results file not found: {cnn_path}")
        sys.exit(1)

    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {gt_path}")
        sys.exit(1)

    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = cnn_path.parent / "plots"

    # For logging, use base_output_dir
    output_dir = base_output_dir

    # Set judge_name to model_name if not specified
    judge_name = args.judge_name if args.judge_name else args.model_name
    reference_name = args.reference_name
    comparison_type = args.comparison_type
    dataset_name = args.dataset

    # Set up logging
    logger = setup_logging(output_dir)

    logger.info(f"CNN results file: {cnn_path}")
    logger.info(f"Ground truth file: {gt_path}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Judge name: {judge_name}")
    logger.info(f"Reference name: {reference_name}")
    logger.info(f"Comparison type: {comparison_type}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Output directory: {output_dir}")

    # Load data
    cnn_results = load_cnn_results(cnn_path, logger)
    ground_truth = load_ground_truth_labels(gt_path, logger)

    if len(cnn_results) == 0:
        logger.error("No human results loaded. Exiting.")
        sys.exit(1)

    if len(ground_truth) == 0:
        logger.error("No ground truth labels loaded. Exiting.")
        sys.exit(1)

    # Extract self-preference probabilities
    logger.info("=" * 80)
    logger.info("EXTRACTING SELF-PREFERENCE PROBABILITIES")
    logger.info("=" * 80)

    probs = extract_self_preference_probs(cnn_results, ground_truth, logger)

    # Extract self-preference examples
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXTRACTING SELF-PREFERENCE EXAMPLES")
    logger.info("=" * 80)

    legitimate_examples, harmful_examples = extract_self_preference_examples(cnn_results, ground_truth, logger)

    # Print summary report
    logger.info("")
    print_summary_report(probs, args.model_name, logger)

    # Create game JSONL files and get the final output directory
    # This must be done BEFORE saving statistics so everything goes in the same folder
    logger.info("")
    logger.info("=" * 80)
    logger.info("CREATING GAME JSONL FILES")
    logger.info("=" * 80)

    output_dir = create_game_jsonl_files(cnn_results, ground_truth, judge_name, reference_name,
                                         comparison_type, dataset_name, base_output_dir, logger)

    # Create visualizations
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    create_visualization(probs, args.model_name, output_dir, logger)

    # Save statistics to JSON
    stats_output = {
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "cnn_results_file": str(cnn_path),
        "ground_truth_file": str(gt_path),
        "total_human_examples": len(cnn_results),
        "probability_stats": {
            "lsp_mean": float(np.mean(probs["lsp"])) if len(probs["lsp"]) > 0 else 0.0,
            "ilsp_mean": float(np.mean(probs["ilsp"])) if len(probs["ilsp"]) > 0 else 0.0,
            "all_mean": float(np.mean(probs["all"])) if len(probs["all"]) > 0 else 0.0,
            "lsp_count_balanced": len(probs["lsp"]),
            "ilsp_count_balanced": len(probs["ilsp"]),
            "lsp_count_original": probs.get("lsp_original_count", len(probs["lsp"])),
            "ilsp_count_original": probs.get("ilsp_original_count", len(probs["ilsp"])),
            "lsp_probs": probs["lsp"],
            "ilsp_probs": probs["ilsp"],
        },
    }

    stats_path = output_dir / "cnn_self_preference_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved statistics to {stats_path}")

    # Save legitimate self-preference examples
    legitimate_path = output_dir / "legitimate_self_preference_examples.json"
    legitimate_output = {
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "description": "Examples where the model prefers its own response AND is correct (legitimate self-preference)",
        "count": len(legitimate_examples),
        "examples": legitimate_examples,
    }
    with open(legitimate_path, 'w', encoding='utf-8') as f:
        json.dump(legitimate_output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(legitimate_examples)} legitimate self-preference examples to {legitimate_path}")

    # Save harmful self-preference examples
    harmful_path = output_dir / "harmful_self_preference_examples.json"
    harmful_output = {
        "model_name": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "description": "Examples where the model prefers its own response BUT is incorrect (harmful self-preference)",
        "count": len(harmful_examples),
        "examples": harmful_examples,
    }
    with open(harmful_path, 'w', encoding='utf-8') as f:
        json.dump(harmful_output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(harmful_examples)} harmful self-preference examples to {harmful_path}")

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
