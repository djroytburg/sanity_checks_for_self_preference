#!/usr/bin/env python3
# analyze_judge_swap_cnn.py: Analyze judge swap for CNN dataset
# Compares J(J vs R) with J(K vs R) where:
#   J = GPT-3.5 (judge)
#   K = Llama (proxy)
#   R = Human (reference)

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import bootstrap


# ----------------------
# --- LOGGING SETUP  ---
# ----------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging."""
    log_dir = Path("file_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"judge_swap_cnn_{timestamp}.log"

    logger = logging.getLogger("judge_swap_cnn")
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
    logger.info("JUDGE SWAP ANALYSIS - CNN DATASET")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")

    return logger


# ----------------------
# --- DATA LOADING   ---
# ----------------------

def load_raw_probabilities(eval_file: Path, gt_file: Path, logger: logging.Logger) -> Dict:
    """
    Load raw probability data from evaluation and ground truth files.

    This function handles two data formats:
    1. New format: Items with 'key', 'forward_comparison', 'backward_comparison', etc.
    2. Old format: Items with 'article_index', 'original_order.top_logprobs', etc.

    Args:
        eval_file: Path to evaluation results JSON
        gt_file: Path to ground truth JSON
        logger: Logger instance

    Returns:
        Dict with 'lsp' and 'ilsp' lists of probabilities
    """
    logger.info(f"  Loading raw data from {eval_file}")

    if not eval_file.exists():
        logger.warning(f"  Evaluation file not found: {eval_file}")
        return None

    if not gt_file.exists():
        logger.warning(f"  Ground truth file not found: {gt_file}")
        return None

    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    # Build ground truth map
    gt_map = {}
    for gt_item in gt_data:
        key = gt_item.get('article_index')
        if key:
            gt_map[key] = gt_item

    lsp_probs = []
    ilsp_probs = []

    # Check if this is the new format (has 'key' and 'forward_comparison')
    if eval_data and 'key' in eval_data[0] and 'forward_comparison' in eval_data[0]:
        # For new format, filter to:
        # 1. Only include model='human' comparisons
        # 2. Only include items that match GT keys
        # 3. Deduplicate - keep only first instance of each key
        original_count = len(eval_data)
        seen_keys = set()
        filtered_data = []
        for item in eval_data:
            key = item.get('key')
            model = item.get('model')
            if model == 'human' and key in gt_map and key not in seen_keys:
                filtered_data.append(item)
                seen_keys.add(key)
        eval_data = filtered_data
        logger.info(f"    Filtered to {len(eval_data)} unique 'human' examples (from {original_count}) matching GT keys")
        logger.info(f"    Detected new data format with 'key' and comparison fields")

        for eval_item in eval_data:
            key = eval_item.get('key')
            if not key or key not in gt_map:
                continue

            gt_item = gt_map[key]

            # Get evaluation verdicts from forward/backward comparisons
            eval_forward = eval_item.get('forward_comparison')
            eval_backward = eval_item.get('backward_comparison')

            # Get ground truth verdicts
            gt_answer_orig = gt_item.get('original_order', {}).get('answer')
            gt_answer_flip = gt_item.get('flipped_order', {}).get('answer')

            if not all([eval_forward, eval_backward, gt_answer_orig, gt_answer_flip]):
                continue

            # Aggregate eval verdicts
            if eval_forward == "1" and eval_backward == "2":
                eval_verdict = "1"
            elif eval_forward == "2" and eval_backward == "1":
                eval_verdict = "2"
            else:
                eval_verdict = "T"

            # Aggregate GT verdicts
            if gt_answer_orig == "1" and gt_answer_flip == "2":
                gt_verdict = "1"
            elif gt_answer_orig == "2" and gt_answer_flip == "1":
                gt_verdict = "2"
            else:
                gt_verdict = "T"

            # Don't filter out ties - include all cases
            judge_correct = (eval_verdict == gt_verdict)

            # Get probabilities
            # Try to get individual probabilities first
            p1_orig = eval_item.get('forward_comparison_probability')
            p2_flip = eval_item.get('backward_comparison_probability')

            if p1_orig is not None and p2_flip is not None:
                # Use averaged probability from forward and backward
                self_pref = (p1_orig + p2_flip) / 2.0
            elif 'self_preference' in eval_item:
                # Fall back to self_preference field if individual probs not available
                self_pref = eval_item.get('self_preference')
            else:
                continue

            if judge_correct:
                lsp_probs.append(self_pref)
            else:
                ilsp_probs.append(self_pref)

    else:
        # Old format with article_index and top_logprobs
        logger.info(f"    Detected old data format with 'article_index' and logprobs")

        for eval_item in eval_data:
            key = eval_item.get('article_index')
            if not key or key not in gt_map:
                continue

            gt_item = gt_map[key]

            # Determine judge correctness
            eval_answer_orig = eval_item.get('original_order', {}).get('answer')
            eval_answer_flip = eval_item.get('flipped_order', {}).get('answer')

            gt_answer_orig = gt_item.get('original_order', {}).get('answer')
            gt_answer_flip = gt_item.get('flipped_order', {}).get('answer')

            if not all([eval_answer_orig, eval_answer_flip, gt_answer_orig, gt_answer_flip]):
                continue

            # Aggregate verdicts
            if eval_answer_orig == "1" and eval_answer_flip == "2":
                eval_verdict = "1"
            elif eval_answer_orig == "2" and eval_answer_flip == "1":
                eval_verdict = "2"
            else:
                eval_verdict = "T"

            if gt_answer_orig == "1" and gt_answer_flip == "2":
                gt_verdict = "1"
            elif gt_answer_orig == "2" and gt_answer_flip == "1":
                gt_verdict = "2"
            else:
                gt_verdict = "T"

            # Don't filter out ties - include all cases
            judge_correct = (eval_verdict == gt_verdict)

            # Extract self-preference probability
            orig_logprobs = eval_item.get('original_order', {}).get('top_logprobs', [])
            flip_logprobs = eval_item.get('flipped_order', {}).get('top_logprobs', [])

            # Get P(1|original) and P(2|flipped)
            p1_orig = None
            p2_flip = None

            for logprob_item in orig_logprobs:
                if logprob_item.get('token') == '1':
                    p1_orig = logprob_item.get('probability')
                    break

            for logprob_item in flip_logprobs:
                if logprob_item.get('token') == '2':
                    p2_flip = logprob_item.get('probability')
                    break

            if p1_orig is None or p2_flip is None:
                continue

            self_pref = (p1_orig + p2_flip) / 2.0

            if judge_correct:
                lsp_probs.append(self_pref)
            else:
                ilsp_probs.append(self_pref)

    logger.info(f"    Loaded {len(lsp_probs)} LSP and {len(ilsp_probs)} ILSP probabilities")

    # Balance (same logic as in the analysis scripts)
    import random
    random.seed(42)
    min_count = min(len(lsp_probs), len(ilsp_probs))
    if min_count > 0:
        lsp_probs = random.sample(lsp_probs, min_count)
        ilsp_probs = random.sample(ilsp_probs, min_count)

    return {
        'lsp': lsp_probs,
        'ilsp': ilsp_probs,
        'all': lsp_probs + ilsp_probs
    }


# ----------------------
# --- STATISTICS     ---
# ----------------------

def compute_hypothesis_tests(stats_j: Dict, stats_k: Dict,
                            probs_j: Dict = None, probs_k: Dict = None,
                            logger: logging.Logger = None) -> Dict:
    """
    Compute comparison statistics between J(J vs R) and J(K vs R).

    If raw probability data is provided, performs t-test and KS test.
    Otherwise uses two-sample proportion tests on summary statistics.

    Args:
        stats_j: Statistics dict for J(J vs R)
        stats_k: Statistics dict for J(K vs R)
        probs_j: Optional raw probability data for J condition
        probs_k: Optional raw probability data for K condition
        logger: Logger instance
    """
    result = {}

    # Extract means and counts
    j_lsp_mean = stats_j['probability_stats']['lsp_mean']
    j_ilsp_mean = stats_j['probability_stats']['ilsp_mean']
    j_all_mean = stats_j['probability_stats']['all_mean']
    j_lsp_count = stats_j['probability_stats']['lsp_count_balanced']
    j_ilsp_count = stats_j['probability_stats']['ilsp_count_balanced']

    k_lsp_mean = stats_k['probability_stats']['lsp_mean']
    k_ilsp_mean = stats_k['probability_stats']['ilsp_mean']
    k_all_mean = stats_k['probability_stats']['all_mean']
    k_lsp_count = stats_k['probability_stats']['lsp_count_balanced']
    k_ilsp_count = stats_k['probability_stats']['ilsp_count_balanced']

    # Compute differences
    result['lsp_diff'] = j_lsp_mean - k_lsp_mean
    result['ilsp_diff'] = j_ilsp_mean - k_ilsp_mean
    result['all_diff'] = j_all_mean - k_all_mean

    result['j_lsp_mean'] = j_lsp_mean
    result['j_ilsp_mean'] = j_ilsp_mean
    result['j_all_mean'] = j_all_mean
    result['j_lsp_count'] = j_lsp_count
    result['j_ilsp_count'] = j_ilsp_count

    result['k_lsp_mean'] = k_lsp_mean
    result['k_ilsp_mean'] = k_ilsp_mean
    result['k_all_mean'] = k_all_mean
    result['k_lsp_count'] = k_lsp_count
    result['k_ilsp_count'] = k_ilsp_count

    # Two-sample z-test for proportions (approximation)
    # H0: p_j = p_k (no difference in proportions)
    # For ILSP (most important for self-preference bias detection)

    # Pooled proportion for ILSP
    p_pooled_ilsp = (j_ilsp_mean * j_ilsp_count + k_ilsp_mean * k_ilsp_count) / (j_ilsp_count + k_ilsp_count)
    se_ilsp = np.sqrt(p_pooled_ilsp * (1 - p_pooled_ilsp) * (1/j_ilsp_count + 1/k_ilsp_count))

    if se_ilsp > 0:
        z_ilsp = result['ilsp_diff'] / se_ilsp
        # Two-tailed p-value
        p_value_ilsp_two = 2 * (1 - stats.norm.cdf(abs(z_ilsp)))
        # One-tailed p-value (testing if J > K, i.e., self-preference)
        p_value_ilsp_one = 1 - stats.norm.cdf(z_ilsp)
    else:
        z_ilsp = 0
        p_value_ilsp_two = 1.0
        p_value_ilsp_one = 1.0

    result['ilsp_z_statistic'] = z_ilsp
    result['ilsp_p_value_two_sided'] = p_value_ilsp_two
    result['ilsp_p_value_one_sided'] = p_value_ilsp_one

    # For LSP
    p_pooled_lsp = (j_lsp_mean * j_lsp_count + k_lsp_mean * k_lsp_count) / (j_lsp_count + k_lsp_count)
    se_lsp = np.sqrt(p_pooled_lsp * (1 - p_pooled_lsp) * (1/j_lsp_count + 1/k_lsp_count))

    if se_lsp > 0:
        z_lsp = result['lsp_diff'] / se_lsp
        p_value_lsp_two = 2 * (1 - stats.norm.cdf(abs(z_lsp)))
        p_value_lsp_one = 1 - stats.norm.cdf(z_lsp)
    else:
        z_lsp = 0
        p_value_lsp_two = 1.0
        p_value_lsp_one = 1.0

    result['lsp_z_statistic'] = z_lsp
    result['lsp_p_value_two_sided'] = p_value_lsp_two
    result['lsp_p_value_one_sided'] = p_value_lsp_one

    # Effect size (Cohen's h for proportions)
    result['ilsp_cohens_h'] = 2 * (np.arcsin(np.sqrt(j_ilsp_mean)) - np.arcsin(np.sqrt(k_ilsp_mean)))
    result['lsp_cohens_h'] = 2 * (np.arcsin(np.sqrt(j_lsp_mean)) - np.arcsin(np.sqrt(k_lsp_mean)))

    # If raw probability data is available, perform t-test and KS test
    if probs_j is not None and probs_k is not None:
        if logger:
            logger.info("")
            logger.info("Computing additional tests with raw probability data...")
            logger.info("Balancing sample sizes between J and K conditions...")

        # Balance sample sizes between J and K for fair comparison
        import random
        random.seed(42)

        # Balance ILSP samples
        if len(probs_j['ilsp']) > 0 and len(probs_k['ilsp']) > 0:
            min_ilsp = min(len(probs_j['ilsp']), len(probs_k['ilsp']))
            probs_j['ilsp'] = random.sample(probs_j['ilsp'], min_ilsp)
            probs_k['ilsp'] = random.sample(probs_k['ilsp'], min_ilsp)
            if logger:
                logger.info(f"  Balanced ILSP to {min_ilsp} samples each")

            # Update means and counts to use balanced samples
            j_ilsp_mean = np.mean(probs_j['ilsp'])
            k_ilsp_mean = np.mean(probs_k['ilsp'])
            j_ilsp_count = len(probs_j['ilsp'])
            k_ilsp_count = len(probs_k['ilsp'])
            result['ilsp_diff'] = j_ilsp_mean - k_ilsp_mean

        # Balance LSP samples
        if len(probs_j['lsp']) > 0 and len(probs_k['lsp']) > 0:
            min_lsp = min(len(probs_j['lsp']), len(probs_k['lsp']))
            probs_j['lsp'] = random.sample(probs_j['lsp'], min_lsp)
            probs_k['lsp'] = random.sample(probs_k['lsp'], min_lsp)
            if logger:
                logger.info(f"  Balanced LSP to {min_lsp} samples each")

            # Update means and counts to use balanced samples
            j_lsp_mean = np.mean(probs_j['lsp'])
            k_lsp_mean = np.mean(probs_k['lsp'])
            j_lsp_count = len(probs_j['lsp'])
            k_lsp_count = len(probs_k['lsp'])
            result['lsp_diff'] = j_lsp_mean - k_lsp_mean

        # Update all_mean
        j_all_mean = (j_lsp_mean * j_lsp_count + j_ilsp_mean * j_ilsp_count) / (j_lsp_count + j_ilsp_count)
        k_all_mean = (k_lsp_mean * k_lsp_count + k_ilsp_mean * k_ilsp_count) / (k_lsp_count + k_ilsp_count)
        result['all_diff'] = j_all_mean - k_all_mean

        # Update result dict with balanced values
        result['j_lsp_mean'] = j_lsp_mean
        result['j_ilsp_mean'] = j_ilsp_mean
        result['j_all_mean'] = j_all_mean
        result['j_lsp_count'] = j_lsp_count
        result['j_ilsp_count'] = j_ilsp_count

        result['k_lsp_mean'] = k_lsp_mean
        result['k_ilsp_mean'] = k_ilsp_mean
        result['k_all_mean'] = k_all_mean
        result['k_lsp_count'] = k_lsp_count
        result['k_ilsp_count'] = k_ilsp_count

        # Recalculate z-tests with balanced samples
        # Pooled proportion for ILSP
        p_pooled_ilsp = (j_ilsp_mean * j_ilsp_count + k_ilsp_mean * k_ilsp_count) / (j_ilsp_count + k_ilsp_count)
        se_ilsp = np.sqrt(p_pooled_ilsp * (1 - p_pooled_ilsp) * (1/j_ilsp_count + 1/k_ilsp_count))

        if se_ilsp > 0:
            z_ilsp = result['ilsp_diff'] / se_ilsp
            p_value_ilsp_two = 2 * (1 - stats.norm.cdf(abs(z_ilsp)))
            p_value_ilsp_one = 1 - stats.norm.cdf(z_ilsp)
        else:
            z_ilsp = 0
            p_value_ilsp_two = 1.0
            p_value_ilsp_one = 1.0

        result['ilsp_z_statistic'] = z_ilsp
        result['ilsp_p_value_two_sided'] = p_value_ilsp_two
        result['ilsp_p_value_one_sided'] = p_value_ilsp_one

        # Pooled proportion for LSP
        p_pooled_lsp = (j_lsp_mean * j_lsp_count + k_lsp_mean * k_lsp_count) / (j_lsp_count + k_lsp_count)
        se_lsp = np.sqrt(p_pooled_lsp * (1 - p_pooled_lsp) * (1/j_lsp_count + 1/k_lsp_count))

        if se_lsp > 0:
            z_lsp = result['lsp_diff'] / se_lsp
            p_value_lsp_two = 2 * (1 - stats.norm.cdf(abs(z_lsp)))
            p_value_lsp_one = 1 - stats.norm.cdf(z_lsp)
        else:
            z_lsp = 0
            p_value_lsp_two = 1.0
            p_value_lsp_one = 1.0

        result['lsp_z_statistic'] = z_lsp
        result['lsp_p_value_two_sided'] = p_value_lsp_two
        result['lsp_p_value_one_sided'] = p_value_lsp_one

        # Effect size (Cohen's h for proportions)
        result['ilsp_cohens_h'] = 2 * (np.arcsin(np.sqrt(j_ilsp_mean)) - np.arcsin(np.sqrt(k_ilsp_mean)))
        result['lsp_cohens_h'] = 2 * (np.arcsin(np.sqrt(j_lsp_mean)) - np.arcsin(np.sqrt(k_lsp_mean)))

        # T-test for ILSP
        if len(probs_j['ilsp']) > 0 and len(probs_k['ilsp']) > 0:
            t_stat_ilsp, t_p_two_ilsp = stats.ttest_ind(probs_j['ilsp'], probs_k['ilsp'])
            # One-sided p-value
            t_p_one_ilsp = t_p_two_ilsp / 2 if t_stat_ilsp > 0 else 1 - (t_p_two_ilsp / 2)

            result['ilsp_t_statistic'] = t_stat_ilsp
            result['ilsp_t_p_value_two_sided'] = t_p_two_ilsp
            result['ilsp_t_p_value_one_sided'] = t_p_one_ilsp
        else:
            result['ilsp_t_statistic'] = None
            result['ilsp_t_p_value_two_sided'] = None
            result['ilsp_t_p_value_one_sided'] = None

        # KS test for ILSP
        if len(probs_j['ilsp']) > 0 and len(probs_k['ilsp']) > 0:
            ks_stat_ilsp, ks_p_ilsp = stats.ks_2samp(probs_j['ilsp'], probs_k['ilsp'])
            result['ilsp_ks_statistic'] = ks_stat_ilsp
            result['ilsp_ks_p_value'] = ks_p_ilsp
        else:
            result['ilsp_ks_statistic'] = None
            result['ilsp_ks_p_value'] = None

        # T-test for LSP
        if len(probs_j['lsp']) > 0 and len(probs_k['lsp']) > 0:
            t_stat_lsp, t_p_two_lsp = stats.ttest_ind(probs_j['lsp'], probs_k['lsp'])
            # One-sided p-value
            t_p_one_lsp = t_p_two_lsp / 2 if t_stat_lsp > 0 else 1 - (t_p_two_lsp / 2)

            result['lsp_t_statistic'] = t_stat_lsp
            result['lsp_t_p_value_two_sided'] = t_p_two_lsp
            result['lsp_t_p_value_one_sided'] = t_p_one_lsp
        else:
            result['lsp_t_statistic'] = None
            result['lsp_t_p_value_two_sided'] = None
            result['lsp_t_p_value_one_sided'] = None

        # KS test for LSP
        if len(probs_j['lsp']) > 0 and len(probs_k['lsp']) > 0:
            ks_stat_lsp, ks_p_lsp = stats.ks_2samp(probs_j['lsp'], probs_k['lsp'])
            result['lsp_ks_statistic'] = ks_stat_lsp
            result['lsp_ks_p_value'] = ks_p_lsp
        else:
            result['lsp_ks_statistic'] = None
            result['lsp_ks_p_value'] = None

    logger.info("=" * 80)
    logger.info("HYPOTHESIS TEST RESULTS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("J(J vs R) - GPT-3.5 judging GPT-3.5 vs Human:")
    logger.info(f"  LSP mean:  {j_lsp_mean:.4f} (n={j_lsp_count})")
    logger.info(f"  ILSP mean: {j_ilsp_mean:.4f} (n={j_ilsp_count})")
    logger.info(f"  All mean:  {j_all_mean:.4f}")
    logger.info("")
    logger.info("J(K vs R) - GPT-3.5 judging Llama vs Human:")
    logger.info(f"  LSP mean:  {k_lsp_mean:.4f} (n={k_lsp_count})")
    logger.info(f"  ILSP mean: {k_ilsp_mean:.4f} (n={k_ilsp_count})")
    logger.info(f"  All mean:  {k_all_mean:.4f}")
    logger.info("")
    logger.info("Differences (J - K):")
    logger.info(f"  LSP:  {result['lsp_diff']:+.4f}")
    logger.info(f"  ILSP: {result['ilsp_diff']:+.4f} (KEY METRIC)")
    logger.info(f"  All:  {result['all_diff']:+.4f}")
    logger.info("")
    logger.info("Statistical Tests:")
    logger.info("")
    logger.info("ILSP (Illegitimate Self-Preference):")
    logger.info(f"  Z-test (Proportions):")
    logger.info(f"    Z-statistic:         {z_ilsp:+.4f}")
    logger.info(f"    P-value (two-sided): {p_value_ilsp_two:.6f}")
    logger.info(f"    P-value (one-sided): {p_value_ilsp_one:.6f}")
    logger.info(f"    Cohen's h:           {result['ilsp_cohens_h']:+.4f}")
    logger.info(f"    Significance (α=0.05): {'SIGNIFICANT' if p_value_ilsp_two < 0.05 else 'NOT SIGNIFICANT'}")

    # Add t-test results if available
    if 'ilsp_t_statistic' in result and result['ilsp_t_statistic'] is not None:
        logger.info(f"  T-test (Independent Samples):")
        logger.info(f"    T-statistic:         {result['ilsp_t_statistic']:+.4f}")
        logger.info(f"    P-value (two-sided): {result['ilsp_t_p_value_two_sided']:.6f}")
        logger.info(f"    P-value (one-sided): {result['ilsp_t_p_value_one_sided']:.6f}")
        logger.info(f"    Significance (α=0.05): {'SIGNIFICANT' if result['ilsp_t_p_value_two_sided'] < 0.05 else 'NOT SIGNIFICANT'}")

    # Add KS test results if available
    if 'ilsp_ks_statistic' in result and result['ilsp_ks_statistic'] is not None:
        logger.info(f"  KS test (Distributions):")
        logger.info(f"    KS-statistic:        {result['ilsp_ks_statistic']:+.4f}")
        logger.info(f"    P-value:             {result['ilsp_ks_p_value']:.6f}")
        logger.info(f"    Significance (α=0.05): {'SIGNIFICANT' if result['ilsp_ks_p_value'] < 0.05 else 'NOT SIGNIFICANT'}")

    logger.info("")
    logger.info("LSP (Legitimate Self-Preference):")
    logger.info(f"  Z-test (Proportions):")
    logger.info(f"    Z-statistic:         {z_lsp:+.4f}")
    logger.info(f"    P-value (two-sided): {p_value_lsp_two:.6f}")
    logger.info(f"    P-value (one-sided): {p_value_lsp_one:.6f}")
    logger.info(f"    Cohen's h:           {result['lsp_cohens_h']:+.4f}")
    logger.info(f"    Significance (α=0.05): {'SIGNIFICANT' if p_value_lsp_two < 0.05 else 'NOT SIGNIFICANT'}")

    # Add t-test results if available
    if 'lsp_t_statistic' in result and result['lsp_t_statistic'] is not None:
        logger.info(f"  T-test (Independent Samples):")
        logger.info(f"    T-statistic:         {result['lsp_t_statistic']:+.4f}")
        logger.info(f"    P-value (two-sided): {result['lsp_t_p_value_two_sided']:.6f}")
        logger.info(f"    P-value (one-sided): {result['lsp_t_p_value_one_sided']:.6f}")
        logger.info(f"    Significance (α=0.05): {'SIGNIFICANT' if result['lsp_t_p_value_two_sided'] < 0.05 else 'NOT SIGNIFICANT'}")

    # Add KS test results if available
    if 'lsp_ks_statistic' in result and result['lsp_ks_statistic'] is not None:
        logger.info(f"  KS test (Distributions):")
        logger.info(f"    KS-statistic:        {result['lsp_ks_statistic']:+.4f}")
        logger.info(f"    P-value:             {result['lsp_ks_p_value']:.6f}")
        logger.info(f"    Significance (α=0.05): {'SIGNIFICANT' if result['lsp_ks_p_value'] < 0.05 else 'NOT SIGNIFICANT'}")

    logger.info("")

    # Interpretation
    logger.info("INTERPRETATION:")
    if p_value_ilsp_two < 0.05:
        if result['ilsp_diff'] > 0:
            logger.info("  ✓ SIGNIFICANT self-preference bias detected (p < 0.05)")
            logger.info("  GPT-3.5 shows higher preference for its own (incorrect) responses")
            logger.info("  compared to Llama's (incorrect) responses when both are wrong.")
        else:
            logger.info("  ✗ SIGNIFICANT negative bias detected (p < 0.05)")
            logger.info("  GPT-3.5 actually prefers Llama's incorrect responses MORE")
            logger.info("  than its own incorrect responses (unexpected result).")
    else:
        logger.info("  No statistically significant self-preference bias detected (p >= 0.05)")
        logger.info("  GPT-3.5 does not show significant preference for its own responses")
        logger.info("  over Llama's responses when both are incorrect.")

    logger.info("=" * 80)

    return result


# ----------------------
# --- VISUALIZATION  ---
# ----------------------

def plot_histogram_with_stats(ax, data, objective, title=None, xlabel=None,
                               color='cornflowerblue', no_spine=False, **hist_kwargs):
    """Plot histogram with mean and objective score (for compatibility)."""
    # Since we only have mean, we'll create a simple bar
    mean = data

    # Create a simple representation
    ax.barh([0], [mean], color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(objective, color='red', linestyle='--', linewidth=2, label='Objective')

    if title:
        ax.set_title(title, fontweight='bold', fontsize=11, pad=15)
        subtitle = f"μ: {mean:.3f} | Obj: {objective:.3f}"
        ax.text(0.5, 0.98, subtitle, ha='center', va='top', transform=ax.transAxes,
                fontsize=8, color='gray')

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def create_comparison_plot(stats_j: Dict, stats_k: Dict, output_file: Path, logger: logging.Logger):
    """
    Create comparison plot for J(J vs R) and J(K vs R).

    Layout: 2 rows x 3 columns
    Row 1: J(J vs R) - ILSP, LSP, All
    Row 2: J(K vs R) - ILSP, LSP, All
    """
    logger.info("Creating comparison plot...")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 150,
    })

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    fig.suptitle('Judge Swap Analysis: GPT-3.5 (J) judging J vs R and K vs R\n' +
                 'J=GPT-3.5, K=Llama, R=Human',
                 fontsize=14, fontweight='bold', y=0.98)

    # Row 1: J(J vs R)
    plot_histogram_with_stats(
        ax=axes[0, 0],
        data=stats_j['probability_stats']['ilsp_mean'],
        objective=0.0,
        title="J(J vs R) - ILSP",
        xlabel="P(J)",
        color='red'
    )

    plot_histogram_with_stats(
        ax=axes[0, 1],
        data=stats_j['probability_stats']['lsp_mean'],
        objective=1.0,
        title="J(J vs R) - LSP",
        xlabel="P(J)",
        color='green'
    )

    plot_histogram_with_stats(
        ax=axes[0, 2],
        data=stats_j['probability_stats']['all_mean'],
        objective=0.5,
        title="J(J vs R) - All",
        xlabel="P(J)",
        color='cornflowerblue'
    )

    # Row 2: J(K vs R)
    plot_histogram_with_stats(
        ax=axes[1, 0],
        data=stats_k['probability_stats']['ilsp_mean'],
        objective=0.0,
        title="J(K vs R) - ILSP",
        xlabel="P(K)",
        color='red'
    )

    plot_histogram_with_stats(
        ax=axes[1, 1],
        data=stats_k['probability_stats']['lsp_mean'],
        objective=1.0,
        title="J(K vs R) - LSP",
        xlabel="P(K)",
        color='green'
    )

    plot_histogram_with_stats(
        ax=axes[1, 2],
        data=stats_k['probability_stats']['all_mean'],
        objective=0.5,
        title="J(K vs R) - All",
        xlabel="P(K)",
        color='cornflowerblue'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    logger.info(f"Saved plot to {output_file}")

    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', dpi=300)
    logger.info(f"Saved plot to {pdf_file}")

    plt.close()


def create_difference_plot(comparison: Dict, output_file: Path, logger: logging.Logger):
    """Create bar plot showing differences between J(J vs R) and J(K vs R)."""
    logger.info("Creating difference plot...")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['ILSP', 'LSP', 'All']
    diffs = [comparison['ilsp_diff'], comparison['lsp_diff'], comparison['all_diff']]
    colors = ['red' if d > 0 else 'blue' for d in diffs]

    bars = ax.bar(categories, diffs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, diff in zip(bars, diffs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{diff:+.4f}',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=10, fontweight='bold')

    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Difference: J(J vs R) - J(K vs R)', fontweight='bold', fontsize=11)
    ax.set_title('Self-Preference Bias Detection\n' +
                 'Positive = GPT-3.5 favors own responses over Llama\'s',
                 fontweight='bold', fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    logger.info(f"Saved difference plot to {output_file}")

    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', dpi=300)
    logger.info(f"Saved difference plot to {pdf_file}")

    plt.close()


def create_distribution_plot(probs_j: Dict, probs_k: Dict, output_file: Path, logger: logging.Logger):
    """
    Create distribution histograms comparing J(J vs R) and J(K vs R).

    Layout: 2 rows x 3 columns
    Row 1: J(J vs R) - ILSP, LSP, Combined
    Row 2: J(K vs R) - ILSP, LSP, Combined
    """
    if probs_j is None and probs_k is None:
        logger.warning("Raw probability data not available for either dataset, skipping distribution plot")
        return

    if probs_j is None:
        logger.warning("J(J vs R) raw probability data not available - will show only J(K vs R) distributions")
    if probs_k is None:
        logger.warning("J(K vs R) raw probability data not available - will show only J(J vs R) distributions")

    logger.info("Creating distribution plot...")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 150,
    })

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    fig.suptitle('Judge Swap Distribution Analysis\n' +
                 'J(J vs R) = Judge judging own responses vs Human, J(K vs R) = Judge judging proxy responses vs Human',
                 fontsize=12, fontweight='bold', y=0.98)

    # Row 1: J(J vs R)
    # ILSP
    if probs_j and len(probs_j['ilsp']) > 0:
        axes[0, 0].hist(probs_j['ilsp'], bins=15, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
        mean_j_ilsp = np.mean(probs_j['ilsp'])
        axes[0, 0].axvline(0.0, color='red', linestyle='--', linewidth=2, label='Objective', alpha=0.7)
        axes[0, 0].axvline(mean_j_ilsp, color='black', linestyle='--', linewidth=2, label=f'μ', alpha=0.7)
        axes[0, 0].set_title(f"J(J vs R) - ILSP\nμ: {mean_j_ilsp:.3f} | Obj: 0.000 | n={len(probs_j['ilsp'])}",
                            fontweight='bold', fontsize=10)
        axes[0, 0].set_xlabel("P(J)")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12, color='gray')
        axes[0, 0].set_title("J(J vs R) - ILSP\nNo data", fontweight='bold', fontsize=10)
        axes[0, 0].set_xlim(0, 1)

    # LSP
    if probs_j and len(probs_j['lsp']) > 0:
        axes[0, 1].hist(probs_j['lsp'], bins=15, color='lightgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
        mean_j_lsp = np.mean(probs_j['lsp'])
        axes[0, 1].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Objective', alpha=0.7)
        axes[0, 1].axvline(mean_j_lsp, color='black', linestyle='--', linewidth=2, label=f'μ', alpha=0.7)
        axes[0, 1].set_title(f"J(J vs R) - LSP\nμ: {mean_j_lsp:.3f} | Obj: 1.000 | n={len(probs_j['lsp'])}",
                            fontweight='bold', fontsize=10)
        axes[0, 1].set_xlabel("P(J)")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].grid(axis='y', linestyle='--', alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12, color='gray')
        axes[0, 1].set_title("J(J vs R) - LSP\nNo data", fontweight='bold', fontsize=10)
        axes[0, 1].set_xlim(0, 1)

    # Combined
    if probs_j and len(probs_j['all']) > 0:
        axes[0, 2].hist(probs_j['all'], bins=15, color='cornflowerblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        mean_j_all = np.mean(probs_j['all'])
        axes[0, 2].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Objective', alpha=0.7)
        axes[0, 2].axvline(mean_j_all, color='black', linestyle='--', linewidth=2, label=f'μ', alpha=0.7)
        axes[0, 2].set_title(f"J(J vs R) - Combined\nμ: {mean_j_all:.3f} | Obj: 0.500 | n={len(probs_j['all'])}",
                            fontweight='bold', fontsize=10)
        axes[0, 2].set_xlabel("P(J)")
        axes[0, 2].set_ylabel("Density")
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].grid(axis='y', linestyle='--', alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12, color='gray')
        axes[0, 2].set_title("J(J vs R) - Combined\nNo data", fontweight='bold', fontsize=10)
        axes[0, 2].set_xlim(0, 1)

    # Row 2: J(K vs R)
    # ILSP
    if probs_k and len(probs_k['ilsp']) > 0:
        axes[1, 0].hist(probs_k['ilsp'], bins=15, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
        mean_k_ilsp = np.mean(probs_k['ilsp'])
        axes[1, 0].axvline(0.0, color='red', linestyle='--', linewidth=2, label='Objective', alpha=0.7)
        axes[1, 0].axvline(mean_k_ilsp, color='black', linestyle='--', linewidth=2, label=f'μ', alpha=0.7)
        axes[1, 0].set_title(f"J(K vs R) - ILSP\nμ: {mean_k_ilsp:.3f} | Obj: 0.000 | n={len(probs_k['ilsp'])}",
                            fontweight='bold', fontsize=10)
        axes[1, 0].set_xlabel("P(K)")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12, color='gray')
        axes[1, 0].set_title("J(K vs R) - ILSP\nNo data", fontweight='bold', fontsize=10)
        axes[1, 0].set_xlim(0, 1)

    # LSP
    if probs_k and len(probs_k['lsp']) > 0:
        axes[1, 1].hist(probs_k['lsp'], bins=15, color='lightgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
        mean_k_lsp = np.mean(probs_k['lsp'])
        axes[1, 1].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Objective', alpha=0.7)
        axes[1, 1].axvline(mean_k_lsp, color='black', linestyle='--', linewidth=2, label=f'μ', alpha=0.7)
        axes[1, 1].set_title(f"J(K vs R) - LSP\nμ: {mean_k_lsp:.3f} | Obj: 1.000 | n={len(probs_k['lsp'])}",
                            fontweight='bold', fontsize=10)
        axes[1, 1].set_xlabel("P(K)")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].grid(axis='y', linestyle='--', alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12, color='gray')
        axes[1, 1].set_title("J(K vs R) - LSP\nNo data", fontweight='bold', fontsize=10)
        axes[1, 1].set_xlim(0, 1)

    # Combined
    if probs_k and len(probs_k['all']) > 0:
        axes[1, 2].hist(probs_k['all'], bins=15, color='cornflowerblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        mean_k_all = np.mean(probs_k['all'])
        axes[1, 2].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Objective', alpha=0.7)
        axes[1, 2].axvline(mean_k_all, color='black', linestyle='--', linewidth=2, label=f'μ', alpha=0.7)
        axes[1, 2].set_title(f"J(K vs R) - Combined\nμ: {mean_k_all:.3f} | Obj: 0.500 | n={len(probs_k['all'])}",
                            fontweight='bold', fontsize=10)
        axes[1, 2].set_xlabel("P(K)")
        axes[1, 2].set_ylabel("Density")
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].grid(axis='y', linestyle='--', alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12, color='gray')
        axes[1, 2].set_title("J(K vs R) - Combined\nNo data", fontweight='bold', fontsize=10)
        axes[1, 2].set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    logger.info(f"Saved distribution plot to {output_file}")

    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight', dpi=300)
    logger.info(f"Saved distribution plot to {pdf_file}")

    plt.close()


# ----------------------
# --- MAIN ENTRY     ---
# ----------------------

def main():
    """Main entry point for judge swap analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze judge swap for CNN dataset"
    )
    parser.add_argument("--j_vs_r_stats", type=str, required=True,
                       help="Path to J(J vs R) statistics JSON (gpt3.5 judging gpt3.5 vs human)")
    parser.add_argument("--k_vs_r_stats", type=str, required=True,
                       help="Path to J(K vs R) statistics JSON (gpt3.5 judging llama vs human)")
    parser.add_argument("--j_vs_r_eval", type=str, default=None,
                       help="Optional: Path to J(J vs R) evaluation JSON for t-test and KS test")
    parser.add_argument("--j_vs_r_gt", type=str, default=None,
                       help="Optional: Path to J(J vs R) ground truth JSON for t-test and KS test")
    parser.add_argument("--k_vs_r_eval", type=str, default=None,
                       help="Optional: Path to J(K vs R) evaluation JSON for t-test and KS test")
    parser.add_argument("--k_vs_r_gt", type=str, default=None,
                       help="Optional: Path to J(K vs R) ground truth JSON for t-test and KS test")
    parser.add_argument("--output_dir", type=str, default="judge_swap_analysis",
                       help="Output directory for plots and results")

    args = parser.parse_args()

    # Set up paths
    j_stats_path = Path(args.j_vs_r_stats)
    k_stats_path = Path(args.k_vs_r_stats)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(output_dir)

    # Load statistics
    logger.info(f"Loading J(J vs R) statistics from {j_stats_path}")
    if not j_stats_path.exists():
        logger.error(f"File not found: {j_stats_path}")
        sys.exit(1)

    with open(j_stats_path, 'r', encoding='utf-8') as f:
        stats_j = json.load(f)

    logger.info(f"Loading J(K vs R) statistics from {k_stats_path}")
    if not k_stats_path.exists():
        logger.error(f"File not found: {k_stats_path}")
        sys.exit(1)

    with open(k_stats_path, 'r', encoding='utf-8') as f:
        stats_k = json.load(f)

    logger.info("")

    # Load raw probability data if provided
    probs_j = None
    probs_k = None

    if args.j_vs_r_eval and args.j_vs_r_gt:
        logger.info("Loading raw probability data for J(J vs R)...")
        probs_j = load_raw_probabilities(
            Path(args.j_vs_r_eval),
            Path(args.j_vs_r_gt),
            logger
        )

    if args.k_vs_r_eval and args.k_vs_r_gt:
        logger.info("Loading raw probability data for J(K vs R)...")
        probs_k = load_raw_probabilities(
            Path(args.k_vs_r_eval),
            Path(args.k_vs_r_gt),
            logger
        )

    logger.info("")

    # Compute comparison statistics
    comparison = compute_hypothesis_tests(stats_j, stats_k, probs_j, probs_k, logger)

    # Create visualizations
    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    comparison_plot = output_dir / "judge_swap_comparison.png"
    create_comparison_plot(stats_j, stats_k, comparison_plot, logger)

    difference_plot = output_dir / "judge_swap_differences.png"
    create_difference_plot(comparison, difference_plot, logger)

    # Create distribution plot if raw data is available (for at least one dataset)
    if probs_j is not None or probs_k is not None:
        distribution_plot = output_dir / "judge_swap_distributions.png"
        create_distribution_plot(probs_j, probs_k, distribution_plot, logger)

    # Save comparison statistics
    comparison_file = output_dir / "judge_swap_statistics.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved comparison statistics to {comparison_file}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
