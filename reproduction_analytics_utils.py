#!/usr/bin/env python3
# reproduction_analytics_utils.py: Utility functions for analyzing reproduction quality
# Created: 2025-12-22 01:00 EST
# Last Modified: 2025-12-22 01:00 EST
"""
This module provides comprehensive analytics utilities for evaluating reproduction quality.

Key Metrics:
1. Per-game accuracy (Game 1 and Game 2 separately)
2. Transition matrix analysis (3x3 A/B/T mappings)
3. Probability distribution comparisons (KL divergence, JS divergence)
4. Per-verdict-type error rates
5. Correlation analysis
"""

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


# ----------------------
# --- UTILITY FUNCTIONS ---
# ----------------------

def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        p (np.ndarray): True distribution.
        q (np.ndarray): Approximate distribution.
        epsilon (float): Small value to avoid log(0).
    
    Returns:
        float: KL divergence D_KL(p || q).
    """
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    
    Args:
        p (np.ndarray): First distribution.
        q (np.ndarray): Second distribution.
    
    Returns:
        float: JS divergence (symmetric, bounded [0,1]).
    """
    return jensenshannon(p, q)


def get_verdict_from_probs(probs: Dict[str, float]) -> str:
    """Get most likely verdict from probability dict."""
    return max(probs.items(), key=lambda x: x[1])[0]


# ----------------------
# --- ANALYSIS FUNCTIONS ---
# ----------------------

def analyze_per_game_accuracy(results: List[Dict], logger: logging.Logger) -> Dict:
    """
    Analyze accuracy separately for Game 1 and Game 2.
    
    Args:
        results (List[Dict]): Reproduction results.
        logger (logging.Logger): Logger instance.
    
    Returns:
        Dict with per-game accuracy metrics.
    """
    game1_correct = 0
    game2_correct = 0
    total = len(results)
    
    for r in results:
        # Game 1
        ref_g1 = get_verdict_from_probs(r["game_1"]["reference_probs"])
        gen_g1 = get_verdict_from_probs(r["game_1"]["generated_probs"])
        if ref_g1 == gen_g1:
            game1_correct += 1
        
        # Game 2
        ref_g2 = get_verdict_from_probs(r["game_2"]["reference_probs"])
        gen_g2 = get_verdict_from_probs(r["game_2"]["generated_probs"])
        if ref_g2 == gen_g2:
            game2_correct += 1
    
    game1_acc = game1_correct / total if total > 0 else 0
    game2_acc = game2_correct / total if total > 0 else 0
    
    logger.info(f"\nPER-GAME ACCURACY:")
    logger.info(f"  Game 1: {game1_correct}/{total} = {game1_acc:.3f}")
    logger.info(f"  Game 2: {game2_correct}/{total} = {game2_acc:.3f}")
    logger.info(f"  Both:   {game1_correct + game2_correct}/{2*total} = {(game1_acc + game2_acc)/2:.3f}")
    
    return {
        "game1_accuracy": game1_acc,
        "game2_accuracy": game2_acc,
        "overall_accuracy": (game1_acc + game2_acc) / 2,
        "game1_correct": game1_correct,
        "game2_correct": game2_correct,
        "total": total,
    }


def analyze_transition_matrices(results: List[Dict], logger: logging.Logger) -> Dict:
    """
    Analyze 3x3 transition matrices for Game 1 and Game 2.
    
    Args:
        results (List[Dict]): Reproduction results.
        logger (logging.Logger): Logger instance.
    
    Returns:
        Dict with transition matrix data.
    """
    verdicts = ["A", "B", "T"]
    
    # Game 1 transition matrix
    g1_matrix = {v: {"A": 0, "B": 0, "T": 0} for v in verdicts}
    g1_totals = {"A": 0, "B": 0, "T": 0}
    
    # Game 2 transition matrix
    g2_matrix = {v: {"A": 0, "B": 0, "T": 0} for v in verdicts}
    g2_totals = {"A": 0, "B": 0, "T": 0}
    
    for r in results:
        # Game 1
        ref_g1 = get_verdict_from_probs(r["game_1"]["reference_probs"])
        gen_g1 = get_verdict_from_probs(r["game_1"]["generated_probs"])
        g1_matrix[ref_g1][gen_g1] += 1
        g1_totals[ref_g1] += 1
        
        # Game 2
        ref_g2 = get_verdict_from_probs(r["game_2"]["reference_probs"])
        gen_g2 = get_verdict_from_probs(r["game_2"]["generated_probs"])
        g2_matrix[ref_g2][gen_g2] += 1
        g2_totals[ref_g2] += 1
    
    # Print Game 1 matrix
    logger.info(f"\nGAME 1 TRANSITION MATRIX:")
    logger.info(f"         Gen: A      B      T")
    for ref_v in verdicts:
        counts = [g1_matrix[ref_v][gen_v] for gen_v in verdicts]
        total_ref = g1_totals[ref_v]
        probs = [c / total_ref if total_ref > 0 else 0 for c in counts]
        logger.info(f"  Ref {ref_v}: {counts[0]:3d}    {counts[1]:3d}    {counts[2]:3d}   "
                   f"({probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f})")
    
    # Print Game 2 matrix
    logger.info(f"\nGAME 2 TRANSITION MATRIX:")
    logger.info(f"         Gen: A      B      T")
    for ref_v in verdicts:
        counts = [g2_matrix[ref_v][gen_v] for gen_v in verdicts]
        total_ref = g2_totals[ref_v]
        probs = [c / total_ref if total_ref > 0 else 0 for c in counts]
        logger.info(f"  Ref {ref_v}: {counts[0]:3d}    {counts[1]:3d}    {counts[2]:3d}   "
                   f"({probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f})")
    
    return {
        "game1_matrix": g1_matrix,
        "game1_totals": g1_totals,
        "game2_matrix": g2_matrix,
        "game2_totals": g2_totals,
    }


def analyze_per_verdict_errors(results: List[Dict], logger: logging.Logger) -> Dict:
    """
    Analyze error rates broken down by reference verdict type.
    
    Args:
        results (List[Dict]): Reproduction results.
        logger (logging.Logger): Logger instance.
    
    Returns:
        Dict with per-verdict error metrics.
    """
    verdict_errors = {"A": [], "B": [], "T": []}
    verdict_counts = {"A": 0, "B": 0, "T": 0}
    
    for r in results:
        # Average across both games
        ref_g1 = get_verdict_from_probs(r["game_1"]["reference_probs"])
        gen_g1 = get_verdict_from_probs(r["game_1"]["generated_probs"])
        ref_g2 = get_verdict_from_probs(r["game_2"]["reference_probs"])
        gen_g2 = get_verdict_from_probs(r["game_2"]["generated_probs"])
        
        # Track errors for each verdict type
        for ref_v, gen_v in [(ref_g1, gen_g1), (ref_g2, gen_g2)]:
            verdict_counts[ref_v] += 1
            is_error = 1 if ref_v != gen_v else 0
            verdict_errors[ref_v].append(is_error)
    
    logger.info(f"\nPER-VERDICT ERROR RATES:")
    error_stats = {}
    for v in ["A", "B", "T"]:
        if verdict_counts[v] > 0:
            error_rate = sum(verdict_errors[v]) / len(verdict_errors[v])
            logger.info(f"  Reference {v}: {sum(verdict_errors[v])}/{verdict_counts[v]} = {error_rate:.3f}")
            error_stats[v] = {
                "error_rate": error_rate,
                "errors": sum(verdict_errors[v]),
                "total": verdict_counts[v],
            }
    
    return error_stats


def analyze_probability_distributions(results: List[Dict], logger: logging.Logger) -> Dict:
    """
    Analyze probability distribution similarity using KL and JS divergence.
    
    Args:
        results (List[Dict]): Reproduction results.
        logger (logging.Logger): Logger instance.
    
    Returns:
        Dict with divergence metrics.
    """
    game1_kl_divs = []
    game1_js_divs = []
    game2_kl_divs = []
    game2_js_divs = []
    
    for r in results:
        # Game 1
        ref_g1 = np.array([r["game_1"]["reference_probs"]["A"], 
                          r["game_1"]["reference_probs"]["B"], 
                          r["game_1"]["reference_probs"]["T"]])
        gen_g1 = np.array([r["game_1"]["generated_probs"]["A"], 
                          r["game_1"]["generated_probs"]["B"], 
                          r["game_1"]["generated_probs"]["T"]])
        
        game1_kl_divs.append(compute_kl_divergence(ref_g1, gen_g1))
        game1_js_divs.append(compute_js_divergence(ref_g1, gen_g1))
        
        # Game 2
        ref_g2 = np.array([r["game_2"]["reference_probs"]["A"], 
                          r["game_2"]["reference_probs"]["B"], 
                          r["game_2"]["reference_probs"]["T"]])
        gen_g2 = np.array([r["game_2"]["generated_probs"]["A"], 
                          r["game_2"]["generated_probs"]["B"], 
                          r["game_2"]["generated_probs"]["T"]])
        
        game2_kl_divs.append(compute_kl_divergence(ref_g2, gen_g2))
        game2_js_divs.append(compute_js_divergence(ref_g2, gen_g2))
    
    logger.info(f"\nPROBABILITY DISTRIBUTION SIMILARITY:")
    logger.info(f"  Game 1 KL Divergence: {np.mean(game1_kl_divs):.4f} ± {np.std(game1_kl_divs):.4f}")
    logger.info(f"  Game 1 JS Divergence: {np.mean(game1_js_divs):.4f} ± {np.std(game1_js_divs):.4f}")
    logger.info(f"  Game 2 KL Divergence: {np.mean(game2_kl_divs):.4f} ± {np.std(game2_kl_divs):.4f}")
    logger.info(f"  Game 2 JS Divergence: {np.mean(game2_js_divs):.4f} ± {np.std(game2_js_divs):.4f}")
    logger.info(f"  Overall KL: {np.mean(game1_kl_divs + game2_kl_divs):.4f}")
    logger.info(f"  Overall JS: {np.mean(game1_js_divs + game2_js_divs):.4f}")
    
    return {
        "game1_kl_mean": float(np.mean(game1_kl_divs)),
        "game1_kl_std": float(np.std(game1_kl_divs)),
        "game1_js_mean": float(np.mean(game1_js_divs)),
        "game1_js_std": float(np.std(game1_js_divs)),
        "game2_kl_mean": float(np.mean(game2_kl_divs)),
        "game2_kl_std": float(np.std(game2_kl_divs)),
        "game2_js_mean": float(np.mean(game2_js_divs)),
        "game2_js_std": float(np.std(game2_js_divs)),
    }


def analyze_probability_correlations(results: List[Dict], logger: logging.Logger) -> Dict:
    """
    Analyze correlations between reference and generated probabilities.
    
    Args:
        results (List[Dict]): Reproduction results.
        logger (logging.Logger): Logger instance.
    
    Returns:
        Dict with correlation metrics.
    """
    # Collect probabilities
    ref_probs_a_g1, gen_probs_a_g1 = [], []
    ref_probs_b_g1, gen_probs_b_g1 = [], []
    ref_probs_t_g1, gen_probs_t_g1 = [], []
    
    ref_probs_a_g2, gen_probs_a_g2 = [], []
    ref_probs_b_g2, gen_probs_b_g2 = [], []
    ref_probs_t_g2, gen_probs_t_g2 = [], []
    
    for r in results:
        # Game 1
        ref_probs_a_g1.append(r["game_1"]["reference_probs"]["A"])
        gen_probs_a_g1.append(r["game_1"]["generated_probs"]["A"])
        ref_probs_b_g1.append(r["game_1"]["reference_probs"]["B"])
        gen_probs_b_g1.append(r["game_1"]["generated_probs"]["B"])
        ref_probs_t_g1.append(r["game_1"]["reference_probs"]["T"])
        gen_probs_t_g1.append(r["game_1"]["generated_probs"]["T"])
        
        # Game 2
        ref_probs_a_g2.append(r["game_2"]["reference_probs"]["A"])
        gen_probs_a_g2.append(r["game_2"]["generated_probs"]["A"])
        ref_probs_b_g2.append(r["game_2"]["reference_probs"]["B"])
        gen_probs_b_g2.append(r["game_2"]["generated_probs"]["B"])
        ref_probs_t_g2.append(r["game_2"]["reference_probs"]["T"])
        gen_probs_t_g2.append(r["game_2"]["generated_probs"]["T"])
    
    # Compute correlations
    corr_a_g1, _ = stats.pearsonr(ref_probs_a_g1, gen_probs_a_g1)
    corr_b_g1, _ = stats.pearsonr(ref_probs_b_g1, gen_probs_b_g1)
    corr_t_g1, _ = stats.pearsonr(ref_probs_t_g1, gen_probs_t_g1)
    
    corr_a_g2, _ = stats.pearsonr(ref_probs_a_g2, gen_probs_a_g2)
    corr_b_g2, _ = stats.pearsonr(ref_probs_b_g2, gen_probs_b_g2)
    corr_t_g2, _ = stats.pearsonr(ref_probs_t_g2, gen_probs_t_g2)
    
    logger.info(f"\nPROBABILITY CORRELATIONS:")
    logger.info(f"  Game 1 - A: r={corr_a_g1:.3f}, B: r={corr_b_g1:.3f}, T: r={corr_t_g1:.3f}")
    logger.info(f"  Game 2 - A: r={corr_a_g2:.3f}, B: r={corr_b_g2:.3f}, T: r={corr_t_g2:.3f}")
    logger.info(f"  Average: r={np.mean([corr_a_g1, corr_b_g1, corr_t_g1, corr_a_g2, corr_b_g2, corr_t_g2]):.3f}")
    
    return {
        "game1_corr_a": corr_a_g1,
        "game1_corr_b": corr_b_g1,
        "game1_corr_t": corr_t_g1,
        "game2_corr_a": corr_a_g2,
        "game2_corr_b": corr_b_g2,
        "game2_corr_t": corr_t_g2,
    }


def analyze_systematic_biases(results: List[Dict], logger: logging.Logger) -> Dict:
    """
    Detect systematic biases in generated verdicts.
    
    Args:
        results (List[Dict]): Reproduction results.
        logger (logging.Logger): Logger instance.
    
    Returns:
        Dict with bias metrics.
    """
    # Count verdict distributions
    ref_g1_dist = {"A": 0, "B": 0, "T": 0}
    gen_g1_dist = {"A": 0, "B": 0, "T": 0}
    ref_g2_dist = {"A": 0, "B": 0, "T": 0}
    gen_g2_dist = {"A": 0, "B": 0, "T": 0}
    
    for r in results:
        ref_g1_dist[get_verdict_from_probs(r["game_1"]["reference_probs"])] += 1
        gen_g1_dist[get_verdict_from_probs(r["game_1"]["generated_probs"])] += 1
        ref_g2_dist[get_verdict_from_probs(r["game_2"]["reference_probs"])] += 1
        gen_g2_dist[get_verdict_from_probs(r["game_2"]["generated_probs"])] += 1
    
    total = len(results)
    
    logger.info(f"\nVERDICT DISTRIBUTION COMPARISON:")
    logger.info(f"  Game 1:")
    logger.info(f"    Reference: A={ref_g1_dist['A']/total:.3f}, B={ref_g1_dist['B']/total:.3f}, T={ref_g1_dist['T']/total:.3f}")
    logger.info(f"    Generated: A={gen_g1_dist['A']/total:.3f}, B={gen_g1_dist['B']/total:.3f}, T={gen_g1_dist['T']/total:.3f}")
    logger.info(f"  Game 2:")
    logger.info(f"    Reference: A={ref_g2_dist['A']/total:.3f}, B={ref_g2_dist['B']/total:.3f}, T={ref_g2_dist['T']/total:.3f}")
    logger.info(f"    Generated: A={gen_g2_dist['A']/total:.3f}, B={gen_g2_dist['B']/total:.3f}, T={gen_g2_dist['T']/total:.3f}")
    
    # Chi-square test for distribution difference
    ref_g1_arr = np.array([ref_g1_dist["A"], ref_g1_dist["B"], ref_g1_dist["T"]])
    gen_g1_arr = np.array([gen_g1_dist["A"], gen_g1_dist["B"], gen_g1_dist["T"]])
    chi2_g1, p_g1 = stats.chisquare(gen_g1_arr, ref_g1_arr)
    
    ref_g2_arr = np.array([ref_g2_dist["A"], ref_g2_dist["B"], ref_g2_dist["T"]])
    gen_g2_arr = np.array([gen_g2_dist["A"], gen_g2_dist["B"], gen_g2_dist["T"]])
    chi2_g2, p_g2 = stats.chisquare(gen_g2_arr, ref_g2_arr)
    
    logger.info(f"\nCHI-SQUARE TESTS:")
    logger.info(f"  Game 1: χ²={chi2_g1:.4f}, p={p_g1:.4f}")
    logger.info(f"  Game 2: χ²={chi2_g2:.4f}, p={p_g2:.4f}")
    
    return {
        "ref_g1_dist": {k: v/total for k, v in ref_g1_dist.items()},
        "gen_g1_dist": {k: v/total for k, v in gen_g1_dist.items()},
        "ref_g2_dist": {k: v/total for k, v in ref_g2_dist.items()},
        "gen_g2_dist": {k: v/total for k, v in gen_g2_dist.items()},
        "chi2_g1": chi2_g1,
        "p_g1": p_g1,
        "chi2_g2": chi2_g2,
        "p_g2": p_g2,
    }


def generate_comprehensive_report(results: List[Dict], output_file: Path, logger: logging.Logger) -> Dict:
    """
    Generate comprehensive analytics report.
    
    Args:
        results (List[Dict]): Reproduction results.
        output_file (Path): Path to save JSON report.
        logger (logging.Logger): Logger instance.
    
    Returns:
        Dict with all analytics.
    """
    logger.info("=" * 80)
    logger.info(f"COMPREHENSIVE REPRODUCTION QUALITY ANALYSIS")
    logger.info(f"Total examples: {len(results)}")
    logger.info("=" * 80)
    
    # Run all analyses
    per_game_acc = analyze_per_game_accuracy(results, logger)
    transition_matrices = analyze_transition_matrices(results, logger)
    per_verdict_errors = analyze_per_verdict_errors(results, logger)
    prob_distributions = analyze_probability_distributions(results, logger)
    prob_correlations = analyze_probability_correlations(results, logger)
    systematic_biases = analyze_systematic_biases(results, logger)
    
    # Compile report
    report = {
        "total_examples": len(results),
        "per_game_accuracy": per_game_acc,
        "transition_matrices": {
            k: v for k, v in transition_matrices.items() 
            if not k.endswith("_totals")
        },
        "per_verdict_errors": per_verdict_errors,
        "probability_distributions": prob_distributions,
        "probability_correlations": prob_correlations,
        "systematic_biases": systematic_biases,
    }
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Report saved to: {output_file}")
    logger.info("=" * 80)
    
    return report
