
"""
This script runs reproduction experiments on small baseline models and generates
comprehensive analytics to debug reproduction quality issues.

Strategy:
1. Run 100-sample baseline for smallest model from each family
2. Generate detailed text analytics (no plots needed)
3. Compare results across families to identify patterns
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from reproduction_analytics_utils import generate_comprehensive_report, load_jsonl


# ----------------------
# --- CONFIGURATION ---
# ----------------------

# Smallest model from each family for quick testing
TEST_MODELS = {
    "llama": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "short": "llama-3.2-3b",
        "family": "llama",
    },
    "qwen": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "short": "qwen-2.5-3b",
        "family": "qwen",
    },
    "gemma": {
        "model": "google/gemma-2-2b-it",
        "short": "gemma-2-2b",
        "family": "gemma",
    },
}

BENCHMARK = "math500"
N_SAMPLES = 100


# ----------------------
# --- LOGGING SETUP ---
# ----------------------

def setup_logging() -> logging.Logger:
    """
    Configure logging to both file and console.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = Path("file_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"debug_reproduction_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logger = logging.getLogger("debug_reproduction")
    return logger


# ----------------------
# --- MAIN FUNCTIONS ---
# ----------------------

def run_reproduction_experiment(model_config: dict, benchmark: str, n_samples: int, 
                                logger: logging.Logger) -> Path:
    """
    Run reproduction experiment for a single model.
    
    Args:
        model_config (dict): Model configuration with 'model', 'short', 'family' keys.
        benchmark (str): Benchmark name.
        n_samples (int): Number of samples to process.
        logger (logging.Logger): Logger instance.
    
    Returns:
        Path: Path to output file.
    """
    model = model_config["model"]
    model_short = model_config["short"]
    family = model_config["family"]
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"RUNNING REPRODUCTION: {model_short}")
    logger.info(f"{'=' * 80}")
    
    # Build command
    # We need to find a matching evaluatee from llm-sp data
    llm_sp_dir = Path("llm-sp/sp")
    data_dir = llm_sp_dir / benchmark / family
    
    # Find any eval file for this judge
    eval_files = list(data_dir.glob(f"{model_short}_*_eval.jsonl"))
    
    if not eval_files:
        logger.error(f"No eval files found for {model_short} in {data_dir}")
        return None
    
    # Use the first one
    data_file = eval_files[0]
    evaluatee_short = data_file.stem.replace("_eval", "").split('_', 1)[1]
    
    logger.info(f"Using data file: {data_file}")
    logger.info(f"Evaluatee: {evaluatee_short}")
    
    cmd = [
        "python3", "reproduce_paper_experiments.py",
        "--model", model,
        "--judge_short", model_short,
        "--judge_family", family,
        "--evaluatee_short", evaluatee_short,
        "--benchmark", benchmark,
        "--data", str(data_file),
        "--n_samples", str(n_samples),
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run experiment
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        logger.error(f"ERROR: Reproduction failed for {model_short}")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        return None
    
    logger.info(f"✓ Completed in {elapsed:.1f}s")
    
    # Find output file
    output_dir = Path("llm-sp-reprod") / benchmark / family
    output_files = list(output_dir.glob(f"{model_short}_*_reprod.jsonl"))
    
    if not output_files:
        logger.error(f"ERROR: Could not find output file for {model_short}")
        return None
    
    # Use most recent file
    output_file = max(output_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Output file: {output_file}")
    
    return output_file


def analyze_results(output_file: Path, model_short: str, logger: logging.Logger) -> dict:
    """
    Analyze reproduction results and generate comprehensive report.
    
    Args:
        output_file (Path): Path to reproduction results file.
        model_short (str): Model short name.
        logger (logging.Logger): Logger instance.
    
    Returns:
        dict: Analytics report.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"ANALYZING RESULTS: {model_short}")
    logger.info(f"{'=' * 80}")
    
    # Load results
    results = load_jsonl(output_file)
    logger.info(f"Loaded {len(results)} examples")
    
    # Generate report
    report_file = output_file.parent / f"{output_file.stem}_debug_report.json"
    report = generate_comprehensive_report(results, report_file, logger)
    
    return report


def compare_across_families(reports: dict, logger: logging.Logger):
    """
    Compare reproduction quality across model families.
    
    Args:
        reports (dict): Dict mapping model_short to analytics report.
        logger (logging.Logger): Logger instance.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"CROSS-FAMILY COMPARISON")
    logger.info(f"{'=' * 80}")
    
    # Overall accuracy comparison
    logger.info(f"\nOVERALL ACCURACY:")
    for model_short, report in reports.items():
        acc = report["per_game_accuracy"]["overall_accuracy"]
        g1_acc = report["per_game_accuracy"]["game1_accuracy"]
        g2_acc = report["per_game_accuracy"]["game2_accuracy"]
        logger.info(f"  {model_short:20s}: Overall={acc:.3f}, Game1={g1_acc:.3f}, Game2={g2_acc:.3f}")
    
    # Probability divergence comparison
    logger.info(f"\nPROBABILITY DIVERGENCE (JS):")
    for model_short, report in reports.items():
        js_g1 = report["probability_distributions"]["game1_js_mean"]
        js_g2 = report["probability_distributions"]["game2_js_mean"]
        logger.info(f"  {model_short:20s}: Game1={js_g1:.4f}, Game2={js_g2:.4f}")
    
    # Correlation comparison
    logger.info(f"\nPROBABILITY CORRELATIONS (average):")
    for model_short, report in reports.items():
        corrs = [
            report["probability_correlations"]["game1_corr_a"],
            report["probability_correlations"]["game1_corr_b"],
            report["probability_correlations"]["game1_corr_t"],
            report["probability_correlations"]["game2_corr_a"],
            report["probability_correlations"]["game2_corr_b"],
            report["probability_correlations"]["game2_corr_t"],
        ]
        avg_corr = sum(corrs) / len(corrs)
        logger.info(f"  {model_short:20s}: r={avg_corr:.3f}")
    
    # Systematic bias comparison
    logger.info(f"\nSYSTEMATIC BIASES (Chi-square p-values):")
    for model_short, report in reports.items():
        p_g1 = report["systematic_biases"]["p_g1"]
        p_g2 = report["systematic_biases"]["p_g2"]
        logger.info(f"  {model_short:20s}: Game1 p={p_g1:.4f}, Game2 p={p_g2:.4f}")
    
    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"SUMMARY & RECOMMENDATIONS")
    logger.info(f"{'=' * 80}")
    
    # Find best and worst performers
    accs = {model_short: report["per_game_accuracy"]["overall_accuracy"] 
            for model_short, report in reports.items()}
    best_model = max(accs.items(), key=lambda x: x[1])
    worst_model = min(accs.items(), key=lambda x: x[1])
    
    logger.info(f"\nBest performer: {best_model[0]} ({best_model[1]:.3f} accuracy)")
    logger.info(f"Worst performer: {worst_model[0]} ({worst_model[1]:.3f} accuracy)")
    
    # Check for common issues
    issues = []
    for model_short, report in reports.items():
        g1_acc = report["per_game_accuracy"]["game1_accuracy"]
        g2_acc = report["per_game_accuracy"]["game2_accuracy"]
        
        if abs(g1_acc - g2_acc) > 0.15:
            issues.append(f"  - {model_short}: Large Game 1/2 accuracy gap ({g1_acc:.3f} vs {g2_acc:.3f})")
        
        # Check for systematic bias
        p_g1 = report["systematic_biases"]["p_g1"]
        p_g2 = report["systematic_biases"]["p_g2"]
        if p_g1 < 0.01 or p_g2 < 0.01:
            issues.append(f"  - {model_short}: Significant distribution shift (p < 0.01)")
        
        # Check for low correlation
        avg_corr = sum([
            report["probability_correlations"]["game1_corr_a"],
            report["probability_correlations"]["game1_corr_b"],
            report["probability_correlations"]["game1_corr_t"],
            report["probability_correlations"]["game2_corr_a"],
            report["probability_correlations"]["game2_corr_b"],
            report["probability_correlations"]["game2_corr_t"],
        ]) / 6
        if avg_corr < 0.5:
            issues.append(f"  - {model_short}: Low probability correlation (r={avg_corr:.3f})")
    
    if issues:
        logger.info(f"\nIDENTIFIED ISSUES:")
        for issue in issues:
            logger.info(issue)
    else:
        logger.info(f"\nNo major issues detected across families.")
    
    logger.info(f"\n{'=' * 80}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Debug reproduction quality with automated analytics")
    parser.add_argument("--families", nargs="+", default=["llama", "qwen", "gemma"],
                       help="Model families to test")
    parser.add_argument("--benchmark", type=str, default=BENCHMARK, help="Benchmark to use")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES, help="Number of samples")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("REPRODUCTION QUALITY DEBUG SESSION")
    logger.info("=" * 80)
    logger.info(f"Families: {', '.join(args.families)}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Run experiments for each family
    results_files = {}
    for family in args.families:
        if family not in TEST_MODELS:
            logger.warning(f"Unknown family: {family}, skipping")
            continue
        
        model_config = TEST_MODELS[family]
        output_file = run_reproduction_experiment(
            model_config, args.benchmark, args.n_samples, logger
        )
        
        if output_file:
            results_files[model_config["short"]] = output_file
    
    if not results_files:
        logger.error("No successful experiments, exiting")
        sys.exit(1)
    
    # Analyze each result
    reports = {}
    for model_short, output_file in results_files.items():
        report = analyze_results(output_file, model_short, logger)
        reports[model_short] = report
    
    # Cross-family comparison
    compare_across_families(reports, logger)
    
    logger.info(f"\n{'=' * 80}")
    logger.info("DEBUG SESSION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
