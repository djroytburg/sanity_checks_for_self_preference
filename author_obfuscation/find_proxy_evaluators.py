# find_proxy_evaluators.py: Find proxy evaluators that co-occur in oracle-determined files
# For evaluator J vs evaluatee R, find evaluators K that appear in same ben/harmful files
# Supports auto-detection of preference files (excluding _2w_ files) and batch processing of all J/R pairs
# Created: Jan 4, 2026, 16:45 EST
# Last Modified: Jan 4, 2026, 17:00 EST

import json
import os
import sys
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# -------------------------
# --- LOGGING SETUP ---
# -------------------------

def setup_logging():
    """Set up logging with file output."""
    log_dir = Path("../file_logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"find_proxy_evaluators_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# -------------------------
# --- DATA LOADING ---
# -------------------------

def find_preference_files(base_dir="data/quality/preference_results"):
    """
    Automatically find ben and harmful preference files, excluding _2w_ files.
    
    Args:
        base_dir (str): Base directory to search for files
    
    Returns:
        tuple: (ben_path, harmful_path) or (None, None) if not found
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return None, None
    
    ben_files = list(base_path.glob("*ben*.json"))
    harmful_files = list(base_path.glob("*harmful*.json"))
    
    # Filter out files with _2w_
    ben_files = [f for f in ben_files if "_2w_" not in f.name]
    harmful_files = [f for f in harmful_files if "_2w_" not in f.name]
    
    if not ben_files or not harmful_files:
        return None, None
    
    return str(ben_files[0]), str(harmful_files[0])

def find_all_reference_pairs(oracle_data):
    """
    Find all (evaluator, evaluatee) pairs that have evaluations.
    
    Args:
        oracle_data (dict): Oracle file memberships
    
    Returns:
        set: Set of (evaluator, evaluatee) tuples
    """
    return {(evaluator, evaluatee) for (evaluator, evaluatee, pid), _ in oracle_data.items()}

def load_oracle_files(ben_path, harmful_path):
    """
    Load oracle-determined file memberships.
    
    Args:
        ben_path (str): Path to ben JSON file (J correct, R wrong)
        harmful_path (str): Path to harmful JSON file (J wrong, R correct)
    
    Returns:
        dict: {(evaluator, evaluatee, pid): file_type} where file_type is 'ben' or 'harmful'
    """
    oracle_data = {}
    
    # Load ben data (oracle: evaluator correct, evaluatee wrong)
    with open(ben_path) as f:
        ben_data = json.load(f)
        for item in ben_data:
            key = (item['evaluator'], item['evaluatee'], item['pid'])
            oracle_data[key] = 'ben'
    
    # Load harmful data (oracle: evaluator wrong, evaluatee correct)  
    with open(harmful_path) as f:
        harmful_data = json.load(f)
        for item in harmful_data:
            key = (item['evaluator'], item['evaluatee'], item['pid'])
            oracle_data[key] = 'harmful'
    
    return oracle_data

# -------------------------
# --- PROXY FINDING ---
# -------------------------

def find_proxy_evaluators(reference_evaluator, reference_evaluatee, oracle_data, logger, min_overlap=5, min_lsp=5, min_ilsp=5):
    """
    Find proxy evaluators that co-occur in same oracle files as reference evaluator.
    
    Args:
        reference_evaluator (str): Reference evaluator J
        reference_evaluatee (str): Reference evaluatee R  
        oracle_data (dict): Oracle file memberships
        logger (logging.Logger): Logger instance
        min_overlap (int): Minimum common examples required
        min_lsp (int): Minimum LSP examples required
        min_ilsp (int): Minimum ILSP examples required
    
    Returns:
        list: List of proxy evaluator results, sorted by ILSP count
    """
    logger.info(f"Finding proxies for {reference_evaluator} vs {reference_evaluatee}")
    
    # Find all pids where reference evaluator evaluated vs reference evaluatee
    reference_evaluations = {(evaluator, evaluatee, pid): file_type 
                           for (evaluator, evaluatee, pid), file_type in oracle_data.items()
                           if evaluator == reference_evaluator and evaluatee == reference_evaluatee}
    
    if not reference_evaluations:
        logger.warning(f"No evaluations found for {reference_evaluator} vs {reference_evaluatee}")
        return []
    
    reference_pids = set(pid for (_, _, pid), _ in reference_evaluations.items())
    logger.info(f"Reference evaluator has {len(reference_pids)} evaluations")
    
    # Count reference file distribution
    ref_ben_count = sum(1 for ft in reference_evaluations.values() if ft == 'ben')
    ref_harmful_count = sum(1 for ft in reference_evaluations.values() if ft == 'harmful')
    ref_winrate = ref_ben_count / len(reference_evaluations) if reference_evaluations else 0
    
    # Find all other evaluators that evaluated vs reference_evaluatee
    all_other_evaluators = {evaluator for (evaluator, evaluatee, pid), _ in oracle_data.items()
                           if evaluatee == reference_evaluatee and evaluator != reference_evaluator}
    
    logger.info(f"Found {len(all_other_evaluators)} potential proxy evaluators")
    
    results = []
    for proxy_evaluator in all_other_evaluators:
        # Find pids where proxy also evaluated vs reference_evaluatee
        proxy_evaluations = {(evaluator, evaluatee, pid): file_type
                           for (evaluator, evaluatee, pid), file_type in oracle_data.items()
                           if evaluator == proxy_evaluator and evaluatee == reference_evaluatee}
        
        proxy_pids = set(pid for (_, _, pid), _ in proxy_evaluations.items())
        common_pids = reference_pids & proxy_pids
        
        if len(common_pids) < min_overlap:  # Require minimum overlap
            continue
        
        # Count co-occurrences in same files
        n_lsp = 0  # Both in 'ben' files (both evaluated R when R is actually wrong)
        n_ilsp = 0  # Both in 'harmful' files (both evaluated R when R is actually correct)
        lsp_ids = []
        ilsp_ids = []
        
        for pid in common_pids:
            ref_file = reference_evaluations[(reference_evaluator, reference_evaluatee, pid)]
            proxy_file = proxy_evaluations[(proxy_evaluator, reference_evaluatee, pid)]
            
            if ref_file == proxy_file:
                if ref_file == 'ben':
                    n_lsp += 1
                    lsp_ids.append(pid)
                else:  # harmful
                    n_ilsp += 1
                    ilsp_ids.append(pid)
        
        # Calculate proxy winrate (fraction of evaluations in 'ben' files)
        proxy_ben_count = sum(1 for ft in proxy_evaluations.values() if ft == 'ben')
        proxy_winrate = proxy_ben_count / len(proxy_evaluations) if proxy_evaluations else 0
        
        # Only include if sufficient examples in both categories
        if n_lsp >= min_lsp and n_ilsp >= min_ilsp:
            result = {
                'proxy': proxy_evaluator,
                'reference_evaluator': reference_evaluator,
                'reference_evaluatee': reference_evaluatee,
                'n_common': len(common_pids),
                'n_lsp': n_lsp,
                'n_ilsp': n_ilsp,
                'lsp_rate': n_lsp / len(common_pids),
                'ilsp_rate': n_ilsp / len(common_pids),
                'reference_winrate': ref_winrate,
                'proxy_winrate': proxy_winrate,
                'lsp_ids': lsp_ids,
                'ilsp_ids': ilsp_ids
            }
            results.append(result)
    
    # Sort by ILSP count (most examples where both evaluated correct R)
    results.sort(key=lambda x: x['n_ilsp'], reverse=True)
    
    logger.info(f"Found {len(results)} suitable proxy evaluators")
    return results

# -------------------------
# --- OUTPUT FORMATTING ---
# -------------------------

def print_proxy_summary(results, reference_evaluator, reference_evaluatee, logger):
    """
    Print formatted summary of proxy evaluators.
    
    Args:
        results (list): Proxy evaluator results
        reference_evaluator (str): Reference evaluator J
        reference_evaluatee (str): Reference evaluatee R
        logger (logging.Logger): Logger instance
    """
    if not results:
        logger.info("No suitable proxy evaluators found")
        return
    
    logger.info("=" * 100)
    logger.info(f"PROXY EVALUATORS for Reference Evaluator={reference_evaluator}, Reference Evaluatee={reference_evaluatee}")
    logger.info("=" * 100)
    logger.info(f"{'Proxy Evaluator':<45} {'Common':>7} {'LSP':>5} {'ILSP':>5} {'LSP%':>5} {'ILSP%':>6} {'Ref.WR':>7} {'K.WR':>6}")
    logger.info("-" * 100)
    
    for r in results[:30]:  # Show top 30
        logger.info(f"{r['proxy']:<45} {r['n_common']:>7} {r['n_lsp']:>5} {r['n_ilsp']:>5} {r['lsp_rate']:>5.1%} {r['ilsp_rate']:>6.1%} {r['reference_winrate']:>7.1%} {r['proxy_winrate']:>6.1%}")
    logger.info("-" * 100)
    logger.info(f"Showing top {min(30, len(results))} of {len(results)} proxy evaluators")

# -------------------------
# --- MAIN EXECUTION ---
# -------------------------

def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(description="Find proxy evaluators that co-occur in oracle-determined files")
    parser.add_argument("--ben_path", help="Path to ben JSON file (oracle: evaluator correct, evaluatee wrong) - auto-detected if not provided")
    parser.add_argument("--harmful_path", help="Path to harmful JSON file (oracle: evaluator wrong, evaluatee correct) - auto-detected if not provided")
    parser.add_argument("reference_evaluator", nargs='?', help="Reference evaluator J (optional - if not provided, process all pairs)")
    parser.add_argument("reference_evaluatee", nargs='?', help="Reference evaluatee R (optional - if not provided, process all pairs)")
    parser.add_argument("--output_dir", default="data/quality/proxies", 
                       help="Output directory for results (default: data/quality/proxies)")
    parser.add_argument("--min_overlap", type=int, default=5, 
                       help="Minimum number of common examples required (default: 5)")
    parser.add_argument("--min_lsp", type=int, default=5, 
                       help="Minimum LSP examples required (default: 5)")
    parser.add_argument("--min_ilsp", type=int, default=5, 
                       help="Minimum ILSP examples required (default: 5)")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting proxy evaluator search")
    logger.info(f"CUDA available: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"User: {os.environ.get('USER', 'Unknown')}")
    
    # Find preference files if not provided
    if not args.ben_path or not args.harmful_path:
        logger.info("Auto-detecting preference files...")
        auto_ben, auto_harmful = find_preference_files()
        if not auto_ben or not auto_harmful:
            logger.error("Could not auto-detect preference files")
            return
        args.ben_path = auto_ben
        args.harmful_path = auto_harmful
        logger.info(f"Using ben file: {args.ben_path}")
        logger.info(f"Using harmful file: {args.harmful_path}")
    
    # Load oracle data
    logger.info("Loading oracle file memberships...")
    oracle_data = load_oracle_files(args.ben_path, args.harmful_path)
    logger.info(f"Loaded {len(oracle_data)} oracle file memberships")
    
    # Determine which pairs to process
    if args.reference_evaluator and args.reference_evaluatee:
        # Process single pair
        reference_pairs = [(args.reference_evaluator, args.reference_evaluatee)]
        logger.info(f"Processing single pair: {args.reference_evaluator} vs {args.reference_evaluatee}")
    else:
        # Process all pairs
        reference_pairs = find_all_reference_pairs(oracle_data)
        logger.info(f"Processing all {len(reference_pairs)} reference pairs")
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_saved = 0
    failed_pairs = []
    
    # Process each reference pair
    for ref_evaluator, ref_evaluatee in reference_pairs:
        logger.info(f"Processing pair: {ref_evaluator} vs {ref_evaluatee}")
        
        try:
            # Find proxy evaluators
            results = find_proxy_evaluators(ref_evaluator, ref_evaluatee, oracle_data, logger, 
                                           min_overlap=args.min_overlap, min_lsp=args.min_lsp, min_ilsp=args.min_ilsp)
            
            # Print summary
            print_proxy_summary(results, ref_evaluator, ref_evaluatee, logger)
            
            # Save results
            output_file = output_dir / f"evaluator_{ref_evaluator}_vs_{ref_evaluatee}.json"
            
            output_data = {
                "reference_evaluator": ref_evaluator,
                "reference_evaluatee": ref_evaluatee,
                "proxies": {}
            }
            
            for result in results:
                proxy_name = result['proxy']
                output_data["proxies"][proxy_name] = {
                    'n_common': result['n_common'],
                    'n_lsp': result['n_lsp'],
                    'n_ilsp': result['n_ilsp'],
                    'lsp_rate': result['lsp_rate'],
                    'ilsp_rate': result['ilsp_rate'],
                    'reference_winrate': result['reference_winrate'],
                    'proxy_winrate': result['proxy_winrate'],
                    'lsp_ids': result['lsp_ids'],
                    'ilsp_ids': result['ilsp_ids']
                }
            
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Saved results to {output_file}")
            total_saved += 1
            
        except Exception as e:
            logger.error(f"Failed to process pair {ref_evaluator} vs {ref_evaluatee}: {e}")
            failed_pairs.append((ref_evaluator, ref_evaluatee))
    
    logger.info(f"Completed processing. Saved {total_saved} pairs.")
    if failed_pairs:
        logger.warning(f"Failed to process {len(failed_pairs)} pairs: {failed_pairs}")

if __name__ == "__main__":
    main()