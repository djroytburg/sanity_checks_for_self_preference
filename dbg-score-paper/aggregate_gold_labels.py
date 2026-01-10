# aggregate_gold_labels.py: converts all gold judges (*_golden dirs) into aggregate pref. labels.
# Written by: Dani
# Created: 2026-1-9, 5:34 EST
# Last Modified: 2026-1-9, 5:34 EST

"""
Extracts gold judge model preferences and aggregates into a single file.
"""

import datetime
import os
import json
import argparse
from pathlib import Path
import logging
from collections import defaultdict
from tqdm import tqdm

# -------------
# --- UTILS ---
# -------------

def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(path, data):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


# ----------------------
# --- LOGGING & ARGS ---
# ----------------------

def setup_logging(logpath: str = '../file_logs'):
    """Set up a simple logger that writes to both console and file."""
    log_dir = Path(logpath)
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s:%(lineno)d - %(funcName)s()  - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'gold_agg_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate gold judge model preferences.")
    parser.add_argument('--gold_dir', type=str, default='model_preferences_fullset',
                        help='Directory containing gold judge preference files. Defaults to model_preferences_fullset')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output filepath for aggregated gold preferences. Defaults to [gold_dir]/gold_aggregated.json')
    parser.add_argument('--dataset', type=str, default='alpaca_eval',
                        help='Dataset name,  defaults to alpaca_eval')
    args = parser.parse_args()
   
    if args.output_path is None:
        args.output_path = str(Path(args.gold_dir) / f"{args.dataset}_500_id_wr" / "gold_aggregated.json")
    return args

# -----------------------
# --- MAIN LOGIC --------
# -----------------------
def main():
    """Main function to aggregate gold judge preferences."""
    args = parse_args()
    logger = setup_logging()

    gold_path = Path(args.gold_dir) / f"{args.dataset}_500_id_wr"
    
    if not gold_path.exists():
        logger.error(f"Gold preference directory not found: {gold_path}")
        return
    
    aggregated_preferences = []
    
    # Scan all gold judge directories
    logger.info(f"Scanning gold judge preference files in {gold_path}")
    gold_dirs = [d for d in gold_path.iterdir() if d.is_dir() and d.name.endswith('_golden')]
    pref_files = {}
    for gold_dir in tqdm(gold_dirs, desc="Processing gold judges"):
        # Process all preference files
        pref_file_list = list(gold_dir.glob('average_*.jsonl')) + list(gold_dir.glob('merge_*.jsonl'))
        logger.info(f"Loading {gold_dir.name} with {len(pref_file_list)} preference files.")
        for file in pref_file_list:
            fname = file.name
            dataset = load_jsonl(file)
            for data in dataset:
                if fname in pref_files:
                    logger.info(f"Updating existing preferences for file {fname}:{data['id']} from {gold_dir.name}, cur count: {len(pref_files[fname].get(data['id'], {}).get('preferences', {}))}.")
                    if data['id'] not in pref_files[fname]:
                        pref_files[fname][data['id']] = data.copy()
                        pref_files[fname][data['id']]['preferences'] = {}
                    pref_files[fname][data['id']]['preferences'].update({gold_dir.name.split("_")[1]: data['preferences']})  # Update with gold judge name prefix
                else:
                    logger.info(f"Adding new preferences for file {fname}:{data['id']} from {gold_dir.name}.")
                    pref_files[fname] = pref_files.get(fname, {})
                    pref_files[fname][data['id']] = data.copy()
                    pref_files[fname][data['id']]['preferences'] = {gold_dir.name.split("_")[1]: data['preferences']}
    # pref_files now contains all preferences from all gold judges
    # save aggregated preferences
    save_pref_files = {}
    for fname in pref_files:
        print(fname)
        print(len(pref_files[fname].keys()))
        print(set(len(v['preferences']) for v in pref_files[fname].values()))
        save_pref_files[fname] = list(pref_files[fname].values())
    with open(args.output_path, 'w') as f:
        json.dump(save_pref_files, f, indent=2)
    logger.info(f"Aggregated gold preferences saved to {args.output_path}")

if __name__ == "__main__":
    main()