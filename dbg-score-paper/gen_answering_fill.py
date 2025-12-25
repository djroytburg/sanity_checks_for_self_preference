# # # ####################################################### # # #
# # #              Filter Generation Answering                # # #
# # # ####################################################### # # #

# --- metadata --- #
# author: dani r.
# created: 2025-12-23, 16:20 EST
# last modified: 2025-12-23, 16:20 EST

# --- description --- #
# we need to enrich the generations that the authors already did with the metadata from the datasets.
# this script does that, noting potential index collisions and offering some baseline sanity checks.


# --- imports --- #
import json
import datasets
from utils import *
from pathlib import Path
from glob import glob
from argparse import ArgumentParser
import os

# --- main function --- #
def fill_generation_answers(
    base_dir: str = "model_responses_fullset",
    dataset: str = "alpaca_eval",
    overwrite: bool = True
    ) -> None:
    """Fill generation answers with metadata from datasets. Also, confirm no index collisions."""
    dataset_dir = os.path.join(base_dir, dataset)
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return
    print(f"Processing dataset: {dataset}")
    for file in glob(os.path.join(dataset_dir, "*.jsonl")):
        model = os.path.basename(file).replace(".jsonl", "")
        print(f"  Processing file: {file}")
        # Load existing generations
        generations = load_jsonl_data(file)
        
        # Load dataset to get metadata
        if dataset == "alpaca_eval":
            ds = datasets.load_from_disk(f"datasets/{dataset}")
        elif dataset == "translation":
            ds = datasets.load_from_disk("datasets/wmt_de-en")
        elif dataset == "truthfulness":
            ds = datasets.load_from_disk("datasets/truthful_qa")
        else:
            print(f"ERROR: Unsupported dataset: {dataset}")
            continue
        
        # Fill in metadata
        for gen in generations:
            idx = gen.get("id", None)
            if idx is None:
                print(f"    WARNING: No 'id' found in generation entry.")
                continue
            if idx < 0 or idx >= len(ds):
                print(f"    WARNING: Index {idx} out of bounds for dataset size {len(ds)}.")
                continue
            data_entry = ds[idx]
            if dataset == 'truthfulness':
                data_entry = ds[ds['id'].index(idx)]
            if dataset == "alpaca_eval":
                assert 'instruction' in data_entry, f"Dataset entry at index {idx} missing 'instruction'."
                assert 'output' in data_entry, f"Dataset entry at index {idx} missing 'output'."
                gen['query'] = data_entry.get('instruction', '')
                gen['golden_response'] = data_entry.get('output', '')
            elif dataset == "translation":
                assert 'de' in data_entry, f"Dataset entry at index {idx} missing 'de'."
                assert 'en' in data_entry, f"Dataset entry at index {idx} missing 'en'."
                gen['german'] = data_entry.get('de', '')
                gen['golden_response'] = data_entry.get('en', '')
            elif dataset == "truthfulness":
                assert 'best_answer' in data_entry, f"Dataset entry at index {idx} missing 'best_answer'."
                assert 'question' in data_entry, f"Dataset entry at index {idx} missing 'question'."
                gen['golden_response'] = data_entry.get('best_answer', '')
                gen['query'] = data_entry.get('question', '')

        # Save back the enriched generations
        if not overwrite and os.path.exists(file):
            print(f" -- {os.path.basename(file)} found in {os.path.dirname(file)}, moving to _archive/ -- ")
            Path(base_dir, "_archive", dataset).mkdir(parents=True, exist_ok=True)
            save_jsonl_data(os.path.join(base_dir, "_archive", dataset, os.path.basename(file)), load_jsonl_data(file))
        save_jsonl_data(file, generations)
        print(f" -- saved enriched generations to: {file} -- ")

# --- execute main function if run as script --- #
if __name__ == "__main__":
    parser = ArgumentParser(description="Fill generation answers with dataset metadata")
    parser.add_argument("--base_dir", type=str, default="model_responses_fullset",
                        help="Base directory for model responses")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether to overwrite existing files")
    args = parser.parse_args()
    for dataset in os.listdir(args.base_dir):
        if dataset.startswith('.'):
            continue
        print(f"\n--- Processing dataset: {dataset} ---")
        fill_generation_answers(
            base_dir=args.base_dir,
            dataset=dataset,
            overwrite=args.overwrite
        )
    
