import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


# Default paths for each source
DEFAULT_PATHS = {
    "author_obfuscation": "author_obfuscation/data/quality/proxies",
    "dbg": "dbg-score-paper/proxy_preference_data",
    "verifiable": "llm-sp-verif",
}

DBG_DATASETS = ["alpaca_eval", "translation", "truthfulness"]
VERIFIABLE_DATASETS = ["math500", "mbpp-plus", "mmlu"]


def load_proxy_files_author_obfuscation(proxy_dir: Path) -> list:
    """
    Load proxy files from author_obfuscation format.
    
    Files: evaluator_{judge}_vs_{reference}.json
    Structure: reference_evaluator, reference_evaluatee, proxies with lsp_ids/ilsp_ids
    """
    proxy_files = list(proxy_dir.glob("evaluator_*.json"))
    data = []
    
    for pf in proxy_files:
        with open(pf) as f:
            content = json.load(f)
        
        # Convert to common format
        proxies_converted = {}
        for proxy_name, proxy_info in content["proxies"].items():
            proxies_converted[proxy_name] = {
                "lsp_ids": proxy_info.get("lsp_ids", []),
                "ilsp_ids": proxy_info.get("ilsp_ids", []),
            }
        
        data.append({
            "judge": content["reference_evaluator"],
            "reference": content["reference_evaluatee"],
            "proxies": proxies_converted,
            "dataset": "quality",
            "file": pf.name
        })
    
    return data


def load_proxy_files_dbg(proxy_dir: Path, datasets: list) -> list:
    """
    Load proxy files from dbg-score format.
    
    Files: judge_{judge}_vs_{reference}.json in dataset subdirectories
    Structure: judge_name, reference_name, proxies with data.lsp/data.ilsp containing objects with id field
    """
    data = []
    
    for dataset in datasets:
        dataset_dir = proxy_dir / dataset
        if not dataset_dir.exists():
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue
        
        proxy_files = list(dataset_dir.glob("judge_*.json"))
        
        for pf in proxy_files:
            with open(pf) as f:
                content = json.load(f)
            
            # Convert to common format - extract IDs from data objects
            proxies_converted = {}
            for proxy_name, proxy_info in content["proxies"].items():
                lsp_data = proxy_info.get("data", {}).get("lsp", [])
                ilsp_data = proxy_info.get("data", {}).get("ilsp", [])
                
                # Extract IDs from the data objects
                lsp_ids = [item["id"] for item in lsp_data if "id" in item]
                ilsp_ids = [item["id"] for item in ilsp_data if "id" in item]
                
                proxies_converted[proxy_name] = {
                    "lsp_ids": lsp_ids,
                    "ilsp_ids": ilsp_ids,
                }
            
            data.append({
                "judge": content["judge_name"],
                "reference": content["reference_name"],
                "proxies": proxies_converted,
                "dataset": dataset,
                "file": pf.name
            })
    
    return data


def load_proxy_files_verifiable(proxy_dir: Path, datasets: list) -> list:
    """
    Load proxy files from verifiable format.
    
    Structure: llm-sp-verif/{dataset}/{family}/proxies/{judge}.json
    Each file contains: {reference} -> {proxy} -> data.lsp/data.ilsp with J_v_R.example_id
    """
    data = []
    
    for dataset in datasets:
        dataset_dir = proxy_dir / dataset
        if not dataset_dir.exists():
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue
        
        # Find all proxies folders (can be in family subdirectories)
        proxy_dirs = list(dataset_dir.glob("*/proxies"))
        
        for pdir in proxy_dirs:
            proxy_files = list(pdir.glob("*.json"))
            
            for pf in proxy_files:
                # Judge name is the filename (e.g., llama-3.1-8b.json -> llama-3.1-8b)
                judge_name = pf.stem
                
                with open(pf) as f:
                    content = json.load(f)
                
                # Iterate over references
                for reference_name, ref_data in content.items():
                    # Iterate over proxies for this reference
                    proxies_converted = {}
                    
                    for proxy_name, proxy_info in ref_data.items():
                        lsp_data = proxy_info.get("data", {}).get("lsp", [])
                        ilsp_data = proxy_info.get("data", {}).get("ilsp", [])
                        
                        # Extract example IDs from J_v_R objects
                        lsp_ids = []
                        for item in lsp_data:
                            if "J_v_R" in item and "example_id" in item["J_v_R"]:
                                lsp_ids.append(item["J_v_R"]["example_id"])
                        
                        ilsp_ids = []
                        for item in ilsp_data:
                            if "J_v_R" in item and "example_id" in item["J_v_R"]:
                                ilsp_ids.append(item["J_v_R"]["example_id"])
                        
                        proxies_converted[proxy_name] = {
                            "lsp_ids": lsp_ids,
                            "ilsp_ids": ilsp_ids,
                        }
                    
                    if proxies_converted:  # Only add if there are proxies
                        data.append({
                            "judge": judge_name,
                            "reference": reference_name,
                            "proxies": proxies_converted,
                            "dataset": dataset,
                            "file": pf.name
                        })
    
    return data


def load_proxy_files(source: str, proxy_dir: Path, datasets: list = None) -> list:
    """
    Load proxy files based on source type.
    """
    if source == "author_obfuscation":
        return load_proxy_files_author_obfuscation(proxy_dir)
    elif source == "dbg":
        if datasets is None:
            datasets = DBG_DATASETS
        return load_proxy_files_dbg(proxy_dir, datasets)
    elif source == "verifiable":
        if datasets is None:
            datasets = VERIFIABLE_DATASETS
        return load_proxy_files_verifiable(proxy_dir, datasets)
    else:
        raise ValueError(f"Unknown source: {source}")


def count_proxies_per_example(proxy_data: list, ilsp_only: bool = True) -> dict:
    """
    For each (example_id, judge, reference) triplet, count how many proxies cover it.
    
    Args:
        proxy_data: List of dicts from load_proxy_files()
        ilsp_only: If True, only count ILSP examples (judge wrong, reference correct).
                   If False, count both LSP and ILSP examples.
    
    Returns:
        Dict mapping (example_id, judge, reference) -> count of proxies
    """
    triplet_counts = defaultdict(int)
    
    for entry in proxy_data:
        judge = entry["judge"]
        reference = entry["reference"]
        
        for proxy_name, proxy_info in entry["proxies"].items():
            if ilsp_only:
                # Only count ILSP examples (judge wrong, reference correct)
                example_ids = set(proxy_info.get("ilsp_ids", []))
            else:
                # Count both LSP and ILSP examples
                lsp_ids = proxy_info.get("lsp_ids", [])
                ilsp_ids = proxy_info.get("ilsp_ids", [])
                example_ids = set(lsp_ids) | set(ilsp_ids)
            
            for example_id in example_ids:
                triplet_counts[(example_id, judge, reference)] += 1
    
    return triplet_counts


def compute_percentage_at_thresholds(triplet_counts: dict, max_threshold: int = None) -> tuple:
    """
    Compute the percentage of triplets with at least N proxies for each threshold N.
    
    Args:
        triplet_counts: Dict mapping triplet -> count
        max_threshold: Maximum threshold to consider (default: max count in data)
    
    Returns:
        Tuple of (thresholds, percentages)
    """
    if not triplet_counts:
        return [], []
    
    counts = list(triplet_counts.values())
    total = len(counts)
    
    if max_threshold is None:
        max_threshold = max(counts)
    
    thresholds = list(range(1, max_threshold + 1))
    percentages = []
    
    for n in thresholds:
        n_at_least = sum(1 for c in counts if c >= n)
        percentages.append(100.0 * n_at_least / total)
    
    return thresholds, percentages


def create_plot(thresholds: list, percentages: list, output_path: Path, title: str = None):
    """
    Create the proxy robustness plot.
    
    Args:
        thresholds: List of N values (at least N proxies)
        percentages: Corresponding percentages
        output_path: Where to save the plot
        title: Optional plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, percentages, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.fill_between(thresholds, percentages, alpha=0.3, color='steelblue')
    
    ax.set_xlabel("Minimum Number of Proxies (N)", fontsize=12)
    ax.set_ylabel("Percentage of Triplets (%)", fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("Percentage of (Example, Judge, Reference) Triplets\nwith At Least N Valid Proxies", fontsize=14)
    
    ax.set_xlim(left=1)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # Add data labels
    for x, y in zip(thresholds, percentages):
        if y > 5:  # Only label if percentage is visible
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved plot to {output_path} and {output_path.with_suffix('.pdf')}")
    plt.close()


def print_summary(triplet_counts: dict, thresholds: list, percentages: list, filter_label: str = "", source_label: str = ""):
    """Print summary statistics."""
    counts = list(triplet_counts.values())
    
    print("\n" + "=" * 60)
    print(f"PROXY COVERAGE SUMMARY")
    if source_label:
        print(f"Source: {source_label}")
    if filter_label:
        print(f"Filter: {filter_label}")
    print("=" * 60)
    print(f"Total (example, judge, reference) triplets: {len(triplet_counts):,}")
    print(f"Min proxies per triplet: {min(counts)}")
    print(f"Max proxies per triplet: {max(counts)}")
    print(f"Mean proxies per triplet: {np.mean(counts):.2f}")
    print(f"Median proxies per triplet: {np.median(counts):.1f}")
    print()
    print("Percentage of triplets with at least N proxies:")
    print("-" * 40)
    for n, pct in zip(thresholds, percentages):
        bar = "█" * int(pct / 5)
        print(f"  N >= {n:2d}: {pct:6.2f}%  {bar}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Plot proxy coverage across (example, judge, reference) triplets"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["author_obfuscation", "dbg", "verifiable"],
        default="author_obfuscation",
        help="Data source: author_obfuscation, dbg, or verifiable"
    )
    parser.add_argument(
        "--proxy_dir",
        type=str,
        default=None,
        help="Directory containing proxy JSON files (default: auto based on source)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name or 'all'. For dbg: alpaca_eval, translation, truthfulness. For verifiable: math500, mbpp-plus, mmlu"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (default: auto based on source/dataset)"
    )
    parser.add_argument(
        "--max_threshold",
        type=int,
        default=None,
        help="Maximum threshold N to show (default: auto from data)"
    )
    parser.add_argument(
        "--include_lsp",
        action="store_true",
        help="Include both LSP and ILSP examples (default: ILSP only)"
    )
    args = parser.parse_args()
    
    # Set default proxy directory based on source
    if args.proxy_dir is None:
        args.proxy_dir = DEFAULT_PATHS[args.source]
    
    proxy_dir = Path(args.proxy_dir)
    
    # Set default output path
    if args.output is None:
        if args.source == "dbg":
            args.output = f"proxy_robustness_plot_dbg_{args.dataset}.png"
        elif args.source == "verifiable":
            args.output = f"proxy_robustness_plot_verifiable_{args.dataset}.png"
        else:
            args.output = "proxy_robustness_plot_author_obfuscation.png"
    
    output_path = Path(args.output)
    
    if not proxy_dir.exists():
        print(f"ERROR: Proxy directory not found: {proxy_dir}")
        return
    
    # Determine datasets for dbg/verifiable source
    datasets = None
    if args.source == "dbg":
        if args.dataset == "all":
            datasets = DBG_DATASETS
        else:
            datasets = [args.dataset]
    elif args.source == "verifiable":
        if args.dataset == "all":
            datasets = VERIFIABLE_DATASETS
        else:
            datasets = [args.dataset]
    
    print(f"Loading proxy files from: {proxy_dir}")
    if args.source in ["dbg", "verifiable"]:
        print(f"Datasets: {datasets}")
    
    proxy_data = load_proxy_files(args.source, proxy_dir, datasets)
    print(f"Loaded {len(proxy_data)} (judge, reference) pair files")
    
    # Count unique datasets
    unique_datasets = set(d["dataset"] for d in proxy_data)
    print(f"Datasets found: {unique_datasets}")
    
    ilsp_only = not args.include_lsp
    filter_label = "ILSP only" if ilsp_only else "LSP + ILSP"
    print(f"Counting proxies per (example, judge, reference) triplet ({filter_label})...")
    triplet_counts = count_proxies_per_example(proxy_data, ilsp_only=ilsp_only)
    
    if not triplet_counts:
        print("ERROR: No triplets found!")
        return
    
    print("Computing percentages at each threshold...")
    thresholds, percentages = compute_percentage_at_thresholds(
        triplet_counts, max_threshold=args.max_threshold
    )
    
    source_label = f"{args.source}"
    if args.source in ["dbg", "verifiable"]:
        source_label += f" ({args.dataset})"
    
    print_summary(triplet_counts, thresholds, percentages, filter_label, source_label)
    
    print(f"\nCreating plot...")
    title = f"Percentage of (Example, Judge, Reference) Triplets\nwith At Least N Valid Proxies\n({source_label}, {filter_label})"
    create_plot(thresholds, percentages, output_path, title=title)


if __name__ == "__main__":
    main()
