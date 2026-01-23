#!/usr/bin/env python3


import json
import sys
from collections import Counter, defaultdict
import matplotlib.pyplot as plt


def extract_model_family(model_name):
    """Extract the base model family from a model name."""
    
    parts = model_name.split('-')
    if parts:
        return parts[0]
    return model_name


def plot_nilsp_distribution(summary_stats_path):
    
    
    with open(summary_stats_path, 'r') as f:
        data = json.load(f)

   
    import os
    path_parts = summary_stats_path.split(os.sep)
    dataset_name = "Unknown Dataset"
    for i, part in enumerate(path_parts):
        if 'analysis' in part and i > 0:
            dataset_name = path_parts[i-1]
            break
    else:
        
        for part in path_parts:
            if part and part not in ['analysis', 'summary_statistics.json', '.', '..']:
                dataset_name = part

    s
    judge_family_nilsp = defaultdict(list)

    for entry in data:
        judge_family = extract_model_family(entry['judge'])
        judge_family_nilsp[judge_family].append(entry['n_ilsp'])

   
    plt.figure(figsize=(12, 7))

    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

   
    for idx, (judge_family, nilsp_values) in enumerate(sorted(judge_family_nilsp.items())):
       
        nilsp_counts = Counter(nilsp_values)

        
        sorted_nilsp = sorted(nilsp_counts.items())
        x_values = [x for x, _ in sorted_nilsp]
        y_values = [y for _, y in sorted_nilsp]

        # Plot the line
        plt.plot(x_values, y_values, marker='o', linewidth=2.5, markersize=6,
                label=judge_family, alpha=0.8, color=colors[idx % len(colors)])

    plt.xlabel('Number of n_ilsp Examples', fontsize=12)
    plt.ylabel('Number of Judge-Proxy-Reference Pairs', fontsize=12)
    plt.title(f'{dataset_name}: Distribution by n_ilsp Count (by Judge Family)',
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # Save the plot
    output_path = summary_stats_path.replace('summary_statistics.json', 'nilsp_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Also show the plot
    plt.show()

    # Print summary statistics
    print(f"\nSummary by Judge Family:")
    print(f"Total judge families: {len(judge_family_nilsp)}")
    for judge_family, nilsp_values in sorted(judge_family_nilsp.items()):
        print(f"\n{judge_family}:")
        print(f"  Total pairs: {len(nilsp_values)}")
        print(f"  n_ilsp range: {min(nilsp_values)} to {max(nilsp_values)}")
        print(f"  Mean n_ilsp: {sum(nilsp_values)/len(nilsp_values):.2f}")
        print(f"  Median n_ilsp: {sorted(nilsp_values)[len(nilsp_values)//2]}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_summary_statistics.json>")
        sys.exit(1)

    plot_nilsp_distribution(sys.argv[1])
