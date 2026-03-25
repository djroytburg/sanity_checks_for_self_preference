#!/usr/bin/env python3
"""
Analyze ILSP mean-of-differences using same-error filtered proxy data.
Reads from llm-sp-verif-same-error/ (or llm-sp-verif/ for unfiltered baseline),
computes ILSP mean_diff, std_diff, n_ilsp per (judge, proxy, reference),
aggregates across proxies and references per judge, then across datasets,

"""

import argparse
import json
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy import stats as scipy_stats


def load_proxy_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_ilsp_stats(proxy_data: dict):
    """
    From a proxy JSON file (judge -> ref -> proxy -> data -> ilsp/lsp),
    compute per-(judge, ref, proxy) ILSP stats.

    Returns list of dicts with keys: judge, reference, proxy, mean_diff, std_diff, n_ilsp
    """
    results = []

    for ref, ref_data in proxy_data.items():
        for proxy, jp_data in ref_data.items():
            judge = jp_data.get('metadata', {}).get('judge_name', None)
            if judge is None:
                
                continue

            ilsp_items = jp_data.get('data', {}).get('ilsp', [])
            if not ilsp_items:
                continue

            diffs = []
            for item in ilsp_items:
                jvr = item.get('J_v_R', {})
                kvr = item.get('K_v_R', {})

                def avg_prob(entry):
                    g1 = entry.get('game_1', {})
                    g2 = entry.get('game_2', {})
                    p1_A = g1.get('generated_probs', {}).get('A', 0.5)
                    p1_B = g1.get('generated_probs', {}).get('B', 0.5)
                    p2_A = g2.get('generated_probs', {}).get('A', 0.5)
                    p2_B = g2.get('generated_probs', {}).get('B', 0.5)
                    denom1 = p1_A + p1_B
                    denom2 = p2_A + p2_B
                    prob1 = p1_A / denom1 if denom1 > 0 else 0.5
                    prob2 = p2_B / denom2 if denom2 > 0 else 0.5
                    return (prob1 + prob2) / 2.0

                diffs.append(avg_prob(jvr) - avg_prob(kvr))

            if not diffs:
                continue

            results.append({
                'judge': judge,
                'reference': ref,
                'proxy': proxy,
                'diffs': diffs,  
                'n_ilsp': len(diffs),
            })

    return results


def load_all_stats(data_dir: str, datasets: list):
   
    all_rows = []
    for ds in datasets:
        pattern = f'{data_dir}/{ds}/*/proxies/*.json'
        files = glob.glob(pattern)
        for fp in sorted(files):
            try:
                proxy_data = load_proxy_file(fp)
            except Exception as e:
                print(f"  Warning: could not load {fp}: {e}")
                continue

            
            # Infer judge from filename: e.g. gemma-2-27b.json -> judge = gemma-2-27b
            judge_name = Path(fp).stem

            # Patch metadata into each ref/proxy entry
            for ref, ref_data in proxy_data.items():
                for proxy, jp_data in ref_data.items():
                    if 'metadata' not in jp_data:
                        jp_data['metadata'] = {}
                    jp_data['metadata']['judge_name'] = judge_name

            rows = compute_ilsp_stats(proxy_data)
            for r in rows:
                r['dataset'] = ds
            all_rows.extend(rows)

    return all_rows


def aggregate_by_judge(rows: list):
    """Average stats across (proxy, reference, dataset) per judge.

    n_ilsp reported = mean across datasets of (total ILSP per judge per dataset),
    i.e. sum over all (proxy, ref) entries within a dataset, then averaged across datasets.
    This gives the average total ILSP examples a judge has per dataset.
    """
   
    by_judge_ds = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_judge_ds[r['judge']][r['dataset']].append(r)

    result = {}
    for judge, ds_map in by_judge_ds.items():
        per_ds_totals = []
        all_diffs = []

        for ds, recs in ds_map.items():
            # Unique ILSP count: sum of max(n_ilsp) per reference
            by_ref = defaultdict(list)
            for r in recs:
                by_ref[r['reference']].append(r['n_ilsp'])
            ds_unique = sum(max(vals) for vals in by_ref.values())
            per_ds_totals.append(ds_unique)

            for r in recs:
                all_diffs.extend(r['diffs'])

        all_diffs = np.array(all_diffs)

        result[judge] = {
            'mean_diff': float(np.mean(all_diffs)) if len(all_diffs) > 0 else 0.0,
            'std_diff': float(np.std(all_diffs, ddof=1)) if len(all_diffs) > 1 else 0.0,
            'n_ilsp_avg': float(np.mean(per_ds_totals)),
            'n_ilsp_total': int(len(all_diffs)),
            'n_datasets': len(ds_map),
        }
    return result


def fmt_model(name: str) -> str:
    return (name.replace('qwen-2.5-', 'Qwen2.5-')
               .replace('llama-3.1-', 'Llama-3.1-')
               .replace('llama-3.2-', 'Llama-3.2-')
               .replace('llama-3.3-', 'Llama-3.3-')
               .replace('gemma-2-', 'Gemma-2-')
               .replace('8b', '8B').replace('7b', '7B').replace('3b', '3B')
               .replace('14b', '14B').replace('32b', '32B').replace('72b', '72B')
               .replace('70b', '70B').replace('9b', '9B').replace('27b', '27B'))


def print_comparison_table(filtered_agg: dict, baseline_agg: dict):
    all_judges = sorted(set(filtered_agg) | set(baseline_agg))

    print()
    print("=" * 110)
    print("ILSP COMPARISON: Same-Error Filtered vs Unfiltered (averaged across datasets, proxies, references)")
    print("=" * 110)
    hdr = (f"{'Judge':<20} | "
           f"{'Filtered mean_diff':>19} {'Filtered std_diff':>18} {'Filtered n_ilsp':>16} | "
           f"{'All mean_diff':>14} {'All std_diff':>13} {'All n_ilsp':>11}")
    print(hdr)
    print("-" * 110)

    for judge in all_judges:
        f = filtered_agg.get(judge, {})
        b = baseline_agg.get(judge, {})
        f_md  = f.get('mean_diff', float('nan'))
        f_std = f.get('std_diff', float('nan'))
        f_n   = f.get('n_ilsp_avg', float('nan'))
        b_md  = b.get('mean_diff', float('nan'))
        b_std = b.get('std_diff', float('nan'))
        b_n   = b.get('n_ilsp_avg', float('nan'))
        print(f"{fmt_model(judge):<20} | "
              f"{f_md:>+19.4f} {f_std:>18.4f} {f_n:>16.1f} | "
              f"{b_md:>+14.4f} {b_std:>13.4f} {b_n:>11.1f}")

    print()


def print_latex_table(filtered_agg: dict, baseline_agg: dict):
    all_judges = sorted(set(filtered_agg) | set(baseline_agg))

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering\small\setlength{\tabcolsep}{4pt}')
    lines.append(r'\begin{tabular}{lrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r' & \multicolumn{3}{c}{\textbf{Same-Error Filtered}} & \multicolumn{3}{c}{\textbf{Unfiltered}} \\')
    lines.append(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}')
    lines.append(r'\textbf{Judge} & $\overline{\Delta}_\text{ILSP}$ & $\sigma_\Delta$ & $\bar{n}_\text{ILSP}$ '
                 r'& $\overline{\Delta}_\text{ILSP}$ & $\sigma_\Delta$ & $\bar{n}_\text{ILSP}$ \\')
    lines.append(r'\midrule')

    for judge in all_judges:
        f = filtered_agg.get(judge, {})
        b = baseline_agg.get(judge, {})
        f_md  = f.get('mean_diff', float('nan'))
        f_std = f.get('std_diff', float('nan'))
        f_n   = f.get('n_ilsp_avg', float('nan'))
        b_md  = b.get('mean_diff', float('nan'))
        b_std = b.get('std_diff', float('nan'))
        b_n   = b.get('n_ilsp_avg', float('nan'))
        lines.append(f'{fmt_model(judge)} & ${f_md:+.4f}$ & ${f_std:.4f}$ & ${f_n:.1f}$ '
                     f'& ${b_md:+.4f}$ & ${b_std:.4f}$ & ${b_n:.1f}$ \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{ILSP mean difference $\overline{J-K}$, std of differences $\sigma_\Delta$, '
                 r'and mean ILSP pair count $\bar{n}_\text{ILSP}$, averaged across datasets and proxies per judge. '
                 r'Same-error filtered restricts ILSP to cases where J and K fail in the same way '
                 r'(same wrong MC answer for MMLU, same wrong final answer for MATH500, '
                 r'same failed test cases for MBPP+).}')
    lines.append(r'\label{tab:ilsp-same-error-vs-unfiltered}')
    lines.append(r'\end{table}')

    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered_dir', default='llm-sp-verif-same-error')
    parser.add_argument('--baseline_dir', default='llm-sp-verif')
    parser.add_argument('--datasets', nargs='+', default=['math500', 'mmlu', 'mbpp-plus'])
    args = parser.parse_args()

    print(f"Loading filtered data from: {args.filtered_dir}")
    filtered_rows = load_all_stats(args.filtered_dir, args.datasets)
    print(f"  -> {len(filtered_rows)} (judge, proxy, ref, dataset) records")

    print(f"Loading baseline data from: {args.baseline_dir}")
    baseline_rows = load_all_stats(args.baseline_dir, args.datasets)
    print(f"  -> {len(baseline_rows)} (judge, proxy, ref, dataset) records")

    filtered_agg = aggregate_by_judge(filtered_rows)
    baseline_agg = aggregate_by_judge(baseline_rows)

    print_comparison_table(filtered_agg, baseline_agg)
    print_latex_table(filtered_agg, baseline_agg)


if __name__ == '__main__':
    main()
