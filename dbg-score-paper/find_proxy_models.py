#!/usr/bin/env python3
"""
Find suitable proxy models for judge swap test.

For a given Judge J and reference model R, find proxy models K where:
- Gold judges agree on quality: Gold(J vs R) = Gold(K vs R) for many examples
- This ensures K has similar capability to J (per gold standard)
"""

from glob import glob
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

def load_jsonl(filepath):
    with open(filepath) as f:
        return [json.loads(line) for line in f]

def load_gold_labels(pref_dir, dataset, model1, model2):
    """Load gold labels with majority vote. Gold judges use 'merge_' prefix."""
    gold_judges = ['gpt-4o-mini_golden', 'gemini-flash-1.5_golden', 'deepseek-v3_golden']
    votes = defaultdict(list)
    
    for judge in gold_judges:
        eval_dir = pref_dir / f"{dataset}_500_id_wr" / f"evaluator_{judge}"
        # Gold judges use 'merge_' prefix
        path1 = eval_dir / f"merge_{model1}_{model2}.jsonl"
        path2 = eval_dir / f"merge_{model2}_{model1}.jsonl"
        path = path1 if path1.exists() else (path2 if path2.exists() else None)
        if not path:
            continue
        data = load_jsonl(path)
        # print(f"Gold labels for {model1} vs {model2}, Judge: {judge}, Length of data: {len(data)}")
        for item in data:

            pref = item.get('preferences', '')
            # preferences is a list [pos1_pref, pos2_pref] for position bias testing
            # Only count as a vote if BOTH positions agree (high confidence)
            if isinstance(pref, list) and len(pref) == 2:
                if pref[0] == pref[1] and pref[0] in [model1, model2]:
                    votes[item['id']].append(pref[0])
            elif isinstance(pref, str) and pref in [model1, model2]:
                votes[item['id']].append(pref)
            
    # Majority vote
    labels = {}
    for eid, v in votes.items():
        counts = defaultdict(int)
        for vote in v:
            counts[vote] += 1
        if counts:
            winner, count = max(counts.items(), key=lambda x: x[1])
            if count >= 2:
                labels[eid] = winner
    
    return labels

def find_proxy_candidates(pref_dir, dataset, reference_name, judge_name = None):
    """Find all candidates gold judges have evaluated."""
    eval_dir = pref_dir / f"{dataset}_500_id_wr"
    if not eval_dir.exists():
        raise ValueError(f"ERROR: {eval_dir} not found")
    gold_judge_paths = eval_dir.glob("*_golden")
    candidates = dict()
    for path in gold_judge_paths:
        gold_judge = Path(path).name.replace("evaluator_", "").replace("_golden", "")
        for filepath in path.glob("merge_*.jsonl"):
            parts = filepath.stem.replace("merge_", "").replace(".jsonl","").split("_")
            if reference_name in parts:
                other = filepath.stem.replace("merge_", "").replace(".jsonl","").replace(reference_name + "_", "").replace("_" + reference_name, "")
                if other and other not in (reference_name, judge_name):
                    if other not in candidates:
                        candidates[other] = []
                    candidates[other].append(gold_judge)
    # import json
    # print(f"Debugging candidate search: Final candidate dict:\n{json.dumps(candidates, indent=2)}")

    return set([candidate for candidate, count in candidates.items() if len(count) > 2])

def compute_agreement(judge_gold, proxy_gold, judge_name, proxy_name, other_name, candidates=None):
    """
    Compute how many examples have matched gold quality.
    Returns: (n_agree, n_total, agreement_rate)
    """
    
    common_ids = set(judge_gold.keys()) & set(proxy_gold.keys())
    all_ids = set(judge_gold.keys()) | set(proxy_gold.keys())
    # print("\tComparing on {} common IDs ({} total IDs)".format(len(common_ids), len(all_ids)))
    # print("\tJudge's ids: {}, Proxy's ids: {}".format(len(judge_gold), len(proxy_gold)))
    
    n_agree = 0
    n_right = 0
    n_wrong = 0
    proxy_winrate = sum(1 for v in proxy_gold.values() if v == proxy_name) / len(proxy_gold) if proxy_gold else 0
    agreement_ids = {'lsp': [], 'ilsp': []}
    for eid in common_ids:
        j_gold = judge_gold[eid]
        k_gold = proxy_gold[eid]
        if candidates:
            if j_gold not in candidates:
                raise ValueError(f"ERROR: Gold labels for candidate {judge_name} not in candidates: {candidates}")
            elif k_gold not in candidates:
                raise ValueError(f"ERROR: Gold labels for candidate {proxy_name} not in candidates: {candidates}")
        # Check if both beat other, or both lose to other
        judge_wins = (j_gold == judge_name)
        proxy_wins = (k_gold == proxy_name)
        
        if judge_wins == proxy_wins:
            if judge_wins:
                n_right += 1
                agreement_ids['lsp'].append(eid)
            else:
                n_wrong += 1
                agreement_ids['ilsp'].append(eid)
            n_agree += 1
    
    n_total = len(common_ids)
    agreement_rate = n_agree / n_total if n_total > 0 else 0
    
    return n_agree, n_total, agreement_rate, n_right, n_wrong, proxy_winrate, agreement_ids

def find_proxy_model(judge_name, reference_name, dataset, pref_dir):
    
    # Find all candidates
    candidates = find_proxy_candidates(pref_dir, dataset, reference_name, judge_name)
    if len(candidates) == 0:
        print(f"No candidate proxy models found for Judge={judge_name}, Reference={reference_name}")
        return []
    
    # Load gold labels for Judge vs Reference (this is constant)
    judge_gold = load_gold_labels(pref_dir, dataset, judge_name, reference_name)
    judge_winrate = sum(1 for v in judge_gold.values() if v == judge_name) / len(judge_gold) if judge_gold else 0
    
    if not judge_gold:
        print(f"ERROR: No gold labels found for {judge_name} vs {reference_name}")
        return []
    
    # For each candidate proxy, check agreement with judge
    all_results = []
    for proxy_name in candidates:
        if proxy_name == reference_name or proxy_name == judge_name:
            continue
        
        # Load gold labels for Proxy vs Reference
        proxy_gold = load_gold_labels(pref_dir, dataset, proxy_name, reference_name)
        
        if not proxy_gold:
            continue
        
        # Compute agreement
        n_agree, n_total, rate, n_right, n_wrong, proxy_winrate, agreement_ids = compute_agreement(
            judge_gold, proxy_gold, judge_name, proxy_name, reference_name
        )
        
        # Store all results with at least 5 LSP and 5 ILSP examples
        if n_right >= 5 and n_wrong >= 5:
            result_data = {
                'proxy': proxy_name,
                'reference': reference_name,
                'n_agree': n_agree,
                'n_total': n_total,
                'n_right': n_right,
                'n_wrong': n_wrong,
                'judge_winrate': judge_winrate,
                'proxy_winrate': proxy_winrate,
                'agreement_rate': rate,
                'agreement_ids': agreement_ids
            }
            all_results.append(result_data)
    
    # Sort by number of ILSP examples (for reporting)
    all_results.sort(key=lambda x: x['n_wrong'], reverse=True)
    
    print()
    print("=" * 80)
    print(f"PROXY MODELS for Judge={judge_name}, Reference={reference_name}")
    print("=" * 80)
    winrate_text = f"J.WR {judge_winrate:.1%}"
    print(f"{'Proxy Model':<35} {'Agree':>6} {'Total':>6} {'Rate':>6} {'LSP':>6} {'ILSP':>6} {winrate_text:>12} {'K.WR':>8}")
    print("-" * 80)
    
    for r in all_results[:30]:  # Show top 30
        print(f"{r['proxy']:<35} {r['n_agree']:>6} {r['n_total']:>6} {r['agreement_rate']:>6.1%} {r['n_right']:>6} {r['n_wrong']:>6} {judge_winrate:>12.1%} {r['proxy_winrate']:>8.1%}")
    print("-" * 80)
    print(f"Found {len(all_results)} suitable proxy models")
    print()
    
    return all_results

def build_proxy_dataset(judge_name, proxy_name, reference_name, agreement_ids, dataset, response_dir):
    info_dir = response_dir / dataset
    if judge_name + ".jsonl" not in os.listdir(info_dir):
        raise ValueError(f"ERROR: Judge info file not found for {judge_name} in {info_dir}")
    elif proxy_name + ".jsonl" not in os.listdir(info_dir):
        raise ValueError(f"ERROR: Proxy info file not found for {proxy_name} in {info_dir}")
    elif reference_name + ".jsonl" not in os.listdir(info_dir):
        raise ValueError(f"ERROR: Reference info file not found for {reference_name} in {info_dir}")
    
    judge_data = load_jsonl(info_dir / f"{judge_name}.jsonl")
    proxy_data = load_jsonl(info_dir / f"{proxy_name}.jsonl")
    reference_data = load_jsonl(info_dir / f"{reference_name}.jsonl")
    agreement_data = {}
    for type, ids in agreement_ids.items():
        agreement_data[type] = []
        for eid in ids:
            judge_item = next((item for item in judge_data if item['id'] == eid), None)
            proxy_item = next((item for item in proxy_data if item['id'] == eid), None)
            reference_item = next((item for item in reference_data if item['id'] == eid), None)
            if judge_item is None:
                raise ValueError(f"ERROR: ID {eid} not found in judge data for {judge_name}, dir {info_dir}")
            elif proxy_item is None:
                raise ValueError(f"ERROR: ID {eid} not found in proxy data for {proxy_name}, dir {info_dir}")
            elif reference_item is None:
                raise ValueError(f"ERROR: ID {eid} not found in reference data for {reference_name}, dir {info_dir}")
            if judge_item and proxy_item and reference_item:
                assert judge_item['id'] == proxy_item['id'] == reference_item['id'] == eid
                query_key = 'german' if dataset == 'translation' else 'query'
                assert judge_item[query_key] == proxy_item[query_key] == reference_item[query_key]
                agreement_data[type].append({
                    'id': eid,
                    'query': judge_item[query_key],
                    'judge': judge_item['model_response'],
                    'proxy': proxy_item['model_response'],
                    'reference': reference_item['model_response']
                })
    return {
        "judge_name": judge_name,
        "proxy_name": proxy_name,
        "reference_name": reference_name,
        "dataset": dataset,
        "data": agreement_data
    }
def main():
    if len(sys.argv) < 3:
        print("Usage: python find_proxy_models.py <dataset> <pref_dir>")
        print("Example: python find_proxy_models.py alpaca_eval dbg-score-paper/model_preferences_fullset")
        sys.exit(1)
    

    dataset = sys.argv[1]
    pref_dir = Path(sys.argv[2])
    eval_dir = pref_dir / f"{dataset}_500_id_wr"
    judges_ref_dict = dict()
    for judge_path in eval_dir.glob("evaluator_*"):
        if "_golden" in judge_path.name:
            continue
        elif "fewshot" in judge_path.name:
            continue
        judge_name = judge_path.name.replace("evaluator_", "")
        for filepath in list(judge_path.glob("average_*.jsonl")) + list(judge_path.glob("merge_*.jsonl")):
            parts = filepath.stem.replace("average_", "").replace("merge_", "").replace(".jsonl","").split("_")
            assert len(parts) == 2
            model1, model2 = parts
            if judge_name not in judges_ref_dict:
                judges_ref_dict[judge_name] = set()
            if model1 == judge_name:
                judges_ref_dict[judge_name].add(model2)
            elif model2 == judge_name:
                judges_ref_dict[judge_name].add(model1)
    print(f"Dataset: {dataset}, Preference dir: {pref_dir}")
    print("Finding all proxy candidates for each Judge-Reference pair...")
    failed = []
    saved_count = 0
    
    for judge_name, references in judges_ref_dict.items():
        for reference_name in references:
            results = find_proxy_model(judge_name, reference_name, dataset, pref_dir)
            if not results:
                failed.append((judge_name, reference_name))
                continue
            
            # Build structure with all proxies for this judge-reference pair
            output_data = {
                "judge_name": judge_name,
                "reference_name": reference_name,
                "dataset": dataset,
                "proxies": {}
            }
            
            # Collect data for each proxy
            for result in results:
                assert reference_name == result['reference']
                proxy_name = result['proxy']
                
                agreement_data = build_proxy_dataset(
                    judge_name, proxy_name, reference_name, result['agreement_ids'], dataset,
                    response_dir=Path("dbg-score-paper/model_responses_fullset")
                )
                assert len(agreement_data['data']['lsp']) == result['n_right'] == len(result['agreement_ids']['lsp'])
                assert len(agreement_data['data']['ilsp']) == result['n_wrong'] == len(result['agreement_ids']['ilsp'])
                
                # Store this proxy's data
                output_data["proxies"][proxy_name] = {
                    'n_agree': result['n_agree'],
                    'n_total': result['n_total'],
                    'n_right': result['n_right'],
                    'n_wrong': result['n_wrong'],
                    'agreement_rate': result['agreement_rate'],
                    'judge_winrate': result['judge_winrate'],
                    'proxy_winrate': result['proxy_winrate'],
                    'data': agreement_data['data']
                }
            
            # Save single file with all proxies for this judge-reference pair
            output_path = Path("dbg-score-paper/proxy_preference_data") / dataset
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"judge_{judge_name}_vs_{reference_name}.json"
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            saved_count += 1
            print(f"  → Saved: {output_file.name} ({len(output_data['proxies'])} proxies)")
    
    print()
    print("=" * 80)
    print(f"SUMMARY: Saved {saved_count} judge-reference pairs")
    print("=" * 80)
    
    if len(failed) > 0:
        print("\nFailed to find proxy models for the following judge-reference pairs:")
        for judge_name, reference_name in failed:
            print(f"  Judge: {judge_name}, Reference: {reference_name}")

if __name__ == "__main__":
    main()
