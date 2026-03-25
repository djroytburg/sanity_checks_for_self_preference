#!/usr/bin/env python3
"""
Filter ILSP pairs to only those where J and K fail in the same way:
Reads from llm-sp-verif proxy JSON files, writes filtered versions.
"""

import json
import re
import sys
import glob
import multiprocessing
import signal
import traceback
from io import StringIO
from pathlib import Path
from contextlib import contextmanager




def extract_mmlu_answer(text: str):
    
    patterns = [
        r'(?:answer is|correct answer is|final answer is|answer:)[^\w]*\(?([A-D])\)?',
        r'The final answer is \$\$([A-D])\$\$',
        r'\*\*\(?([A-D])\)?\*\*',
        r'\b([A-D])\b\s*[.)\n]',
        r'^\s*([A-D])\s*$',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    
    m = re.search(r'\b([A-D])\b', text[::-1], re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def extract_math500_answer(text: str):
    
    
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    
    m = re.search(r'(?:answer is|=)\s*([^\n\.]+)', text[::-1], re.IGNORECASE)
    if m:
        return m.group(1).strip()[::-1].strip()
    
    for line in reversed(text.strip().split('\n')):
        line = line.strip()
        if re.search(r'[\d\-\+\*/\^]', line) and len(line) < 60:
            return line
    return None




class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    def handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _run_single_test(code: str, test: str, timeout: float) -> bool:
    """Run one assert test against code. Returns True if passed."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        with time_limit(timeout):
            g = {'__builtins__': __builtins__, '__name__': '__main__'}
            exec(code, g)
            exec(test, g)
            return True
    except Exception:
        return False
    finally:
        sys.stdout = old_stdout


def _worker(code, test, timeout, q):
    q.put(_run_single_test(code, test, timeout))


def run_test_safe(code: str, test: str, timeout: float = 5.0) -> bool:
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(code, test, timeout, q))
    p.start()
    p.join(timeout + 1)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    return q.get() if not q.empty() else False


def extract_code(response: str) -> str:
    """Extract Python code from a response (code block or raw)."""
    m = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\s*(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def extract_tests_from_question(question: str):
    """Extract assert statements from MBPP+ question field."""
    return [line.strip() for line in question.split('\n')
            if line.strip().startswith('assert')]


def get_failed_tests(code: str, tests: list, timeout: float = 5.0):
    """Return frozenset of indices of failed tests."""
    failed = set()
    for i, test in enumerate(tests):
        if not run_test_safe(code, test, timeout):
            failed.add(i)
    return frozenset(failed)




def is_same_error_mmlu(jvr: dict, kvr: dict) -> bool:
    j_ans = extract_mmlu_answer(jvr.get('response_a', ''))
    k_ans = extract_mmlu_answer(kvr.get('response_a', ''))
    if j_ans is None or k_ans is None:
        return False
    return j_ans == k_ans


def is_same_error_math500(jvr: dict, kvr: dict) -> bool:
    j_ans = extract_math500_answer(jvr.get('response_a', ''))
    k_ans = extract_math500_answer(kvr.get('response_a', ''))
    if j_ans is None or k_ans is None:
        return False
    return j_ans.lower().strip() == k_ans.lower().strip()


def is_same_error_mbpp(jvr: dict, kvr: dict, timeout: float = 5.0) -> bool:
    question = jvr.get('question', '')
    tests = extract_tests_from_question(question)
    if not tests:
        return False
    j_code = extract_code(jvr.get('response_a', ''))
    k_code = extract_code(kvr.get('response_a', ''))
    j_failed = get_failed_tests(j_code, tests, timeout)
    k_failed = get_failed_tests(k_code, tests, timeout)
   
    return len(j_failed) > 0 and j_failed == k_failed


DATASET_FILTER = {
    'mmlu': is_same_error_mmlu,
    'math500': is_same_error_math500,
    'mbpp-plus': is_same_error_mbpp,
}


def filter_proxy_file(input_path: str, output_path: str, dataset: str):
    filter_fn = DATASET_FILTER[dataset]
    with open(input_path) as f:
        data = json.load(f)

    filtered = {}
    stats = {'ilsp_before': 0, 'ilsp_after': 0, 'lsp': 0}

    for ref, ref_data in data.items():
        filtered[ref] = {}
        for proxy, jp_data in ref_data.items():
            ilsp_items = jp_data.get('data', {}).get('ilsp', [])
            lsp_items  = jp_data.get('data', {}).get('lsp', [])
            stats['ilsp_before'] += len(ilsp_items)
            stats['lsp'] += len(lsp_items)

            kept_ilsp = []
            for item in ilsp_items:
                jvr = item.get('J_v_R', {})
                kvr = item.get('K_v_R', {})
                # Only filter when both J and K are wrong
                if not jvr.get('j_right', True) and not kvr.get('j_right', True):
                    if filter_fn(jvr, kvr):
                        kept_ilsp.append(item)
                else:
                    kept_ilsp.append(item)  # keep if not both-wrong

            stats['ilsp_after'] += len(kept_ilsp)

            filtered[ref][proxy] = {
                **{k: v for k, v in jp_data.items() if k != 'data'},
                'data': {'ilsp': kept_ilsp, 'lsp': lsp_items},
            }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(filtered, f)

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['mmlu', 'math500', 'mbpp-plus'])
    parser.add_argument('--input_dir', default='llm-sp-verif')
    parser.add_argument('--output_dir', default='llm-sp-verif-same-error')
    args = parser.parse_args()

    ds = args.dataset
    pattern = f'{args.input_dir}/{ds}/*/proxies/*.json'
    files = glob.glob(pattern)
    print(f"Found {len(files)} proxy files for {ds}")

    total_before, total_after = 0, 0
    for fp in sorted(files):
        rel = fp.replace(args.input_dir + '/', '')
        out = f'{args.output_dir}/{rel}'
        print(f"  Processing {rel} ...", end=' ', flush=True)
        stats = filter_proxy_file(fp, out, ds)
        total_before += stats['ilsp_before']
        total_after  += stats['ilsp_after']
        pct = 100 * stats['ilsp_after'] / stats['ilsp_before'] if stats['ilsp_before'] else 0
        print(f"ILSP: {stats['ilsp_before']} -> {stats['ilsp_after']} ({pct:.1f}% kept)")

    print(f"\nTotal ILSP: {total_before} -> {total_after} "
          f"({100*total_after/total_before:.1f}% kept)" if total_before else "")


if __name__ == '__main__':
    main()
