#!/usr/bin/env python3
"""
Evaluate generated MBPP solutions

Usage:
    python evaluate_mbpp_solutions.py \
        --solutions_file ./solutions/meta-llama_Meta-Llama-3-8B-Instruct_mbpp_solutions.jsonl \
        --output_file ./evaluated/meta-llama_Meta-Llama-3-8B-Instruct_mbpp_evaluated.jsonl \
        --timeout 5
"""

import argparse
import json
import logging
import multiprocessing
import signal
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Exception raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds: float):
    """Context manager to limit execution time."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def unsafe_execute(code: str, test_code: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Execute code and test in a subprocess.
    """
    result = {
        'passed': False,
        'error': None,
        'output': None,
        'num_tests': 0,
        'tests_passed': 0,
    }
    
    # Combine code and test
    full_code = f"{code}\n\n{test_code}"
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        with time_limit(timeout):
            # Create a restricted globals dict
            exec_globals = {
                '__builtins__': __builtins__,
                '__name__': '__main__',
            }
            
            # Execute the code
            exec(full_code, exec_globals)
            
            result['passed'] = True
            result['output'] = sys.stdout.getvalue()
            
    except TimeoutError as e:
        result['error'] = f"Timeout: {str(e)}"
    except AssertionError as e:
        result['error'] = f"AssertionError: {str(e)}"
    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
        result['traceback'] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
    
    return result


def run_in_subprocess(code: str, test_code: str, timeout: float) -> Dict[str, Any]:
    """
    Run code execution in a subprocess for isolation.
    """
    def worker(code: str, test_code: str, result_queue):
        result = unsafe_execute(code, test_code, timeout=timeout)
        result_queue.put(result)
    
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=worker,
        args=(code, test_code, result_queue)
    )
    
    process.start()
    process.join(timeout=timeout + 1)  # Extra second for overhead
    
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            'passed': False,
            'error': f'Process timeout after {timeout}s',
            'output': None,
        }
    
    if result_queue.empty():
        return {
            'passed': False,
            'error': 'Process terminated without result',
            'output': None,
        }
    
    return result_queue.get()


def evaluate_solution(
    solution: Dict[str, Any],
    timeout: float = 5.0,
    use_subprocess: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a single solution against its test cases.
    """
    code = solution.get('extracted_code', '')
    test_list = solution.get('test_list', [])
    test_setup = solution.get('test_setup_code', '')
    challenge_tests = solution.get('challenge_test_list', [])
    
    # Combine all tests
    all_tests = test_list + challenge_tests
    test_code = '\n'.join(all_tests)
    
    if test_setup:
        test_code = f"{test_setup}\n{test_code}"
    
    # Run evaluation
    if use_subprocess:
        result = run_in_subprocess(code, test_code, timeout)
    else:
        result = unsafe_execute(code, test_code, timeout)
    
    # Add results to solution
    evaluated = solution.copy()
    evaluated['passed'] = result['passed']
    evaluated['error'] = result['error']
    evaluated['execution_output'] = result.get('output')
    evaluated['num_tests'] = len(all_tests)
    
    # Also run individual tests to get partial credit info
    individual_results = []
    for test in test_list:
        if use_subprocess:
            test_result = run_in_subprocess(code, test, timeout)
        else:
            test_result = unsafe_execute(code, test, timeout)
        individual_results.append({
            'test': test,
            'passed': test_result['passed'],
            'error': test_result['error'],
        })
    
    evaluated['individual_test_results'] = individual_results
    evaluated['tests_passed'] = sum(1 for r in individual_results if r['passed'])
    
    return evaluated


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MBPP solutions")
    
    parser.add_argument(
        "--solutions_file",
        type=str,
        required=True,
        help="Path to solutions JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output evaluated JSONL file (default: auto-generated)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Execution timeout per test in seconds"
    )
    parser.add_argument(
        "--no_subprocess",
        action="store_true",
        help="Don't use subprocess isolation (faster but less safe)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    solutions_file = Path(args.solutions_file)
    if not solutions_file.exists():
        logger.error(f"Solutions file not found: {solutions_file}")
        sys.exit(1)
    
    # Auto-generate output filename if not specified
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = solutions_file.parent / solutions_file.name.replace(
            '_solutions.jsonl', '_evaluated.jsonl'
        )
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load solutions
    logger.info(f"Loading solutions from {solutions_file}")
    solutions = []
    with open(solutions_file, 'r', encoding='utf-8') as f:
        for line in f:
            solutions.append(json.loads(line))
    
    logger.info(f"Loaded {len(solutions)} solutions")
    
    # Evaluate each solution
    logger.info("Evaluating solutions...")
    evaluated_solutions = []
    
    passed_count = 0
    for solution in tqdm(solutions, desc="Evaluating"):
        evaluated = evaluate_solution(
            solution,
            timeout=args.timeout,
            use_subprocess=not args.no_subprocess
        )
        evaluated_solutions.append(evaluated)
        
        if evaluated['passed']:
            passed_count += 1
    
    # Save results
    logger.info(f"Saving evaluated solutions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sol in evaluated_solutions:
            f.write(json.dumps(sol, ensure_ascii=False) + '\n')
    
    # Print summary
    total = len(evaluated_solutions)
    pass_rate = (passed_count / total * 100) if total > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total problems: {total}")
    logger.info(f"Passed: {passed_count} ({pass_rate:.1f}%)")
    logger.info(f"Failed: {total - passed_count} ({100 - pass_rate:.1f}%)")
    logger.info(f"{'='*60}")
    
    # Save summary
    summary = {
        'solutions_file': str(solutions_file),
        'output_file': str(output_file),
        'total': total,
        'passed': passed_count,
        'failed': total - passed_count,
        'pass_rate': pass_rate,
        'timeout': args.timeout,
        'timestamp': datetime.now().isoformat(),
    }
    
    summary_file = output_file.parent / output_file.name.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

