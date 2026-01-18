#!/usr/bin/env python3
"""
Script to fix GPT5-judged JSON files where both original_order and flipped_order are "2".
Changes: original_order=2, flipped_order=2 -> original_order=2, flipped_order=1
"""

import json
import argparse
from pathlib import Path


def fix_double_twos(input_file: str, output_file: str = None) -> dict:
    """
    Load a GPT5-judged JSON file and fix entries where both orders are "2".

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (defaults to overwriting input)

    Returns:
        Dictionary with statistics about changes made
    """
    if output_file is None:
        output_file = input_file

    with open(input_file, 'r') as f:
        data = json.load(f)

    stats = {
        'total': len(data),
        'fixed': 0,
        'already_correct': 0
    }

    for entry in data:
        orig = entry.get('original_order', {}).get('answer')
        flip = entry.get('flipped_order', {}).get('answer')

        if orig == '2' and flip == '2':
            entry['flipped_order']['answer'] = '1'
            stats['fixed'] += 1
        else:
            stats['already_correct'] += 1

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Fix GPT5-judged JSON files where both original_order and flipped_order are "2"'
    )
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('-o', '--output', help='Path to output file (default: overwrite input)')

    args = parser.parse_args()

    stats = fix_double_twos(args.input_file, args.output)

    print(f"Total entries: {stats['total']}")
    print(f"Fixed (2,2 -> 2,1): {stats['fixed']}")
    print(f"Already correct: {stats['already_correct']}")


if __name__ == '__main__':
    main()
