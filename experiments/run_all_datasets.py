#!/usr/bin/env python3
"""
Run comprehensive evaluation on all three currency pairs
"""
import subprocess
import sys
from pathlib import Path

DATASETS = [
    ('USD_TWD', 'dataset/USD_TWD.csv'),
    ('JPY_TWD', 'dataset/JPY_TWD.csv'),
    ('KRW_TWD', 'dataset/KRW_TWD.csv'),
]

def run_evaluation(dataset_name, dataset_path, epochs=100, seed=42):
    """Run evaluation on a single dataset"""
    print("\n" + "="*80)
    print(f"Running: {dataset_name}")
    print("="*80 + "\n")

    cmd = [
        'uv', 'run', 'python', 'experiments/comprehensive_evaluation.py',
        '--experiment', 'all',
        '--dataset', dataset_path,
        '--epochs', str(epochs),
        '--seed', str(seed)
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {dataset_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {dataset_name} failed with error: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run evaluation on all datasets')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--datasets', type=str, nargs='+',
                       choices=['USD_TWD', 'JPY_TWD', 'KRW_TWD'],
                       help='Specific datasets to run (default: all)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 Running Comprehensive Evaluation on All Datasets")
    print("="*80)

    # Filter datasets if specified
    datasets_to_run = DATASETS
    if args.datasets:
        datasets_to_run = [(name, path) for name, path in DATASETS if name in args.datasets]

    total = len(datasets_to_run)
    success_count = 0

    for i, (dataset_name, dataset_path) in enumerate(datasets_to_run, 1):
        print(f"\n>>> [{i}/{total}] {dataset_name}")
        if run_evaluation(dataset_name, dataset_path, args.epochs, args.seed):
            success_count += 1

    # Summary
    print("\n" + "="*80)
    print(f"✅ Completed: {success_count}/{total} datasets")
    print("="*80)

    # List results
    results_dir = Path('experiments/comprehensive_results')
    if results_dir.exists():
        print("\n📊 Results saved:")
        for json_file in sorted(results_dir.glob('*.json')):
            print(f"   {json_file.name}")

    return 0 if success_count == total else 1

if __name__ == "__main__":
    sys.exit(main())
