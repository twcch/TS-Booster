"""Run ablation study on all 13 datasets sequentially."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.comprehensive_evaluation import ComprehensiveTSBoosterExperiment, set_random_seed

DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

def main():
    experiment = ComprehensiveTSBoosterExperiment()
    all_results = {}

    for i, ds in enumerate(DATASETS, 1):
        print(f"\n{'#'*80}")
        print(f"### [{i}/{len(DATASETS)}] Dataset: {ds}")
        print(f"{'#'*80}\n")

        dataset_path = f"dataset/{ds}.csv"
        if not Path(dataset_path).exists():
            print(f"SKIP: {dataset_path} not found")
            continue

        set_random_seed(42)
        results = experiment.run_ablation_study(
            dataset_path=dataset_path,
            epochs=100,
            quick_test=False,
            clean_old_results=True
        )
        all_results[ds] = results

        # Print summary
        print(f"\n{'='*60}")
        print(f"Summary for {ds}:")
        for name, m in results.items():
            print(f"  {name:12s} | RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.4f} | MDA: {m['MDA']:.2f}%")

        if 'TS-Booster' in results and '+Expert' in results:
            gap = (results['+Expert']['RMSE'] - results['TS-Booster']['RMSE']) / results['+Expert']['RMSE'] * 100
            print(f"  TS-Booster vs +Expert: {gap:+.2f}% RMSE improvement")
        print(f"{'='*60}")

    # Final cross-dataset summary
    print(f"\n\n{'#'*80}")
    print("FINAL CROSS-DATASET SUMMARY")
    print(f"{'#'*80}")
    print(f"{'Dataset':12s} | {'Vanilla':>8s} | {'+Decomp':>8s} | {'+Expert':>8s} | {'TS-Boost':>8s} | {'Gap vs +E':>10s}")
    print("-" * 70)

    wins = 0
    for ds, results in all_results.items():
        v = results.get('Vanilla', {}).get('RMSE', 0)
        d = results.get('+Decomp', {}).get('RMSE', 0)
        e = results.get('+Expert', {}).get('RMSE', 0)
        t = results.get('TS-Booster', {}).get('RMSE', 0)
        gap = (e - t) / e * 100 if e > 0 else 0
        winner = "WIN" if t < e else "LOSS"
        if t < e: wins += 1
        print(f"{ds:12s} | {v:8.4f} | {d:8.4f} | {e:8.4f} | {t:8.4f} | {gap:+8.2f}% {winner}")

    print(f"\nTS-Booster wins: {wins}/{len(all_results)} datasets")

if __name__ == "__main__":
    main()
