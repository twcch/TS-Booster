"""
Multi-seed ablation: +Expert vs TS-Booster.

Measures the marginal contribution of decomposition + SNR gate + trend enrichment.
Runs 5 seeds on all 16 datasets.

Usage:
  uv run python experiments/multi_seed_test.py              # full run
  uv run python experiments/multi_seed_test.py --analyze    # analyze saved results
"""
import sys, random, json, argparse
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.layers import LinearBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.engine.trainer import train_v11
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data
from sklearn.metrics import mean_squared_error

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

SEEDS = [42, 123, 456, 789, 2024]
RESULTS_PATH = Path("experiments/comprehensive_results/ablation/multi_seed_expert_vs_tsb.json")
FIGURES_DIR = Path("experiments/comprehensive_results/ablation/figures")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, test_loader, scaler, device):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["raw_input"].to(device); y = batch["target"].to(device)
            output = model(x)
            pred = output[0] if isinstance(output, tuple) else output
            preds.append(pred.cpu().numpy()); targets.append(y.cpu().numpy())
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    if scaler: preds = preds * scaler.scale_[0] + scaler.mean_[0]; targets = targets * scaler.scale_[0] + scaler.mean_[0]
    return np.sqrt(mean_squared_error(targets.flatten(), preds.flatten()))


def run_pair(ds, seed, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BEST_STAGE3_CONFIG.copy()
    config['epochs'] = epochs; config['input_channels'] = 2
    seq_len, pred_len = 30, 3

    path = f"dataset/{ds}.csv"
    if not Path(path).exists(): return None, None
    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(df, lookback=seq_len, horizon=pred_len, batch_size=32)

    base_kwargs = dict(
        seq_len=seq_len, pred_len=pred_len, input_channels=2,
        decomp_kernels=[7, 13],
        expert_kernels=config.get('expert_kernels', [3, 5, 7]),
        expert_hidden_dim=config.get('expert_hidden_dim', 32),
        dropout_p=config.get('dropout', 0.3),
        n_boost_rounds=config.get('n_boost_rounds', 2),
        expert_type=config.get('expert_type', 'cnn'),
    )

    # +Expert
    set_seed(seed)
    em = FlexibleTSBooster(base_model=LinearBackbone(seq_len, 2, pred_len),
                           use_decomp=False, use_experts=True, **base_kwargs)
    em = train_v11(train_loader, val_loader, test_loader, device=device,
                   horizon=pred_len, config=config, seq_len=seq_len, verbose=False, model=em)
    er = evaluate(em, test_loader, scaler, device)

    # TS-Booster
    set_seed(seed)
    bm = FlexibleTSBooster(base_model=LinearBackbone(seq_len, 2, pred_len),
                           use_decomp=True, use_experts=True,
                           expert_gate='snr', enrich_mode='trend',
                           **base_kwargs)
    bm = train_v11(train_loader, val_loader, test_loader, device=device,
                   horizon=pred_len, config=config, seq_len=seq_len, verbose=False, model=bm)
    br = evaluate(bm, test_loader, scaler, device)

    return er, br


def plot_expert_vs_tsb(results):
    """Bar chart: RMSE improvement (%) of TS-Booster over +Expert, with 5-seed error bars."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [ds for ds in ALL_DATASETS if ds in results]
    means = []
    stds = []
    for ds in datasets:
        gaps = results[ds]["gaps"]
        means.append(np.mean(gaps))
        stds.append(np.std(gaps))

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    colors = ['#4CAF50' if m > 0 else '#EF5350' for m in means]

    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor='white', width=0.6)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('RMSE Improvement (%)', fontsize=11)
    ax.set_title('+Expert → TS-Booster: Marginal Gain from Decomp + SNR Gate (5 seeds)', fontsize=13)

    # Value labels
    for i, (val, std) in enumerate(zip(means, stds)):
        y_pos = val + std + 0.15 if val >= 0 else val - std - 0.15
        va = 'bottom' if val >= 0 else 'top'
        ax.text(i, y_pos, f'{val:+.2f}%', ha='center', va=va, fontsize=8)

    # Win count
    wins = sum(1 for m in means if m > 0)
    ax.text(0.98, 0.95, f'TS-Booster wins: {wins}/{len(datasets)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.8))

    # Average line
    avg = np.mean(means)
    ax.axhline(y=avg, color='#1565C0', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(0.78, 0.88, f'avg: {avg:+.2f}%',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10, color='#1565C0')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "expert_vs_tsb.png", dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "expert_vs_tsb.pdf", bbox_inches='tight')
    plt.close()
    print("  Saved: expert_vs_tsb.png/pdf")


def analyze(results):
    """Print summary table and generate figure."""
    print(f"\n{'='*80}")
    print(f"ABLATION: +Expert vs TS-Booster ({len(SEEDS)} seeds)")
    print(f"{'='*80}")
    print(f"{'Dataset':<12} | {'Mean':>7} | {'Std':>6} | {'Min':>7} | {'Max':>7} | {'Wins':>5} | {'Result':>6}")
    print("-" * 70)

    total_wins = 0
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        gaps = results[ds]["gaps"]
        mean_g = np.mean(gaps)
        std_g = np.std(gaps)
        min_g = np.min(gaps)
        max_g = np.max(gaps)
        wins = sum(1 for g in gaps if g > 0)
        result = "WIN" if mean_g > 0 else "LOSS"
        if mean_g > 0: total_wins += 1
        print(f"  {ds:<12} | {mean_g:>+6.2f}% | {std_g:>5.2f}% | {min_g:>+6.2f}% | {max_g:>+6.2f}% | {wins}/{len(gaps)} | {result}")

    total = len([ds for ds in ALL_DATASETS if ds in results])
    print("-" * 70)
    print(f"  TS-Booster wins: {total_wins}/{total}")
    avg_mean = np.mean([np.mean(results[ds]["gaps"]) for ds in ALL_DATASETS if ds in results])
    print(f"  Overall avg gap: {avg_mean:+.2f}%")

    plot_expert_vs_tsb(results)


def main():
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # {dataset: {expert_rmse: [...], tsb_rmse: [...], gaps: [...]}}
    results = {}

    for ds in ALL_DATASETS:
        results[ds] = {"expert_rmse": [], "tsb_rmse": [], "gaps": []}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"Seed: {seed}")
        print(f"{'='*70}")
        for ds in ALL_DATASETS:
            er, br = run_pair(ds, seed)
            if er is not None:
                gap = (er - br) / er * 100
                results[ds]["expert_rmse"].append(er)
                results[ds]["tsb_rmse"].append(br)
                results[ds]["gaps"].append(gap)
                win = "WIN" if gap > 0 else "LOSS"
                print(f"  {ds:12s} | +Expert={er:.4f} TSB={br:.4f} gap={gap:+.2f}% {win}")

        # Save incrementally
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    analyze(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="Analyze saved results only")
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No results found at {RESULTS_PATH}. Run the experiment first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        analyze(results)
    else:
        main()
