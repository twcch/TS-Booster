"""
Run Cross-Backbone Validation on all 16 datasets.
For each dataset: test Linear, LSTM, GRU, Transformer as Vanilla vs TS-Booster.
Runs 5 seeds per configuration for statistical robustness.

Generates:
  1. Per-dataset bar chart figures
  2. Summary heatmap/table figure across all datasets

Usage:
  uv run python experiments/run_cross_backbone_all.py              # Full run (5 seeds)
  uv run python experiments/run_cross_backbone_all.py --analyze    # Tables/figures from saved JSON
"""
import sys
import json
import random
import argparse
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.layers import LinearBackbone, LSTMBackbone, GRUBackbone, TransformerBackbone
from src.models.ts_booster import TS_Booster
from src.engine.trainer import train_v11
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data
from sklearn.metrics import mean_squared_error

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

BACKBONES = {
    'Linear': lambda sl, ic, pl: LinearBackbone(sl, ic, pl),
    'LSTM': lambda sl, ic, pl: LSTMBackbone(sl, ic, pl, hidden_dim=64, num_layers=2),
    'GRU': lambda sl, ic, pl: GRUBackbone(sl, ic, pl, hidden_dim=64, num_layers=2),
    'Transformer': lambda sl, ic, pl: TransformerBackbone(sl, ic, pl, hidden_dim=64, num_layers=2),
}

SEEDS = [42, 123, 456, 789, 2024]
RESULTS_PATH = Path('experiments/comprehensive_results/cross_backbone/cross_backbone_all_results.json')
FIGURES_DIR = Path('experiments/comprehensive_results/cross_backbone/figures')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, test_loader, scaler, device):
    """Return dict with rmse, mae, mda."""
    model.eval()
    preds, targets, last_obs_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["raw_input"].to(device)
            y = batch["target"].to(device)
            output = model(x)
            pred = output[0] if isinstance(output, tuple) else output
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            last_obs_list.append(x[:, -1, 0].cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    last_obs = np.concatenate(last_obs_list)
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    if targets.ndim == 3:
        targets = targets.squeeze(-1)
    if scaler:
        preds = preds * scaler.scale_[0] + scaler.mean_[0]
        targets = targets * scaler.scale_[0] + scaler.mean_[0]
        last_obs = last_obs * scaler.scale_[0] + scaler.mean_[0]
    rmse = float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))
    mae = float(np.mean(np.abs(targets.flatten() - preds.flatten())))
    pred_dir = np.sign(preds - last_obs[:, None])
    actual_dir = np.sign(targets - last_obs[:, None])
    mda = float(np.mean(pred_dir == actual_dir))
    return {"rmse": rmse, "mae": mae, "mda": mda}


def _get_mean(val):
    """Get mean from a list or return scalar (backward compat with old format)."""
    if isinstance(val, list):
        return float(np.mean(val))
    return float(val)


def _is_complete(results, ds):
    """Check if a dataset has complete 5-seed results with all metrics for all backbones."""
    if ds not in results:
        return False
    for bb in BACKBONES:
        if bb not in results[ds]:
            return False
        entry = results[ds][bb]
        # Check all 3 metrics exist with enough seeds
        for metric in ['rmse', 'mae', 'mda']:
            v = entry.get(f'vanilla_{metric}')
            b = entry.get(f'boosted_{metric}')
            if v is None or b is None:
                return False
            if not (isinstance(v, list) and len(v) >= len(SEEDS) and isinstance(b, list) and len(b) >= len(SEEDS)):
                return False
    return True


def run_cross_backbone(dataset_name, device):
    """Test all 4 backbones × 5 seeds on a single dataset."""
    config = BEST_STAGE3_CONFIG.copy()
    config['epochs'] = 100
    config['input_channels'] = 2
    seq_len, pred_len = 30, 3

    path = f"dataset/{dataset_name}.csv"
    if not Path(path).exists():
        return None

    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=seq_len, horizon=pred_len, batch_size=32
    )

    results = {}

    for bb_name, bb_fn in BACKBONES.items():
        vanilla_metrics = {"rmse": [], "mae": [], "mda": []}
        boosted_metrics = {"rmse": [], "mae": [], "mda": []}

        t0 = time.time()

        for seed in SEEDS:
            # Vanilla
            set_seed(seed)
            vanilla = bb_fn(seq_len, 2, pred_len)
            vanilla = train_v11(train_loader, val_loader, test_loader,
                               device=device, horizon=pred_len, config=config,
                               seq_len=seq_len, verbose=False, model=vanilla)
            v_metrics = evaluate(vanilla, test_loader, scaler, device)
            for k in vanilla_metrics:
                vanilla_metrics[k].append(v_metrics[k])

            # TS-Booster
            set_seed(seed)
            boosted = TS_Booster(
                base_model=bb_fn(seq_len, 2, pred_len),
                seq_len=seq_len, pred_len=pred_len, input_channels=2,
                decomp_kernels=config.get('decomp_kernels', [7, 13]),
                expert_kernels=config.get('expert_kernels', [3, 5, 7]),
                expert_hidden_dim=config.get('expert_hidden_dim', 32),
                dropout_p=config.get('dropout', 0.3),
                n_boost_rounds=config.get('n_boost_rounds', 2),
                expert_type=config.get('expert_type', 'cnn'),
                expert_gate=config.get('expert_gate', 'snr'),
                enrich_mode=config.get('enrich_mode', 'trend'),
            )
            boosted = train_v11(train_loader, val_loader, test_loader,
                               device=device, horizon=pred_len, config=config,
                               seq_len=seq_len, verbose=False, model=boosted)
            b_metrics = evaluate(boosted, test_loader, scaler, device)
            for k in boosted_metrics:
                boosted_metrics[k].append(b_metrics[k])

            # Free GPU memory
            del vanilla, boosted
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        v_mean = np.mean(vanilla_metrics["rmse"])
        b_mean = np.mean(boosted_metrics["rmse"])
        gap = (v_mean - b_mean) / v_mean * 100
        win = "WIN" if gap > 0 else "LOSS"

        results[bb_name] = {
            'vanilla_rmse': vanilla_metrics["rmse"],
            'boosted_rmse': boosted_metrics["rmse"],
            'vanilla_mae': vanilla_metrics["mae"],
            'boosted_mae': boosted_metrics["mae"],
            'vanilla_mda': vanilla_metrics["mda"],
            'boosted_mda': boosted_metrics["mda"],
        }

        print(f"  {dataset_name:12s} | {bb_name:12s} | "
              f"V: {v_mean:.4f}±{np.std(vanilla_metrics['rmse']):.4f} | "
              f"B: {b_mean:.4f}±{np.std(boosted_metrics['rmse']):.4f} | "
              f"{gap:+6.2f}% {win}  ({elapsed:.0f}s)")

    return results


def analyze(all_results):
    """Print terminal table and summary statistics."""
    bb_names = list(BACKBONES.keys())

    print(f"\n\n{'='*100}")
    print("CROSS-BACKBONE SUMMARY (5-seed mean)")
    print(f"{'='*100}")
    print(f"{'Dataset':12s} | {'Linear':>14s} | {'LSTM':>14s} | {'GRU':>14s} | {'Transf.':>14s} | Wins")
    print("-" * 90)

    bb_wins = {bb: 0 for bb in bb_names}
    bb_gaps = {bb: [] for bb in bb_names}
    ds_wins = {}

    for ds in ALL_DATASETS:
        if ds not in all_results:
            continue
        row = f"{ds:12s}"
        wins = 0
        for bb in bb_names:
            if bb in all_results[ds]:
                entry = all_results[ds][bb]
                v_mean = _get_mean(entry['vanilla_rmse'])
                b_mean = _get_mean(entry['boosted_rmse'])
                gap = (v_mean - b_mean) / v_mean * 100
                w = "W" if gap > 0 else "L"
                row += f" | {gap:+7.2f}% {w:>3s}"
                if gap > 0:
                    bb_wins[bb] += 1
                    wins += 1
                bb_gaps[bb].append(gap)
            else:
                row += f" | {'?':>14s}"
        row += f" | {wins}/4"
        ds_wins[ds] = wins
        print(row)

    print("-" * 90)
    total_all = sum(bb_wins.values())
    total_possible = sum(len(bb_gaps[bb]) for bb in bb_names)
    for bb in bb_names:
        total = len(bb_gaps[bb])
        avg = np.mean(bb_gaps[bb]) if bb_gaps[bb] else 0
        print(f"  {bb:12s}: {bb_wins[bb]}/{total} wins, avg improvement: {avg:+.2f}%")
    print(f"\n  TOTAL: {total_all}/{total_possible} wins ({total_all/total_possible*100:.1f}%)")

    # LaTeX table
    print(f"\n{'='*100}")
    print("LATEX TABLE")
    print(f"{'='*100}")
    print()
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Cross-backbone validation: RMSE improvement (\%) of TS-Booster over Vanilla (5-seed mean).}")
    print(r"\label{tab:cross_backbone}")
    print(r"\begin{tabular}{l" + "r" * len(bb_names) + "}")
    print(r"\toprule")
    print("Dataset & " + " & ".join(bb_names) + r" \\")
    print(r"\midrule")

    for ds in ALL_DATASETS:
        if ds not in all_results:
            continue
        ds_latex = ds.replace("_", r"\_")
        cells = []
        for bb in bb_names:
            if bb in all_results[ds]:
                entry = all_results[ds][bb]
                v_mean = _get_mean(entry['vanilla_rmse'])
                b_mean = _get_mean(entry['boosted_rmse'])
                gap = (v_mean - b_mean) / v_mean * 100
                if gap > 0:
                    cells.append(f"+{gap:.1f}\\%")
                else:
                    cells.append(f"\\cellcolor{{red!15}}{gap:.1f}\\%")
            else:
                cells.append("--")
        print(f"{ds_latex} & " + " & ".join(cells) + r" \\")

    # Summary rows
    print(r"\midrule")
    win_cells = []
    avg_cells = []
    for bb in bb_names:
        total = len(bb_gaps[bb])
        win_cells.append(f"{bb_wins[bb]}/{total}")
        avg = np.mean(bb_gaps[bb]) if bb_gaps[bb] else 0
        avg_cells.append(f"+{avg:.1f}\\%" if avg > 0 else f"{avg:.1f}\\%")
    print(r"\textit{Wins} & " + " & ".join(win_cells) + r" \\")
    print(r"\textit{Avg. Impr.} & " + " & ".join(avg_cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def plot_per_dataset(all_results, output_dir):
    """Per-dataset bar chart: Vanilla vs TS-Booster for each backbone (5-seed mean ± std)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    for ds in ALL_DATASETS:
        if ds not in all_results:
            continue

        backbones = [bb for bb in BACKBONES if bb in all_results[ds]]
        if not backbones:
            continue

        v_means, v_stds, b_means, b_stds = [], [], [], []
        for bb in backbones:
            entry = all_results[ds][bb]
            v = entry['vanilla_rmse']
            b = entry['boosted_rmse']
            v_arr = v if isinstance(v, list) else [v]
            b_arr = b if isinstance(b, list) else [b]
            v_means.append(np.mean(v_arr))
            v_stds.append(np.std(v_arr))
            b_means.append(np.mean(b_arr))
            b_stds.append(np.std(b_arr))

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(backbones))
        width = 0.35

        ax.bar(x - width/2, v_means, width, yerr=v_stds, capsize=4,
               label='Vanilla', color='#90CAF9', edgecolor='#37474F', linewidth=0.8)
        ax.bar(x + width/2, b_means, width, yerr=b_stds, capsize=4,
               label='+ TS-Booster', color='#A5D6A7', edgecolor='#37474F', linewidth=0.8)

        # Improvement labels
        for i, (vm, bm) in enumerate(zip(v_means, b_means)):
            gap = (vm - bm) / vm * 100
            color = '#1B5E20' if gap > 0 else '#B71C1C'
            ax.text(i + width/2, bm + b_stds[i] + 0.001, f'{gap:+.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

        ax.set_xlabel('Backbone', fontsize=11)
        ax.set_ylabel('Test RMSE', fontsize=11)
        ax.set_title(f'{ds}: Vanilla vs TS-Booster (5-seed mean ± std)',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(backbones, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'cross_backbone_{ds}.png',
                    dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"  Saved: {len([ds for ds in ALL_DATASETS if ds in all_results])} per-dataset bar charts")


def _build_improvement_matrix(all_results, metric, higher_is_better=False):
    """Build improvement % matrix for a given metric across datasets × backbones."""
    datasets = [ds for ds in ALL_DATASETS if ds in all_results]
    backbones = list(BACKBONES.keys())
    matrix = np.zeros((len(datasets), len(backbones)))
    for i, ds in enumerate(datasets):
        for j, bb in enumerate(backbones):
            if bb in all_results[ds]:
                entry = all_results[ds][bb]
                v_mean = _get_mean(entry[f'vanilla_{metric}'])
                b_mean = _get_mean(entry[f'boosted_{metric}'])
                if higher_is_better:
                    matrix[i, j] = (b_mean - v_mean) / v_mean * 100
                else:
                    matrix[i, j] = (v_mean - b_mean) / v_mean * 100
    return matrix, datasets, backbones


def _plot_heatmap(matrix, datasets, backbones, metric, output_dir):
    """Plot a single heatmap for a given metric."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(10, 12))
    cmap = mcolors.LinearSegmentedColormap.from_list('rg',
        ['#EF5350', '#FFCDD2', '#FFFFFF', '#C8E6C9', '#43A047'], N=256)

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(backbones)))
    ax.set_xticklabels(backbones, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10)

    for i in range(len(datasets)):
        for j in range(len(backbones)):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                   fontsize=9, fontweight='bold', color=color)

    for j, bb in enumerate(backbones):
        wins = sum(1 for i in range(len(datasets)) if matrix[i, j] > 0)
        ax.text(j, len(datasets) + 0.3, f'{wins}/{len(datasets)}',
               ha='center', va='center', fontsize=11, fontweight='bold',
               color='#1B5E20' if wins > len(datasets)/2 else '#B71C1C')
    ax.text(-0.8, len(datasets) + 0.3, 'Wins:', ha='right', va='center',
           fontsize=10, fontweight='bold', color='#424242')

    for j, bb in enumerate(backbones):
        avg = matrix[:, j].mean()
        ax.text(j, len(datasets) + 0.8, f'{avg:+.1f}%',
               ha='center', va='center', fontsize=10,
               color='#1B5E20' if avg > 0 else '#B71C1C')
    ax.text(-0.8, len(datasets) + 0.8, 'Avg:', ha='right', va='center',
           fontsize=10, fontweight='bold', color='#424242')

    for i, ds in enumerate(datasets):
        wins = sum(1 for j in range(len(backbones)) if matrix[i, j] > 0)
        color = '#1B5E20' if wins >= 3 else '#E65100' if wins >= 2 else '#B71C1C'
        ax.text(len(backbones) + 0.2, i, f'{wins}/4', ha='left', va='center',
               fontsize=9, fontweight='bold', color=color)

    metric_upper = metric.upper()
    ax.set_title(f'Cross-Backbone Validation: TS-Booster {metric_upper} Improvement (%)\n'
                f'Vanilla → + TS-Booster (5-seed mean, per backbone, per dataset)',
                fontsize=13, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.12)
    cbar.set_label(f'{metric_upper} Improvement (%)', fontsize=10)

    plt.tight_layout()
    fname = f'cross_backbone_summary_heatmap_{metric}.png'
    plt.savefig(output_dir / fname, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fname}")


def _plot_bars(matrix, datasets, backbones, metric, output_dir):
    """Plot a grouped bar chart for a given metric."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.2
    colors = ['#42A5F5', '#66BB6A', '#FFA726', '#AB47BC']

    # Find best backbone per dataset (highest improvement)
    best_per_ds = {}
    for i in range(len(datasets)):
        best_j = int(np.argmax(matrix[i, :]))
        best_per_ds[i] = best_j

    bar_containers = []
    for j, (bb, color) in enumerate(zip(backbones, colors)):
        vals = [matrix[i, j] for i in range(len(datasets))]
        offset = (j - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=bb, color=color,
               edgecolor='#37474F', linewidth=0.8)
        bar_containers.append((j, bars))

    # Dim non-best bars
    for j, bars in bar_containers:
        for i, bar in enumerate(bars):
            if best_per_ds.get(i) != j:
                bar.set_alpha(0.35)

    metric_upper = metric.upper()
    ax.axhline(y=0, color='#9E9E9E', linewidth=1, linestyle='--')
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel(f'{metric_upper} Improvement (%)', fontsize=11)
    ax.set_title(f'Cross-Backbone: TS-Booster {metric_upper} Improvement by Dataset & Backbone (5-seed mean)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = f'cross_backbone_summary_bars_{metric}.png'
    plt.savefig(output_dir / fname, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fname}")


def plot_summary(all_results, output_dir):
    """Summary heatmaps + grouped bar charts for RMSE, MAE, MDA."""
    import matplotlib
    matplotlib.use('Agg')

    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, higher_is_better in [('rmse', False), ('mae', False), ('mda', True)]:
        # Check if metric data exists
        ds0 = next((ds for ds in ALL_DATASETS if ds in all_results), None)
        if ds0 is None:
            continue
        bb0 = next(iter(all_results[ds0]))
        if f'vanilla_{metric}' not in all_results[ds0][bb0]:
            print(f"  SKIP {metric} (no data)")
            continue

        matrix, datasets, backbones = _build_improvement_matrix(all_results, metric, higher_is_better)
        _plot_heatmap(matrix, datasets, backbones, metric, output_dir)
        _plot_bars(matrix, datasets, backbones, metric, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-backbone validation for TS-Booster paper")
    parser.add_argument("--analyze", action="store_true", help="Only analyze saved results")
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No saved results at {RESULTS_PATH}. Run without --analyze first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        analyze(all_results)
        print("\nGenerating figures...")
        plot_per_dataset(all_results, FIGURES_DIR)
        plot_summary(all_results, FIGURES_DIR)
        print("Done!")
        sys.exit(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")

    # Load existing results for resume
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded existing results ({len(all_results)} datasets)")
    else:
        all_results = {}

    for i, ds in enumerate(ALL_DATASETS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(ALL_DATASETS)}] {ds}")
        print(f"{'='*70}")

        # Skip if already complete
        if _is_complete(all_results, ds):
            print(f"  SKIP (already has {len(SEEDS)}-seed results)")
            continue

        results = run_cross_backbone(ds, device)
        if results:
            all_results[ds] = results

            # Incremental save
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved ({len(all_results)}/{len(ALL_DATASETS)} datasets)")

    analyze(all_results)
    print("\nGenerating figures...")
    plot_per_dataset(all_results, FIGURES_DIR)
    plot_summary(all_results, FIGURES_DIR)
    print("\nDone!")
