"""
Run Cross-Backbone Validation on all 16 datasets.
For each dataset: test Linear, LSTM, GRU, Transformer as Vanilla vs TS-Booster.
Generates:
  1. Per-dataset bar chart figures
  2. Summary heatmap/table figure across all datasets
"""
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, test_loader, scaler, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["raw_input"].to(device)
            y = batch["target"].to(device)
            output = model(x)
            pred = output[0] if isinstance(output, tuple) else output
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    if scaler:
        preds = preds * scaler.scale_[0] + scaler.mean_[0]
        targets = targets * scaler.scale_[0] + scaler.mean_[0]
    return np.sqrt(mean_squared_error(targets.flatten(), preds.flatten()))


def run_cross_backbone(dataset_name, device):
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
        # Vanilla
        set_seed(42)
        vanilla = bb_fn(seq_len, 2, pred_len)
        vanilla = train_v11(train_loader, val_loader, test_loader,
                           device=device, horizon=pred_len, config=config,
                           seq_len=seq_len, verbose=False, model=vanilla)
        v_rmse = evaluate(vanilla, test_loader, scaler, device)

        # TS-Booster
        set_seed(42)
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
        b_rmse = evaluate(boosted, test_loader, scaler, device)

        gap = (v_rmse - b_rmse) / v_rmse * 100
        results[bb_name] = {
            'vanilla_rmse': float(v_rmse),
            'boosted_rmse': float(b_rmse),
            'improvement': float(gap),
        }

        win = "WIN" if gap > 0 else "LOSS"
        print(f"  {dataset_name:12s} | {bb_name:12s} | V: {v_rmse:.4f} | B: {b_rmse:.4f} | {gap:+6.2f}% {win}")

    return results


def plot_per_dataset(dataset_name, results, output_dir):
    """Bar chart: Vanilla vs TS-Booster for each backbone."""
    import matplotlib.pyplot as plt

    backbones = list(results.keys())
    vanilla_rmse = [results[b]['vanilla_rmse'] for b in backbones]
    boosted_rmse = [results[b]['boosted_rmse'] for b in backbones]
    improvements = [results[b]['improvement'] for b in backbones]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: RMSE comparison
    x = np.arange(len(backbones))
    width = 0.35
    bars1 = ax1.bar(x - width/2, vanilla_rmse, width, label='Vanilla',
                    color='#90CAF9', edgecolor='#1565C0', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, boosted_rmse, width, label='+ TS-Booster',
                    color='#EF9A9A', edgecolor='#C62828', linewidth=1.2)

    ax1.set_xlabel('Backbone', fontsize=11)
    ax1.set_ylabel('RMSE', fontsize=11)
    ax1.set_title(f'{dataset_name} — RMSE Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(backbones)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01*max(vanilla_rmse),
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.01*max(vanilla_rmse),
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

    # Right: Improvement %
    colors = ['#66BB6A' if imp > 0 else '#EF5350' for imp in improvements]
    bars3 = ax2.bar(x, improvements, 0.6, color=colors, edgecolor='#37474F', linewidth=1.0)
    ax2.axhline(y=0, color='#9E9E9E', linewidth=1, linestyle='--')
    ax2.set_xlabel('Backbone', fontsize=11)
    ax2.set_ylabel('RMSE Improvement (%)', fontsize=11)
    ax2.set_title(f'{dataset_name} — TS-Booster Improvement', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(backbones)
    ax2.grid(axis='y', alpha=0.3)

    for bar, imp in zip(bars3, improvements):
        h = bar.get_height()
        offset = 0.3 if h >= 0 else -0.5
        ax2.text(bar.get_x() + bar.get_width()/2, h + offset,
                f'{imp:+.1f}%', ha='center', va='bottom' if h >= 0 else 'top',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_path = output_dir / f'cross_backbone_{dataset_name}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_summary(all_results, output_dir):
    """Summary heatmap + table across all datasets and backbones."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    datasets = list(all_results.keys())
    backbones = list(BACKBONES.keys())

    # Build improvement matrix
    matrix = np.zeros((len(datasets), len(backbones)))
    for i, ds in enumerate(datasets):
        for j, bb in enumerate(backbones):
            if ds in all_results and bb in all_results[ds]:
                matrix[i, j] = all_results[ds][bb]['improvement']

    # === Figure 1: Heatmap ===
    fig, ax = plt.subplots(figsize=(10, 12))

    # Custom colormap: red for negative, white for zero, green for positive
    cmap = mcolors.LinearSegmentedColormap.from_list('rg',
        ['#EF5350', '#FFCDD2', '#FFFFFF', '#C8E6C9', '#43A047'], N=256)

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(backbones)))
    ax.set_xticklabels(backbones, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10)

    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(backbones)):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                   fontsize=9, fontweight='bold', color=color)

    # Add win count per backbone (bottom)
    for j, bb in enumerate(backbones):
        wins = sum(1 for i in range(len(datasets)) if matrix[i, j] > 0)
        ax.text(j, len(datasets) + 0.3, f'{wins}/{len(datasets)}',
               ha='center', va='center', fontsize=11, fontweight='bold',
               color='#1B5E20' if wins > len(datasets)/2 else '#B71C1C')
    ax.text(-0.8, len(datasets) + 0.3, 'Wins:', ha='right', va='center',
           fontsize=10, fontweight='bold', color='#424242')

    # Add avg improvement per backbone
    for j, bb in enumerate(backbones):
        avg = matrix[:, j].mean()
        ax.text(j, len(datasets) + 0.8, f'{avg:+.1f}%',
               ha='center', va='center', fontsize=10,
               color='#1B5E20' if avg > 0 else '#B71C1C')
    ax.text(-0.8, len(datasets) + 0.8, 'Avg:', ha='right', va='center',
           fontsize=10, fontweight='bold', color='#424242')

    # Add win count per dataset (right side)
    for i, ds in enumerate(datasets):
        wins = sum(1 for j in range(len(backbones)) if matrix[i, j] > 0)
        color = '#1B5E20' if wins >= 3 else '#E65100' if wins >= 2 else '#B71C1C'
        ax.text(len(backbones) + 0.2, i, f'{wins}/4', ha='left', va='center',
               fontsize=9, fontweight='bold', color=color)

    ax.set_title('Cross-Backbone Validation: TS-Booster RMSE Improvement (%)\n'
                'Vanilla → + TS-Booster (per backbone, per dataset)',
                fontsize=13, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.12)
    cbar.set_label('RMSE Improvement (%)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_backbone_summary_heatmap.png',
               dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'cross_backbone_summary_heatmap.pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved summary heatmap")

    # === Figure 2: Grouped bar chart ===
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(datasets))
    width = 0.2
    colors = ['#42A5F5', '#66BB6A', '#FFA726', '#AB47BC']

    for j, (bb, color) in enumerate(zip(backbones, colors)):
        vals = [matrix[i, j] for i in range(len(datasets))]
        offset = (j - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=bb, color=color,
                     edgecolor='#37474F', linewidth=0.8, alpha=0.85)

    ax.axhline(y=0, color='#9E9E9E', linewidth=1, linestyle='--')
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('RMSE Improvement (%)', fontsize=11)
    ax.set_title('Cross-Backbone: TS-Booster Improvement by Dataset & Backbone',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_backbone_summary_bars.png',
               dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'cross_backbone_summary_bars.pdf',
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved summary bar chart")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path('experiments/comprehensive_results/cross_backbone/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for i, ds in enumerate(ALL_DATASETS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(ALL_DATASETS)}] {ds}")
        print(f"{'='*70}")

        results = run_cross_backbone(ds, device)
        if results:
            all_results[ds] = results
            plot_per_dataset(ds, results, output_dir)
            print(f"  Saved figure: cross_backbone_{ds}.png")

    # Save raw results
    with open(output_dir.parent / 'cross_backbone_all_results.json', 'w') as f:  # saves to cross_backbone/
        json.dump(all_results, f, indent=2)
    print("\nSaved raw results to cross_backbone_all_results.json")

    # Print summary
    print(f"\n\n{'='*80}")
    print("CROSS-BACKBONE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':12s} | {'Linear':>10s} | {'LSTM':>10s} | {'GRU':>10s} | {'Transf.':>10s}")
    print("-" * 65)

    bb_names = list(BACKBONES.keys())
    bb_wins = {bb: 0 for bb in bb_names}
    bb_gaps = {bb: [] for bb in bb_names}

    for ds in ALL_DATASETS:
        if ds not in all_results:
            continue
        row = f"{ds:12s}"
        for bb in bb_names:
            if bb in all_results[ds]:
                gap = all_results[ds][bb]['improvement']
                w = "W" if gap > 0 else "L"
                row += f" | {gap:+7.2f}% {w}"
                if gap > 0:
                    bb_wins[bb] += 1
                bb_gaps[bb].append(gap)
            else:
                row += f" | {'?':>10s}"
        print(row)

    print("-" * 65)
    for bb in bb_names:
        total = len(bb_gaps[bb])
        avg = np.mean(bb_gaps[bb]) if bb_gaps[bb] else 0
        print(f"  {bb:12s}: {bb_wins[bb]}/{total} wins, avg: {avg:+.2f}%")

    # Generate summary plots
    plot_summary(all_results, output_dir)
    print("\nDone!")
