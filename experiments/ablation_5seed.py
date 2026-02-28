"""
Full 4-config ablation with 5 seeds: Vanilla → +Decomp → +Expert → TS-Booster.

Reuses existing 5-seed results where available:
  - Vanilla / TS-Booster RMSE: from statistical_test_results.json
  - +Expert RMSE: from multi_seed_expert_vs_tsb.json
  - +Decomp: runs fresh (only config without 5-seed data)

Output: ablation/ablation_5seed.json  (unified, all 4 configs × 16 datasets × 5 seeds)

Usage:
  uv run python experiments/ablation_5seed.py              # Run +Decomp, merge all
  uv run python experiments/ablation_5seed.py --analyze    # Figures/tables from saved JSON
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

FX_DATASETS = ALL_DATASETS[:10]
STOCK_DATASETS = ALL_DATASETS[10:]
CONFIGS = ["Vanilla", "+Decomp", "+Expert", "TS-Booster"]
SEEDS = [42, 123, 456, 789, 2024]

RESULTS_DIR = Path("experiments/comprehensive_results/ablation")
RESULTS_PATH = RESULTS_DIR / "ablation_5seed.json"
FIGURES_DIR = RESULTS_DIR / "figures"

# Paths to reuse existing 5-seed data
MAIN_RESULTS = Path("experiments/comprehensive_results/main_experiment/statistical_test_results.json")
EXPERT_RESULTS = RESULTS_DIR / "multi_seed_expert_vs_tsb.json"


def set_seed(seed):
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


def load_existing_results():
    """Load existing 5-seed Vanilla, TS-Booster, and +Expert results."""
    results = {}

    # Vanilla + TS-Booster from main experiment
    if MAIN_RESULTS.exists():
        with open(MAIN_RESULTS) as f:
            main = json.load(f)
        for ds in ALL_DATASETS:
            if ds in main:
                results.setdefault(ds, {})
                v = main[ds]["vanilla"]
                b = main[ds]["booster"]
                results[ds]["Vanilla"] = {"rmse": v["rmse"] if isinstance(v, dict) else v}
                results[ds]["TS-Booster"] = {"rmse": b["rmse"] if isinstance(b, dict) else b}
        print(f"  Loaded Vanilla + TS-Booster from {MAIN_RESULTS} ({len(results)} datasets)")

    # +Expert from multi_seed_test
    if EXPERT_RESULTS.exists():
        with open(EXPERT_RESULTS) as f:
            expert = json.load(f)
        for ds in ALL_DATASETS:
            if ds in expert:
                results.setdefault(ds, {})
                results[ds]["+Expert"] = {"rmse": expert[ds]["expert_rmse"]}
        print(f"  Loaded +Expert from {EXPERT_RESULTS}")

    return results


def run_decomp(ds, seed, device):
    """Run +Decomp config (decomp + dual backbone, no expert) for one dataset and seed."""
    config = BEST_STAGE3_CONFIG.copy()
    config['epochs'] = 100
    config['input_channels'] = 2
    seq_len, pred_len = 30, 3

    path = f"dataset/{ds}.csv"
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=seq_len, horizon=pred_len, batch_size=32
    )

    set_seed(seed)
    model = FlexibleTSBooster(
        base_model=LinearBackbone(seq_len, 2, pred_len),
        seq_len=seq_len, pred_len=pred_len, input_channels=2,
        use_decomp=True, use_experts=False,
        decomp_kernels=[7, 13],
        expert_kernels=config.get('expert_kernels', [3, 5, 7]),
        expert_hidden_dim=config.get('expert_hidden_dim', 32),
        dropout_p=config.get('dropout', 0.3),
    )
    model = train_v11(train_loader, val_loader, test_loader,
                      device=device, horizon=pred_len, config=config,
                      seq_len=seq_len, verbose=False, model=model)
    return evaluate(model, test_loader, scaler, device)


def run_experiment():
    """Run +Decomp 5-seed, merge with existing data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}\n")

    print("Loading existing 5-seed results...")
    results = load_existing_results()

    # Check if +Decomp already has 5-seed data (for resume)
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            saved = json.load(f)
        for ds in ALL_DATASETS:
            if ds in saved and "+Decomp" in saved[ds]:
                rmse_list = saved[ds]["+Decomp"]["rmse"]
                if isinstance(rmse_list, list) and len(rmse_list) >= len(SEEDS):
                    results.setdefault(ds, {})["+Decomp"] = saved[ds]["+Decomp"]
        n_done = sum(1 for ds in ALL_DATASETS
                     if ds in results and "+Decomp" in results.get(ds, {})
                     and len(results[ds]["+Decomp"].get("rmse", [])) >= len(SEEDS))
        if n_done > 0:
            print(f"  Resumed +Decomp: {n_done}/{len(ALL_DATASETS)} already complete")

    print(f"\nRunning +Decomp (5 seeds × 16 datasets)...")
    print(f"{'Dataset':12s} | {'Seed':>5} | {'RMSE':>8} | Status")
    print("-" * 50)

    for ds in ALL_DATASETS:
        results.setdefault(ds, {})

        # Skip if already complete
        if "+Decomp" in results[ds]:
            existing = results[ds]["+Decomp"].get("rmse", [])
            if isinstance(existing, list) and len(existing) >= len(SEEDS):
                print(f"  {ds:12s} | SKIP  (already has {len(existing)} seeds)")
                continue

        decomp_rmses = []
        t0 = time.time()
        for seed in SEEDS:
            rmse = run_decomp(ds, seed, device)
            if rmse is not None:
                decomp_rmses.append(float(rmse))
                print(f"  {ds:12s} | {seed:>5} | {rmse:.4f}  |")

        elapsed = time.time() - t0
        if decomp_rmses:
            results[ds]["+Decomp"] = {"rmse": decomp_rmses}
            print(f"  {ds:12s} | mean={np.mean(decomp_rmses):.4f} ± {np.std(decomp_rmses):.4f}  ({elapsed:.0f}s)")

        # Incremental save
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    return results


def analyze(results):
    """Print terminal table with 5-seed mean ± std."""
    print(f"\n{'='*100}")
    print("ABLATION: 4-Config Progression (5-seed mean ± std)")
    print(f"{'='*100}")

    header = f"{'Dataset':12s}"
    for cfg in CONFIGS:
        header += f" | {cfg:>16s}"
    header += " | Best"
    print(header)
    print("-" * len(header))

    cfg_improvements = {cfg: [] for cfg in CONFIGS[1:]}  # skip Vanilla

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        row = f"  {ds:12s}"
        v_mean = np.mean(results[ds]["Vanilla"]["rmse"])

        rmse_means = {}
        for cfg in CONFIGS:
            if cfg in results[ds]:
                arr = results[ds][cfg]["rmse"]
                m = np.mean(arr)
                s = np.std(arr)
                rmse_means[cfg] = m
                row += f" | {m:>7.4f}±{s:.4f}"

                if cfg != "Vanilla":
                    imp = (v_mean - m) / v_mean * 100
                    cfg_improvements[cfg].append(imp)
            else:
                row += f" | {'N/A':>16s}"

        # Best config
        if rmse_means:
            best = min(rmse_means, key=lambda k: rmse_means[k])
            row += f" | {best}"
        print(row)

    print("-" * len(header))

    # Summary
    for cfg in CONFIGS[1:]:
        if cfg_improvements[cfg]:
            avg = np.mean(cfg_improvements[cfg])
            wins = sum(1 for x in cfg_improvements[cfg] if x > 0)
            total = len(cfg_improvements[cfg])
            print(f"  {cfg:12s}: avg improvement {avg:+.2f}%, wins {wins}/{total}")

    # LaTeX table
    print(f"\n{'='*100}")
    print("LATEX TABLE")
    print(f"{'='*100}")
    print()
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Ablation study: progressive component addition (5-seed mean RMSE).}")
    print(r"\label{tab:ablation}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l" + "r" * len(CONFIGS) + "}")
    print(r"\toprule")
    print("Dataset & " + " & ".join(CONFIGS) + r" \\")
    print(r"\midrule")

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        ds_latex = ds.replace("_", r"\_")
        rmse_means = {}
        for cfg in CONFIGS:
            if cfg in results[ds]:
                rmse_means[cfg] = np.mean(results[ds][cfg]["rmse"])

        best_rmse = min(rmse_means.values()) if rmse_means else None
        cells = []
        for cfg in CONFIGS:
            if cfg in rmse_means:
                val = rmse_means[cfg]
                if best_rmse is not None and abs(val - best_rmse) < 1e-6:
                    cells.append(r"\textbf{" + f"{val:.4f}" + "}")
                else:
                    cells.append(f"{val:.4f}")
            else:
                cells.append("--")
        print(f"{ds_latex} & " + " & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")


def plot_figures(results):
    """Generate ablation figures from 5-seed data."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [ds for ds in ALL_DATASETS if ds in results]

    # === Figure: Improvement Heatmap ===
    configs_plot = ["+Decomp", "+Expert", "TS-Booster"]
    matrix = np.zeros((len(datasets), len(configs_plot)))
    for i, ds in enumerate(datasets):
        v_mean = np.mean(results[ds]["Vanilla"]["rmse"])
        for j, cfg in enumerate(configs_plot):
            if cfg in results[ds]:
                c_mean = np.mean(results[ds][cfg]["rmse"])
                matrix[i, j] = (v_mean - c_mean) / v_mean * 100

    cmap = mcolors.LinearSegmentedColormap.from_list('rg',
        ['#EF5350', '#FFCDD2', '#FFFFFF', '#C8E6C9', '#43A047'], N=256)

    fig, ax = plt.subplots(figsize=(8, 10))
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(configs_plot)))
    ax.set_xticklabels(configs_plot, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10)

    # FX / Stocks separator
    n_fx = len([d for d in datasets if d in FX_DATASETS])
    ax.axhline(y=n_fx - 0.5, color='#424242', linewidth=2, linestyle='-')
    n_stock = len(datasets) - n_fx
    ax.annotate('FX', xy=(-0.18, 1 - (n_fx / 2) / len(datasets)),
                xycoords='axes fraction', fontsize=11, fontweight='bold',
                color='#1565C0', ha='center', va='center', rotation=90)
    ax.annotate('Stocks', xy=(-0.18, 1 - (n_fx + n_stock / 2) / len(datasets)),
                xycoords='axes fraction', fontsize=11, fontweight='bold',
                color='#C62828', ha='center', va='center', rotation=90)

    for i in range(len(datasets)):
        for j in range(len(configs_plot)):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    # Win count per config (bottom)
    for j, cfg in enumerate(configs_plot):
        wins = sum(1 for i in range(len(datasets)) if matrix[i, j] > 0)
        ax.text(j, len(datasets) + 0.3, f'{wins}/{len(datasets)}',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='#1B5E20' if wins > len(datasets) / 2 else '#B71C1C')
    ax.text(-0.5, len(datasets) + 0.3, 'Wins:', ha='right', va='center',
            fontsize=10, fontweight='bold', color='#424242')

    # Avg improvement
    for j, cfg in enumerate(configs_plot):
        avg = matrix[:, j].mean()
        ax.text(j, len(datasets) + 0.8, f'{avg:+.1f}%',
                ha='center', va='center', fontsize=10,
                color='#1B5E20' if avg > 0 else '#B71C1C')
    ax.text(-0.5, len(datasets) + 0.8, 'Avg:', ha='right', va='center',
            fontsize=10, fontweight='bold', color='#424242')

    ax.set_title('Ablation Study: RMSE Improvement over Vanilla (%)\n'
                 'Progressive Component Addition (5-seed mean)',
                 fontsize=13, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label('RMSE Improvement (%)', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ablation_heatmap_5seed.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: ablation_heatmap_5seed.png")

    # === Figure: Component Contribution Grouped Bar ===
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs_plot))
    width = 0.25

    for group_idx, (group_name, group_ds, color) in enumerate([
        ('FX', [d for d in datasets if d in FX_DATASETS], '#42A5F5'),
        ('Stocks', [d for d in datasets if d in STOCK_DATASETS], '#EF5350'),
        ('All', datasets, '#66BB6A'),
    ]):
        avgs, stds = [], []
        for cfg in configs_plot:
            improvements = []
            for ds in group_ds:
                v = np.mean(results[ds]["Vanilla"]["rmse"])
                c = np.mean(results[ds][cfg]["rmse"])
                improvements.append((v - c) / v * 100)
            avgs.append(np.mean(improvements))
            stds.append(np.std(improvements))

        bars = ax.bar(x + (group_idx - 1) * width, avgs, width,
                      label=group_name, color=color, edgecolor='#37474F',
                      linewidth=0.8, alpha=0.85, yerr=stds, capsize=3)

        for bar, avg, std in zip(bars, avgs, stds):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + std + 0.5,
                    f'{avg:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=0, color='#9E9E9E', linewidth=1, linestyle='--')
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Avg RMSE Improvement over Vanilla (%)', fontsize=11)
    ax.set_title('Component Contribution Summary (5-seed mean)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_plot, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ablation_component_5seed.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: ablation_component_5seed.png")

    # === Figure: Absolute RMSE Table ===
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')

    col_labels = ['Dataset'] + CONFIGS + ['Best']
    n_cols = len(col_labels)

    cell_text = []
    cell_colors = []

    for ds in datasets:
        row = [ds]
        rmse_means = []
        for cfg in CONFIGS:
            if cfg in results[ds]:
                m = np.mean(results[ds][cfg]["rmse"])
                s = np.std(results[ds][cfg]["rmse"])
                rmse_means.append(m)
                row.append(f'{m:.3f}±{s:.3f}')
            else:
                rmse_means.append(999)
                row.append('N/A')

        best_idx = int(np.argmin(rmse_means))
        row.append(CONFIGS[best_idx])
        cell_text.append(row)

        colors = ['#FAFAFA'] * n_cols
        colors[best_idx + 1] = '#C8E6C9'
        if ds in FX_DATASETS:
            colors[0] = '#E3F2FD'
        else:
            colors[0] = '#FCE4EC'
        cell_colors.append(colors)

    # Summary row
    summary_row = ['Average']
    for cfg in CONFIGS:
        vals = [np.mean(results[ds][cfg]["rmse"]) for ds in datasets if cfg in results[ds]]
        if vals:
            summary_row.append(f'{np.mean(vals):.3f}')
        else:
            summary_row.append('--')
    summary_row.append('')
    cell_text.append(summary_row)
    cell_colors.append(['#E0E0E0'] * n_cols)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    for j in range(n_cols):
        table[0, j].set_text_props(fontweight='bold')
        table[0, j].set_facecolor('#424242')
        table[0, j].set_text_props(color='white', fontweight='bold')

    n_fx = len([d for d in datasets if d in FX_DATASETS])
    for j in range(n_cols):
        table[n_fx, j].set_edgecolor('#424242')
        table[n_fx, j].set_linewidth(2)

    last_row = len(datasets) + 1
    for j in range(n_cols):
        table[last_row, j].set_text_props(fontweight='bold')

    ax.set_title('Ablation Study: Test RMSE (5-seed mean ± std)\n'
                 'Green = best per dataset',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ablation_table_5seed.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: ablation_table_5seed.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full 4-config ablation with 5 seeds")
    parser.add_argument("--analyze", action="store_true", help="Only analyze saved results")
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No results at {RESULTS_PATH}. Run without --analyze first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = run_experiment()

    analyze(results)
    print("\nGenerating figures...")
    plot_figures(results)
    print("Done!")
