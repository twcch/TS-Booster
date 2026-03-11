"""
Hyperparameter Sensitivity Analysis for TS-Booster Paper.

Varies one hyperparameter at a time while holding others at default.
Evaluates on 16 datasets x 3 seeds per config variant.

Hyperparameters tested:
  1. decomp_kernels: [5,11], [7,13]*, [11,21], [7,13,21]
  2. n_boost_rounds: 1, 2*, 3
  3. aux_weight: 0.1, 0.3*, 0.5
  4. base_weight: 0.0, 0.1*, 0.3

(* = default)

Usage:
  uv run python experiments/sensitivity_analysis.py                     # Full run
  uv run python experiments/sensitivity_analysis.py --analyze           # Analyze saved results
  uv run python experiments/sensitivity_analysis.py --param decomp      # Run one param group only
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
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

SEEDS = [42, 123, 456]  # 3 seeds for sensitivity (sufficient for trend analysis)

RESULTS_DIR = Path("experiments/comprehensive_results/sensitivity")
RESULTS_PATH = RESULTS_DIR / "sensitivity_results.json"
FIGURES_DIR = RESULTS_DIR / "figures"

# Default config values (for reference)
DEFAULTS = {
    "decomp_kernels": [7, 13],
    "n_boost_rounds": 2,
    "fixed_aux_weight": 0.3,
    "fixed_base_weight": 0.1,
}

# Sensitivity sweep definitions: param_name -> list of (label, value)
PARAM_SWEEPS = OrderedDict({
    "decomp": [
        ("[5,11]", [5, 11]),
        ("[7,13]", [7, 13]),       # default
        ("[11,21]", [11, 21]),
        ("[7,13,21]", [7, 13, 21]),
    ],
    "boost_rounds": [
        ("1", 1),
        ("2", 2),   # default
        ("3", 3),
    ],
    "aux_weight": [
        ("0.1", 0.1),
        ("0.3", 0.3),  # default
        ("0.5", 0.5),
    ],
    "base_weight": [
        ("0.0", 0.0),
        ("0.1", 0.1),  # default
        ("0.3", 0.3),
    ],
})

# Which label is the default for each param
DEFAULT_LABELS = {
    "decomp": "[7,13]",
    "boost_rounds": "2",
    "aux_weight": "0.3",
    "base_weight": "0.1",
}


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
    return float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))


def run_variant(ds, seed, device, decomp_kernels=None, n_boost_rounds=None,
                aux_weight=None, base_weight=None):
    """Run TS-Booster with one hyperparameter varied."""
    config = BEST_STAGE3_CONFIG.copy()
    config['epochs'] = 100
    config['input_channels'] = 2
    seq_len, pred_len = 30, 3

    # Override specific params
    dk = decomp_kernels if decomp_kernels is not None else DEFAULTS["decomp_kernels"]
    nbr = n_boost_rounds if n_boost_rounds is not None else DEFAULTS["n_boost_rounds"]
    aw = aux_weight if aux_weight is not None else DEFAULTS["fixed_aux_weight"]
    bw = base_weight if base_weight is not None else DEFAULTS["fixed_base_weight"]

    config['fixed_aux_weight'] = aw
    config['fixed_base_weight'] = bw

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
        use_decomp=True, use_experts=True,
        expert_gate='snr', enrich_mode='trend',
        decomp_kernels=dk,
        expert_kernels=config.get('expert_kernels', [3, 5, 7]),
        expert_hidden_dim=config.get('expert_hidden_dim', 32),
        dropout_p=config.get('dropout', 0.3),
        n_boost_rounds=nbr,
        expert_type=config.get('expert_type', 'cnn'),
    )
    model = train_v11(
        train_loader, val_loader, test_loader, device=device,
        horizon=pred_len, config=config, seq_len=seq_len, verbose=False, model=model
    )
    return evaluate(model, test_loader, scaler, device)


def run_experiments(param_filter=None):
    """Run sensitivity sweep."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")

    # Load existing results for incremental runs
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    params_to_run = [param_filter] if param_filter else list(PARAM_SWEEPS.keys())

    for param_name in params_to_run:
        if param_name not in PARAM_SWEEPS:
            print(f"ERROR: Unknown param '{param_name}'. Choose from {list(PARAM_SWEEPS.keys())}")
            sys.exit(1)

        sweep = PARAM_SWEEPS[param_name]
        print(f"\n{'#'*70}")
        print(f"  Sweeping: {param_name}")
        print(f"  Variants: {[label for label, _ in sweep]}")
        print(f"  Default:  {DEFAULT_LABELS[param_name]}")
        print(f"{'#'*70}")

        results.setdefault(param_name, {})

        for label, value in sweep:
            print(f"\n  --- {param_name} = {label} ---")
            results[param_name].setdefault(label, {})

            for ds in ALL_DATASETS:
                # Check if already done
                existing = results[param_name][label].get(ds, [])
                if len(existing) >= len(SEEDS):
                    print(f"    {ds:12s}: SKIP (already {len(existing)} seeds)")
                    continue

                rmses = []
                for seed in SEEDS:
                    t0 = time.time()
                    kwargs = {}
                    if param_name == "decomp":
                        kwargs["decomp_kernels"] = value
                    elif param_name == "boost_rounds":
                        kwargs["n_boost_rounds"] = value
                    elif param_name == "aux_weight":
                        kwargs["aux_weight"] = value
                    elif param_name == "base_weight":
                        kwargs["base_weight"] = value

                    rmse = run_variant(ds, seed, device, **kwargs)
                    if rmse is not None:
                        rmses.append(rmse)
                    elapsed = time.time() - t0
                    print(f"    {ds:12s} | seed={seed} | RMSE={rmse:.4f} | {elapsed:.1f}s")

                if rmses:
                    results[param_name][label][ds] = rmses

                # Incremental save
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                with open(RESULTS_PATH, 'w') as f:
                    json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    return results


def analyze(results):
    """Print sensitivity analysis tables."""
    print("\n" + "=" * 100)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 100)

    for param_name, sweep in PARAM_SWEEPS.items():
        if param_name not in results:
            continue

        labels = [label for label, _ in sweep]
        default_label = DEFAULT_LABELS[param_name]

        print(f"\n{'─'*80}")
        print(f"  Parameter: {param_name}  (default = {default_label})")
        print(f"{'─'*80}")

        # Per-dataset table
        header = f"  {'Dataset':<12}"
        for label in labels:
            marker = "*" if label == default_label else " "
            header += f" | {label + marker:>10}"
        header += f" | {'Best':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        all_means = {label: [] for label in labels}
        best_counts = {label: 0 for label in labels}

        for ds in ALL_DATASETS:
            row = f"  {ds:<12}"
            ds_means = {}
            for label in labels:
                rmses = results[param_name].get(label, {}).get(ds, [])
                if rmses:
                    m = np.mean(rmses)
                    ds_means[label] = m
                    all_means[label].append(m)
                    row += f" | {m:>10.4f}"
                else:
                    row += f" | {'N/A':>10}"

            if ds_means:
                best = min(ds_means, key=lambda k: ds_means[k])
                best_counts[best] += 1
                row += f" | {best:>10}"
            print(row)

        # Summary
        print("  " + "-" * (len(header) - 2))
        avg_row = f"  {'Average':<12}"
        for label in labels:
            if all_means[label]:
                avg_row += f" | {np.mean(all_means[label]):>10.4f}"
            else:
                avg_row += f" | {'N/A':>10}"
        avg_row += f" |"
        print(avg_row)

        wins_row = f"  {'Wins':<12}"
        for label in labels:
            wins_row += f" | {best_counts[label]:>10d}"
        wins_row += f" |"
        print(wins_row)

        # Relative to default
        if all_means.get(default_label):
            default_avg = np.mean(all_means[default_label])
            rel_row = f"  {'vs Default':<12}"
            for label in labels:
                if all_means[label]:
                    gap = (default_avg - np.mean(all_means[label])) / default_avg * 100
                    rel_row += f" | {gap:>+9.2f}%"
                else:
                    rel_row += f" | {'N/A':>10}"
            rel_row += f" |"
            print(rel_row)

    # LaTeX table
    print(f"\n{'='*100}")
    print("LATEX TABLE")
    print("=" * 100)
    print()
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Hyperparameter sensitivity analysis (3-seed mean RMSE, 16 datasets). "
          r"Default values marked with $\dagger$.}")
    print(r"\label{tab:sensitivity}")
    print(r"\begin{tabular}{llrrr}")
    print(r"\toprule")
    print(r"Parameter & Value & Avg RMSE & Wins & $\Delta$(\%) \\")
    print(r"\midrule")

    for param_name, sweep in PARAM_SWEEPS.items():
        if param_name not in results:
            continue
        labels = [label for label, _ in sweep]
        default_label = DEFAULT_LABELS[param_name]

        # Compute stats
        all_means = {}
        best_counts = {label: 0 for label in labels}
        for label in labels:
            ds_means = []
            for ds in ALL_DATASETS:
                rmses = results[param_name].get(label, {}).get(ds, [])
                if rmses:
                    ds_means.append(np.mean(rmses))
            all_means[label] = ds_means

        # Count wins
        for ds in ALL_DATASETS:
            ds_vals = {}
            for label in labels:
                rmses = results[param_name].get(label, {}).get(ds, [])
                if rmses:
                    ds_vals[label] = np.mean(rmses)
            if ds_vals:
                best = min(ds_vals, key=lambda k: ds_vals[k])
                best_counts[best] += 1

        default_avg = np.mean(all_means[default_label]) if all_means.get(default_label) else None

        # Pretty param names for LaTeX
        param_latex = {
            "decomp": r"Decomp.\ kernels",
            "boost_rounds": r"Boost rounds",
            "aux_weight": r"$\lambda_{\text{aux}}$",
            "base_weight": r"$\lambda_{\text{base}}$",
        }.get(param_name, param_name)

        for i, label in enumerate(labels):
            pname = param_latex if i == 0 else ""
            dagger = r"$^\dagger$" if label == default_label else ""
            avg = np.mean(all_means[label]) if all_means[label] else float('nan')
            wins = best_counts[label]
            if default_avg and all_means[label]:
                gap = (default_avg - avg) / default_avg * 100
                gap_str = f"{gap:+.2f}"
            else:
                gap_str = "--"

            if label == default_label:
                print(f"{pname} & \\textbf{{{label}}}{dagger} & \\textbf{{{avg:.4f}}} & \\textbf{{{wins}}} & {gap_str} \\\\")
            else:
                print(f"{pname} & {label}{dagger} & {avg:.4f} & {wins} & {gap_str} \\\\")

        if param_name != list(PARAM_SWEEPS.keys())[-1]:
            print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def plot_figures(results):
    """Generate sensitivity analysis figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n_params = len([p for p in PARAM_SWEEPS if p in results])
    if n_params == 0:
        print("  No data to plot.")
        return

    fig, axes = plt.subplots(1, n_params, figsize=(4.5 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    param_titles = {
        "decomp": "Decomposition Kernels",
        "boost_rounds": "Boosting Rounds",
        "aux_weight": "Expert Aux Weight ($\\lambda_{aux}$)",
        "base_weight": "Base Weight ($\\lambda_{base}$)",
    }

    colors = ['#42A5F5', '#66BB6A', '#EF5350', '#FF9800', '#AB47BC']

    for ax_idx, param_name in enumerate(PARAM_SWEEPS):
        if param_name not in results:
            continue
        ax = axes[ax_idx]

        sweep = PARAM_SWEEPS[param_name]
        labels = [label for label, _ in sweep]
        default_label = DEFAULT_LABELS[param_name]

        # Compute per-dataset means and overall stats
        x_pos = np.arange(len(labels))
        means = []
        for label in labels:
            ds_means = []
            for ds in ALL_DATASETS:
                rmses = results[param_name].get(label, {}).get(ds, [])
                if rmses:
                    ds_means.append(np.mean(rmses))
            means.append(np.mean(ds_means) if ds_means else 0)

        bar_colors = []
        for label in labels:
            if label == default_label:
                bar_colors.append('#43A047')
            else:
                bar_colors.append('#90CAF9')

        bars = ax.bar(x_pos, means, color=bar_colors,
                      edgecolor='#37474F', linewidth=0.8, width=0.6)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.002,
                    f'{m:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Mark default
        default_idx = labels.index(default_label)
        ax.annotate('default', xy=(default_idx, means[default_idx] * 0.999),
                    ha='center', fontsize=8, color='#1B5E20', fontstyle='italic')

        ax.set_xlabel(param_titles.get(param_name, param_name), fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Mean RMSE (16 datasets)', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Zoom y-axis to show differences clearly
        y_min = min(means) * 0.995
        y_max = max(means) * 1.008
        ax.set_ylim(y_min, y_max)

    plt.suptitle('Hyperparameter Sensitivity Analysis (3 seeds, 16 datasets)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sensitivity_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: sensitivity_analysis.png")

    # === Per-dataset heatmap for decomp_kernels ===
    if "decomp" in results:
        _plot_decomp_heatmap(results)

    # === Line plot: boost rounds effect ===
    if "boost_rounds" in results:
        _plot_boost_rounds_line(results)


def _plot_decomp_heatmap(results):
    """Heatmap: decomp kernel sensitivity per dataset."""
    sweep = PARAM_SWEEPS["decomp"]
    labels = [label for label, _ in sweep]
    default_label = DEFAULT_LABELS["decomp"]

    # Compute improvement vs default
    matrix = np.zeros((len(ALL_DATASETS), len(labels)))
    for i, ds in enumerate(ALL_DATASETS):
        default_rmse = results["decomp"].get(default_label, {}).get(ds, [])
        default_mean = np.mean(default_rmse) if default_rmse else None
        for j, label in enumerate(labels):
            rmses = results["decomp"].get(label, {}).get(ds, [])
            if rmses and default_mean:
                gap = (default_mean - np.mean(rmses)) / default_mean * 100
                matrix[i, j] = gap

    fig, ax = plt.subplots(figsize=(7, 9))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 1)
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(ALL_DATASETS)))
    ax.set_yticklabels(ALL_DATASETS, fontsize=9)

    for i in range(len(ALL_DATASETS)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)

    ax.set_title('Decomposition Kernel Sensitivity\n(RMSE % vs default [7,13])',
                 fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Improvement %', shrink=0.7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sensitivity_decomp_heatmap.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: sensitivity_decomp_heatmap.png")


def _plot_boost_rounds_line(results):
    """Line plot: avg RMSE vs boost rounds, with per-category lines."""
    sweep = PARAM_SWEEPS["boost_rounds"]
    labels = [label for label, _ in sweep]
    x_vals = [int(label) for label in labels]

    categories = {
        "All (16)": ALL_DATASETS,
        "FX (10)": ALL_DATASETS[:10],
        "Stocks (6)": ALL_DATASETS[10:],
    }
    cat_colors = {"All (16)": '#424242', "FX (10)": '#1565C0', "Stocks (6)": '#C62828'}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for cat_name, cat_ds in categories.items():
        means = []
        for label in labels:
            ds_means = []
            for ds in cat_ds:
                rmses = results["boost_rounds"].get(label, {}).get(ds, [])
                if rmses:
                    ds_means.append(np.mean(rmses))
            means.append(np.mean(ds_means) if ds_means else 0)
        ax.plot(x_vals, means, 'o-', color=cat_colors[cat_name], label=cat_name,
                linewidth=2, markersize=7)
        for xv, m in zip(x_vals, means):
            ax.text(xv + 0.08, m, f'{m:.3f}', fontsize=8, va='center')

    ax.set_xlabel('Number of Boosting Rounds', fontsize=11)
    ax.set_ylabel('Mean RMSE', fontsize=11)
    ax.set_title('Effect of Boosting Rounds', fontsize=12, fontweight='bold')
    ax.set_xticks(x_vals)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sensitivity_boost_rounds.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: sensitivity_boost_rounds.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity analysis for TS-Booster")
    parser.add_argument("--analyze", action="store_true", help="Only analyze saved results")
    parser.add_argument("--param", type=str, choices=list(PARAM_SWEEPS.keys()),
                        help="Run only one parameter group (default: all)")
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No saved results at {RESULTS_PATH}. Run without --analyze first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = run_experiments(param_filter=args.param)

    analyze(results)
    print("\nGenerating figures...")
    plot_figures(results)
    print("Done!")
