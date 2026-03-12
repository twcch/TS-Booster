"""
SOTA Deep Learning Baselines Benchmark for TS-Booster Paper.

Compares Linear+TSB against 4 recent SOTA DL models (DLinear, PatchTST,
iTransformer, TimeMixer) plus vanilla Linear across 16 datasets × 5 seeds.

DL results (Linear vanilla & Linear+TSB) are loaded from saved 5-seed results
to avoid redundant GPU training.

Usage:
  uv run python experiments/sota_baselines.py                              # Full run
  uv run python experiments/sota_baselines.py --analyze                    # Tables from saved JSON
  uv run python experiments/sota_baselines.py --models DLinear PatchTST    # Only specific models
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
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.models.sota_baselines import create_sota_model, SOTA_MODELS
from src.engine.trainer import train_baseline
from src.utils.data_loader import prepare_data

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

CATEGORIES = {
    "Taiwan Stocks": ["0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"],
    "TWD FX": ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD"],
    "International FX": ["EUR_USD", "GBP_USD", "USD_JPY"],
}

DL_RESULTS_PATH = Path("experiments/comprehensive_results/main_experiment/statistical_test_results.json")
RESULTS_PATH = Path("experiments/comprehensive_results/sota_baselines/sota_baselines_results.json")
FIGURES_DIR = Path("experiments/comprehensive_results/sota_baselines/figures")

SEEDS = [42, 123, 456, 789, 2024]
SEQ_LEN = 30
PRED_LEN = 3
INPUT_CHANNELS = 2
SOTA_MODEL_NAMES = ["DLinear", "PatchTST", "iTransformer"]

# Per-model learning rates from original papers/repos
SOTA_LR = {
    "DLinear": 1e-4,      # Official repo default
    "PatchTST": 1e-4,     # Official repo default
    "iTransformer": 5e-4,  # Official repo default
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


def load_existing_dl_results():
    """Load vanilla/booster metrics from saved 5-seed results."""
    if not DL_RESULTS_PATH.exists():
        print(f"WARNING: {DL_RESULTS_PATH} not found. Linear/TSB results will be missing.")
        return {}, {}

    with open(DL_RESULTS_PATH) as f:
        raw = json.load(f)

    vanilla_metrics = {}
    booster_metrics = {}
    for ds in ALL_DATASETS:
        if ds in raw:
            v = raw[ds]["vanilla"]
            b = raw[ds]["booster"]
            if isinstance(v, dict):
                vanilla_metrics[ds] = {k: v[k] for k in ["rmse", "mae", "mda"] if k in v}
            else:
                vanilla_metrics[ds] = {"rmse": v}
            if isinstance(b, dict):
                booster_metrics[ds] = {k: b[k] for k in ["rmse", "mae", "mda"] if k in b}
            else:
                booster_metrics[ds] = {"rmse": b}

    return vanilla_metrics, booster_metrics


def _get_metric_mean(results, ds, method, metric="rmse"):
    """Get mean of a metric from results dict."""
    val = results.get(ds, {}).get(method)
    if val is None:
        return None
    if isinstance(val, dict):
        arr = val.get(metric, [])
        return float(np.mean(arr)) if arr else None
    if metric == "rmse":
        if isinstance(val, list):
            return float(np.mean(val)) if val else None
        return float(val)
    return None


def _get_metric_list(results, ds, method, metric="rmse"):
    """Get list of a metric from results dict."""
    val = results.get(ds, {}).get(method)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get(metric, None)
    if metric == "rmse":
        if isinstance(val, list):
            return val
        return [val]
    return None


def run_single_model(model_name, ds, seed, device):
    """Train one SOTA model for one dataset/seed. Returns metrics dict."""
    path = f"dataset/{ds}.csv"
    if not Path(path).exists():
        return None

    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32
    )

    set_seed(seed)
    model = create_sota_model(model_name, seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=INPUT_CHANNELS)
    lr = SOTA_LR.get(model_name, 1e-3)
    model = train_baseline(
        train_loader, val_loader, test_loader, device=device,
        model=model, num_epochs=100, lr=lr, verbose=False
    )
    metrics = evaluate(model, test_loader, scaler, device)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return metrics


def run_experiments(args):
    """Run SOTA baseline experiments across datasets × seeds."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Determine which models to run
    models_to_run = args.models if args.models else SOTA_MODEL_NAMES
    for m in models_to_run:
        if m not in SOTA_MODELS:
            print(f"ERROR: Unknown model '{m}'. Choose from {SOTA_MODEL_NAMES}")
            sys.exit(1)

    # Load existing results for incremental runs
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    # Load Linear/TSB results
    vanilla_metrics, booster_metrics = load_existing_dl_results()
    for ds in ALL_DATASETS:
        if ds not in results:
            results[ds] = {}
        if ds in vanilla_metrics:
            results[ds]["Linear"] = vanilla_metrics[ds]
        if ds in booster_metrics:
            results[ds]["Linear+TSB"] = booster_metrics[ds]

    total_runs = len(models_to_run) * len(ALL_DATASETS) * len(SEEDS)
    run_count = 0
    total_start = time.time()

    for model_name in models_to_run:
        print(f"\n{'#'*70}")
        print(f"  Model: {model_name}")
        print(f"{'#'*70}")

        # Print param count
        dummy = create_sota_model(model_name, SEQ_LEN, PRED_LEN, INPUT_CHANNELS)
        n_params = sum(p.numel() for p in dummy.parameters())
        print(f"  Parameters: {n_params:,}")
        del dummy

        for ds in ALL_DATASETS:
            print(f"\n  {'='*50}")
            print(f"  Dataset: {ds}")

            # Check if already completed (all 5 seeds present with all metrics)
            existing = _get_metric_list(results, ds, model_name, "rmse")
            existing_mae = _get_metric_list(results, ds, model_name, "mae")
            if existing and len(existing) >= len(SEEDS) and existing_mae and len(existing_mae) >= len(SEEDS):
                run_count += len(SEEDS)
                print(f"  SKIP (already have {len(existing)} seeds with all metrics)")
                continue

            seed_metrics = {"rmse": [], "mae": [], "mda": []}
            for seed in SEEDS:
                run_count += 1
                t0 = time.time()
                print(f"    Seed {seed} ({run_count}/{total_runs}) ... ", end="", flush=True)

                metrics = run_single_model(model_name, ds, seed, device)
                if metrics is None:
                    print("SKIP (file not found)")
                    continue

                for k in seed_metrics:
                    seed_metrics[k].append(metrics[k])
                elapsed = time.time() - t0
                print(f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} MDA={metrics['mda']:.4f} ({elapsed:.1f}s)")

            if seed_metrics["rmse"]:
                results[ds][model_name] = seed_metrics
                print(f"  {model_name}: RMSE={np.mean(seed_metrics['rmse']):.4f} MAE={np.mean(seed_metrics['mae']):.4f} MDA={np.mean(seed_metrics['mda']):.4f}")

                # Incremental save
                RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(RESULTS_PATH, 'w') as f:
                    json.dump(results, f, indent=2)

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to {RESULTS_PATH}")
    return results


# ──────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────

def analyze(results):
    """Print terminal table, paired t-tests, category breakdown, and LaTeX."""
    methods = ["DLinear", "PatchTST", "iTransformer", "Linear", "Linear+TSB"]

    print("\n" + "=" * 130)
    print("SOTA BASELINES BENCHMARK: All Methods x 16 Datasets (5-seed mean RMSE)")
    print("=" * 130)

    # --- Terminal table ---
    header = f"{'Dataset':<12}"
    for m in methods:
        header += f" | {m:>14}"
    header += f" | {'TSB vs Best':>14}"
    print(f"\n{header}")
    print("-" * len(header))

    tsb_vs_best_gaps = []

    for ds in ALL_DATASETS:
        if ds not in results:
            continue

        line = f"  {ds:<12}"
        rmse_means = {}

        for m in methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                rmse_means[m] = val
                line += f" | {val:>14.4f}"
            else:
                line += f" | {'N/A':>14}"

        # TSB vs best SOTA
        sota_methods = ["DLinear", "PatchTST", "iTransformer"]
        external = {k: v for k, v in rmse_means.items() if k in sota_methods}
        if external and "Linear+TSB" in rmse_means:
            best_ext = min(external.values())
            gap = (best_ext - rmse_means["Linear+TSB"]) / best_ext * 100
            tsb_vs_best_gaps.append(gap)
            tag = "WIN" if gap > 0 else "LOSS"
            line += f" | {gap:>+12.2f}% {tag}"
        else:
            line += f" | {'N/A':>14}"

        print(line)

    print("-" * len(header))
    if tsb_vs_best_gaps:
        wins = sum(1 for g in tsb_vs_best_gaps if g > 0)
        print(f"  TSB vs Best SOTA: {wins}/{len(tsb_vs_best_gaps)} wins, avg gap = {np.mean(tsb_vs_best_gaps):+.2f}%")

    # --- Pairwise win counts ---
    print(f"\n  TSB vs each method (RMSE wins):")
    for m in methods:
        if m == "Linear+TSB":
            continue
        wins, total = 0, 0
        for ds in ALL_DATASETS:
            tsb_val = _get_metric_mean(results, ds, "Linear+TSB")
            other_val = _get_metric_mean(results, ds, m)
            if tsb_val is not None and other_val is not None:
                total += 1
                if tsb_val < other_val:
                    wins += 1
        if total > 0:
            print(f"    vs {m:<14}: {wins}/{total} ({wins/total*100:.0f}%)")

    # --- Paired t-tests (TSB vs each SOTA, using 5-seed arrays) ---
    print(f"\n{'='*80}")
    print("PAIRED T-TESTS: Linear+TSB vs SOTA Models (per-dataset, 5 seeds)")
    print("=" * 80)

    sota_methods = ["DLinear", "PatchTST", "iTransformer"]
    for m in sota_methods:
        print(f"\n  Linear+TSB vs {m}:")
        sig_wins = 0
        tested = 0
        for ds in ALL_DATASETS:
            tsb_list = _get_metric_list(results, ds, "Linear+TSB")
            other_list = _get_metric_list(results, ds, m)
            if tsb_list is None or other_list is None:
                continue
            if len(tsb_list) < 2 or len(other_list) < 2:
                continue
            tested += 1
            # One-sided t-test: H1: TSB < SOTA
            t_stat, p_two = stats.ttest_ind(tsb_list, other_list)
            p_val = p_two / 2 if t_stat < 0 else 1 - p_two / 2
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            gap = (np.mean(other_list) - np.mean(tsb_list)) / np.mean(other_list) * 100
            if p_val < 0.05 and gap > 0:
                sig_wins += 1
            print(f"    {ds:<12}: gap={gap:+6.2f}%, p={p_val:.4f} {sig}")
        if tested > 0:
            print(f"    Significant wins: {sig_wins}/{tested}")

    # --- Category breakdown ---
    print(f"\n{'='*80}")
    print("CATEGORY BREAKDOWN (mean RMSE)")
    print("=" * 80)

    for cat_name, cat_datasets in CATEGORIES.items():
        print(f"\n  {cat_name}:")
        cat_means = {m: [] for m in methods}
        for ds in cat_datasets:
            for m in methods:
                val = _get_metric_mean(results, ds, m)
                if val is not None:
                    cat_means[m].append(val)

        line = "    Avg RMSE:"
        for m in methods:
            if cat_means[m]:
                line += f"  {m}={np.mean(cat_means[m]):.4f}"
        print(line)

    # --- Rank analysis ---
    print(f"\n{'='*80}")
    print("AVERAGE RANK ACROSS DATASETS")
    print("=" * 80)

    ranks_per_method = {m: [] for m in methods}
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        available = {}
        for m in methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                available[m] = val
        if not available:
            continue
        sorted_methods = sorted(available, key=lambda k: available[k])
        for rank, m in enumerate(sorted_methods, 1):
            ranks_per_method[m].append(rank)

    for m in methods:
        if ranks_per_method[m]:
            avg_rank = np.mean(ranks_per_method[m])
            print(f"  {m:<14}: avg rank = {avg_rank:.2f}  (ranks: {ranks_per_method[m]})")

    # --- LaTeX table ---
    print(f"\n{'='*80}")
    print("LATEX TABLE")
    print("=" * 80)
    print()
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{RMSE comparison: SOTA deep learning baselines vs.\ Linear+TS-Booster on 16 datasets (5-seed mean). \textbf{Bold} = best.}")
    print(r"\label{tab:sota_baselines}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l" + "r" * len(methods) + "}")
    print(r"\toprule")
    print("Dataset & " + " & ".join(methods) + r" \\")
    print(r"\midrule")

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        ds_latex = ds.replace("_", r"\_")

        available_rmse = {}
        for m in methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                available_rmse[m] = val
        best_rmse = min(available_rmse.values()) if available_rmse else None

        cells = []
        for m in methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                if best_rmse is not None and abs(val - best_rmse) < 1e-8:
                    cells.append(r"\textbf{" + f"{val:.4f}" + "}")
                else:
                    cells.append(f"{val:.4f}")
            else:
                cells.append("--")

        print(f"{ds_latex} & " + " & ".join(cells) + r" \\")

    # Average row
    print(r"\midrule")
    avg_rmse = {}
    avg_cells = []
    for m in methods:
        vals = [_get_metric_mean(results, ds, m) for ds in ALL_DATASETS]
        vals = [v for v in vals if v is not None]
        if vals:
            avg = np.mean(vals)
            avg_rmse[m] = avg
            avg_cells.append(f"{avg:.4f}")
        else:
            avg_cells.append("--")
    if avg_rmse:
        best_avg = min(avg_rmse.values())
        final_cells = []
        for m, c in zip(methods, avg_cells):
            if m in avg_rmse and abs(avg_rmse[m] - best_avg) < 1e-8:
                final_cells.append(r"\textbf{" + c + "}")
            else:
                final_cells.append(c)
        avg_cells = final_cells
    print(r"\textit{Average} & " + " & ".join(avg_cells) + r" \\")

    # Average rank row
    rank_cells = []
    for m in methods:
        if ranks_per_method[m]:
            rank_cells.append(f"{np.mean(ranks_per_method[m]):.2f}")
        else:
            rank_cells.append("--")
    print(r"\textit{Avg.\ Rank} & " + " & ".join(rank_cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

FX_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
]


def plot_tsb_vs_sota(results):
    """Horizontal bar: TSB RMSE improvement % over best SOTA baseline per dataset."""
    sota_methods = ["DLinear", "PatchTST", "iTransformer"]

    datasets, diffs, colors, best_names = [], [], [], []
    for ds in ALL_DATASETS:
        tsb = _get_metric_mean(results, ds, "Linear+TSB")
        if tsb is None:
            continue
        ext_rmse = {}
        for m in sota_methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                ext_rmse[m] = val
        if not ext_rmse:
            continue
        best_m = min(ext_rmse, key=lambda k: ext_rmse[k])
        best_val = ext_rmse[best_m]
        gap = (best_val - tsb) / best_val * 100
        datasets.append(ds)
        diffs.append(gap)
        colors.append('#43A047' if gap > 0 else '#E53935')
        best_names.append(best_m)

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(datasets))
    bars = ax.barh(y, diffs, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    for i, (bar, diff, bm) in enumerate(zip(bars, diffs, best_names)):
        w = bar.get_width()
        label = f'{diff:+.1f}% vs {bm}'
        if w >= 0:
            ax.text(w + 0.3, i, label, ha='left', va='center', fontsize=8, fontweight='bold')
        else:
            ax.text(w - 0.3, i, label, ha='right', va='center', fontsize=8, fontweight='bold')

    ax.axvline(x=0, color='#424242', linewidth=1)

    n_fx = len([d for d in datasets if d in FX_DATASETS])
    if 0 < n_fx < len(datasets):
        ax.axhline(y=n_fx - 0.5, color='#424242', linewidth=1, linestyle='--', alpha=0.5)

    wins = sum(1 for d in diffs if d > 0)
    max_val = max(diffs) if diffs else 10
    min_val = min(diffs) if diffs else -10
    ax.set_xlim(min_val - 8, max_val + 12)

    ax.set_xlabel('RMSE Improvement over Best SOTA Baseline (%)', fontsize=11)
    ax.set_title(f'Linear+TSB vs Best SOTA DL Model ({wins}/{len(diffs)} wins)', fontsize=13, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'tsb_vs_best_sota.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: tsb_vs_best_sota.png")


def plot_average_rank(results):
    """Bar chart: average rank of each method across 16 datasets."""
    methods = ["DLinear", "PatchTST", "iTransformer", "Linear", "Linear+TSB"]
    method_colors = ["#FF9800", "#03A9F4", "#E91E63", "#42A5F5", "#AB47BC"]

    ranks_per_method = {m: [] for m in methods}
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        available = {}
        for m in methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                available[m] = val
        if not available:
            continue
        sorted_methods = sorted(available, key=lambda k: available[k])
        for rank, m in enumerate(sorted_methods, 1):
            ranks_per_method[m].append(rank)

    avg_ranks = []
    labels = []
    bar_colors = []
    for m, c in zip(methods, method_colors):
        if ranks_per_method[m]:
            avg_ranks.append(np.mean(ranks_per_method[m]))
            labels.append(m)
            bar_colors.append(c)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, avg_ranks, color=bar_colors, edgecolor='#37474F', linewidth=0.8, width=0.6)

    for bar, rank in zip(bars, avg_ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{rank:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Average Rank (lower = better)', fontsize=11)
    ax.set_title('Average RMSE Rank: SOTA DL Baselines vs Linear+TSB', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylim(0, max(avg_ranks) + 0.8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sota_average_rank.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: sota_average_rank.png")


def plot_heatmap(results):
    """Heatmap: RMSE improvement of TSB over each SOTA model per dataset."""
    sota_methods = ["DLinear", "PatchTST", "iTransformer"]

    # Build matrix
    data_matrix = []
    ds_labels = []
    for ds in ALL_DATASETS:
        tsb = _get_metric_mean(results, ds, "Linear+TSB")
        if tsb is None:
            continue
        row = []
        has_data = False
        for m in sota_methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                gap = (val - tsb) / val * 100
                row.append(gap)
                has_data = True
            else:
                row.append(0)
        if has_data:
            data_matrix.append(row)
            ds_labels.append(ds)

    if not data_matrix:
        print("  SKIP heatmap (no data)")
        return

    matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=30)

    ax.set_xticks(np.arange(len(sota_methods)))
    ax.set_xticklabels(sota_methods, fontsize=10)
    ax.set_yticks(np.arange(len(ds_labels)))
    ax.set_yticklabels(ds_labels, fontsize=10)

    # Annotate cells
    for i in range(len(ds_labels)):
        for j in range(len(sota_methods)):
            val = matrix[i, j]
            color = 'white' if abs(val) > 15 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

    ax.set_title('RMSE Improvement: Linear+TSB vs SOTA Models (%)', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Improvement %', shrink=0.8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sota_heatmap.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: sota_heatmap.png")


def plot_metric_comparison(results, metric, higher_is_better=False):
    """Grouped bar chart: all methods × 16 datasets for a given metric."""
    methods = ["DLinear", "PatchTST", "iTransformer", "Linear", "Linear+TSB"]
    method_colors = ["#FF9800", "#03A9F4", "#E91E63", "#42A5F5", "#AB47BC"]

    datasets_with_data = []
    for ds in ALL_DATASETS:
        has_any = any(_get_metric_mean(results, ds, m, metric) is not None for m in methods)
        if has_any:
            datasets_with_data.append(ds)

    if not datasets_with_data:
        print(f"  SKIP {metric} chart (no data)")
        return

    n_ds = len(datasets_with_data)
    n_methods = len(methods)
    bar_width = 0.15
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(14, n_ds * 0.9), 6))

    # Find best method per dataset
    best_per_ds = {}
    for i, ds in enumerate(datasets_with_data):
        ds_vals = {}
        for m in methods:
            v = _get_metric_mean(results, ds, m, metric)
            if v is not None:
                ds_vals[m] = v
        if ds_vals:
            best_per_ds[i] = max(ds_vals, key=lambda k: ds_vals[k]) if higher_is_better else min(ds_vals, key=lambda k: ds_vals[k])

    bar_containers = []
    for j, (m, color) in enumerate(zip(methods, method_colors)):
        vals = []
        for ds in datasets_with_data:
            v = _get_metric_mean(results, ds, m, metric)
            vals.append(v if v is not None else 0)
        offset = (j - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=m, color=color, edgecolor='white', linewidth=0.5)
        bar_containers.append((m, bars))

    # Dim non-best bars
    for m, bars in bar_containers:
        for i, bar in enumerate(bars):
            if best_per_ds.get(i) != m:
                bar.set_alpha(0.35)

    metric_upper = metric.upper()
    better = "higher" if higher_is_better else "lower"
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel(f'{metric_upper} ({better} is better)', fontsize=11)
    ax.set_title(f'{metric_upper} Comparison: SOTA DL Models × 16 Datasets', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_with_data, fontsize=8, rotation=45, ha='right')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'sota_{metric}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: sota_{metric}.png")


def generate_figures(results):
    """Generate all SOTA benchmark figures."""
    print("\nGenerating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_tsb_vs_sota(results)
    plot_average_rank(results)
    plot_heatmap(results)
    plot_metric_comparison(results, "rmse", higher_is_better=False)
    plot_metric_comparison(results, "mae", higher_is_better=False)
    plot_metric_comparison(results, "mda", higher_is_better=True)
    print(f"Figures saved to {FIGURES_DIR}/")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTA DL baselines benchmark for TS-Booster")
    parser.add_argument("--analyze", action="store_true", help="Only analyze saved results")
    parser.add_argument("--models", nargs="+", choices=SOTA_MODEL_NAMES,
                        help="Run only specific models (default: all)")
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No saved results at {RESULTS_PATH}. Run without --analyze first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = run_experiments(args)

    analyze(results)
    generate_figures(results)
