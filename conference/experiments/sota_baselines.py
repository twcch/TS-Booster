"""
SOTA Baselines Benchmark: DLinear, PatchTST, iTransformer vs Linear+TSB-Lite.

Usage:
  cd conference
  uv run python experiments/sota_baselines.py              # Full run
  uv run python experiments/sota_baselines.py --analyze    # Analyze saved results
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.models.sota_baselines import create_sota_model, SOTA_MODELS, SOTA_LR
from src.engine.trainer import train_baseline
from src.utils.data_loader import prepare_data

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

SEEDS = [42, 123, 456, 789, 2024]
DATASET_DIR = Path("../dataset")
MAIN_RESULTS_PATH = Path("results/main/results.json")
RESULTS_PATH = Path("results/sota/results.json")
FIGURES_DIR = Path("results/sota/figures")

SEQ_LEN = 30
PRED_LEN = 3
SOTA_MODEL_NAMES = ["DLinear", "PatchTST", "iTransformer"]


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


def load_main_results():
    """Load Linear/TSB-Lite results from main experiment."""
    if not MAIN_RESULTS_PATH.exists():
        print(f"WARNING: {MAIN_RESULTS_PATH} not found. Run main_experiment.py first.")
        return {}, {}
    with open(MAIN_RESULTS_PATH) as f:
        raw = json.load(f)
    vanilla, booster = {}, {}
    for ds in ALL_DATASETS:
        if ds in raw:
            vanilla[ds] = {k: raw[ds]["vanilla"][k] for k in ["rmse", "mae", "mda"]}
            booster[ds] = {k: raw[ds]["booster"][k] for k in ["rmse", "mae", "mda"]}
    return vanilla, booster


def _mean(results, ds, method, metric="rmse"):
    val = results.get(ds, {}).get(method)
    if val is None:
        return None
    if isinstance(val, dict):
        arr = val.get(metric, [])
        return float(np.mean(arr)) if arr else None
    return float(np.mean(val)) if isinstance(val, list) else float(val)


def _list(results, ds, method, metric="rmse"):
    val = results.get(ds, {}).get(method)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get(metric)
    return val if isinstance(val, list) else [val]


def run_experiments(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_run = args.models if args.models else SOTA_MODEL_NAMES

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    # Load Linear/TSB from main experiment
    vanilla_m, booster_m = load_main_results()
    for ds in ALL_DATASETS:
        if ds not in results:
            results[ds] = {}
        if ds in vanilla_m:
            results[ds]["Linear"] = vanilla_m[ds]
        if ds in booster_m:
            results[ds]["Linear+TSB"] = booster_m[ds]

    for model_name in models_to_run:
        print(f"\n{'#'*50}  {model_name}  {'#'*50}")

        for ds in ALL_DATASETS:
            existing = _list(results, ds, model_name, "rmse")
            if existing and len(existing) >= len(SEEDS):
                print(f"  {ds}: SKIP (already done)")
                continue

            seed_metrics = {"rmse": [], "mae": [], "mda": []}
            for seed in SEEDS:
                t0 = time.time()
                print(f"  {ds} | {model_name} | Seed {seed} ... ", end="", flush=True)

                path = DATASET_DIR / f"{ds}.csv"
                if not path.exists():
                    print("SKIP")
                    continue

                df = pd.read_csv(path)
                train_loader, val_loader, test_loader, scaler = prepare_data(
                    df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32
                )

                set_seed(seed)
                model = create_sota_model(model_name, SEQ_LEN, PRED_LEN, 2)
                lr = SOTA_LR.get(model_name, 1e-3)
                model = train_baseline(
                    train_loader, val_loader, device, model,
                    num_epochs=100, lr=lr, verbose=False,
                )
                metrics = evaluate(model, test_loader, scaler, device)

                for k in seed_metrics:
                    seed_metrics[k].append(metrics[k])

                elapsed = time.time() - t0
                print(f"RMSE={metrics['rmse']:.4f} ({elapsed:.1f}s)")

                del model
                torch.cuda.empty_cache()

            if seed_metrics["rmse"]:
                results[ds][model_name] = seed_metrics
                RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2)

    return results


# ──────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────

def analyze(results):
    methods = ["DLinear", "PatchTST", "iTransformer", "Linear", "Linear+TSB"]

    print(f"\n{'='*120}")
    print("SOTA BENCHMARK (5-seed mean RMSE)")
    print("=" * 120)

    header = f"{'Dataset':<12}"
    for m in methods:
        header += f" | {m:>14}"
    print(f"\n{header}")
    print("-" * len(header))

    ranks_per_method = {m: [] for m in methods}

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        line = f"  {ds:<12}"
        available = {}
        for m in methods:
            val = _mean(results, ds, m)
            if val is not None:
                available[m] = val
                line += f" | {val:>14.4f}"
            else:
                line += f" | {'N/A':>14}"
        print(line)

        if available:
            sorted_m = sorted(available, key=lambda k: available[k])
            for rank, m in enumerate(sorted_m, 1):
                ranks_per_method[m].append(rank)

    print("-" * len(header))

    # Win counts
    print(f"\n  TSB vs each method (RMSE wins):")
    for m in methods:
        if m == "Linear+TSB":
            continue
        wins, total = 0, 0
        for ds in ALL_DATASETS:
            tsb = _mean(results, ds, "Linear+TSB")
            other = _mean(results, ds, m)
            if tsb is not None and other is not None:
                total += 1
                if tsb < other:
                    wins += 1
        if total > 0:
            print(f"    vs {m:<14}: {wins}/{total}")

    # Average ranks
    print(f"\n  Average ranks:")
    for m in methods:
        if ranks_per_method[m]:
            print(f"    {m:<14}: {np.mean(ranks_per_method[m]):.2f}")

    # LaTeX
    print(f"\n{'='*80}")
    print("LATEX TABLE")
    print("=" * 80)
    print()
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{RMSE: SOTA baselines vs.\ Linear+TSB-Lite (5-seed mean). \textbf{Bold} = best.}")
    print(r"\label{tab:sota}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l" + "r" * len(methods) + "}")
    print(r"\toprule")
    print("Dataset & " + " & ".join(methods) + r" \\")
    print(r"\midrule")

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        ds_latex = ds.replace("_", r"\_")
        rmse_vals = {}
        for m in methods:
            val = _mean(results, ds, m)
            if val is not None:
                rmse_vals[m] = val
        best = min(rmse_vals.values()) if rmse_vals else None

        cells = []
        for m in methods:
            val = _mean(results, ds, m)
            if val is not None:
                if best is not None and abs(val - best) < 1e-8:
                    cells.append(r"\textbf{" + f"{val:.4f}" + "}")
                else:
                    cells.append(f"{val:.4f}")
            else:
                cells.append("--")
        print(f"{ds_latex} & " + " & ".join(cells) + r" \\")

    # Average row
    print(r"\midrule")
    avg_cells = []
    avg_vals = {}
    for m in methods:
        vals = [_mean(results, ds, m) for ds in ALL_DATASETS]
        vals = [v for v in vals if v is not None]
        if vals:
            avg_vals[m] = np.mean(vals)
            avg_cells.append(f"{avg_vals[m]:.4f}")
        else:
            avg_cells.append("--")
    if avg_vals:
        best_avg = min(avg_vals.values())
        final_cells = []
        for m, c in zip(methods, avg_cells):
            if m in avg_vals and abs(avg_vals[m] - best_avg) < 1e-8:
                final_cells.append(r"\textbf{" + c + "}")
            else:
                final_cells.append(c)
        avg_cells = final_cells
    print(r"\textit{Average} & " + " & ".join(avg_cells) + r" \\")

    # Rank row
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

def plot_tsb_vs_sota(results):
    """Horizontal bar: TSB improvement % over best SOTA per dataset."""
    sota_methods = ["DLinear", "PatchTST", "iTransformer"]
    datasets, diffs, colors, best_names = [], [], [], []

    for ds in ALL_DATASETS:
        tsb = _mean(results, ds, "Linear+TSB")
        if tsb is None:
            continue
        ext_rmse = {}
        for m in sota_methods:
            val = _mean(results, ds, m)
            if val is not None:
                ext_rmse[m] = val
        if not ext_rmse:
            continue
        best_m = min(ext_rmse, key=lambda k: ext_rmse[k])
        best_val = ext_rmse[best_m]
        gap = (best_val - tsb) / best_val * 100
        datasets.append(ds)
        diffs.append(gap)
        colors.append("#43A047" if gap > 0 else "#E53935")
        best_names.append(best_m)

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(datasets))
    bars = ax.barh(y, diffs, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    for i, (bar, diff, bm) in enumerate(zip(bars, diffs, best_names)):
        w = bar.get_width()
        label = f"{diff:+.1f}% vs {bm}"
        if w >= 0:
            ax.text(w + 0.3, i, label, ha="left", va="center", fontsize=8, fontweight="bold")
        else:
            ax.text(w - 0.3, i, label, ha="right", va="center", fontsize=8, fontweight="bold")

    ax.axvline(x=0, color="#424242", linewidth=1)
    wins = sum(1 for d in diffs if d > 0)
    ax.set_title(f"Linear+TSB-Lite vs Best SOTA ({wins}/{len(diffs)} wins)",
                 fontsize=13, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE Improvement over Best SOTA (%)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "tsb_vs_best_sota.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: tsb_vs_best_sota.png")


def plot_average_rank(results):
    """Bar chart: average rank."""
    methods = ["DLinear", "PatchTST", "iTransformer", "Linear", "Linear+TSB"]
    method_colors = ["#FF9800", "#03A9F4", "#E91E63", "#42A5F5", "#AB47BC"]

    ranks_per = {m: [] for m in methods}
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        available = {}
        for m in methods:
            val = _mean(results, ds, m)
            if val is not None:
                available[m] = val
        if not available:
            continue
        sorted_m = sorted(available, key=lambda k: available[k])
        for rank, m in enumerate(sorted_m, 1):
            ranks_per[m].append(rank)

    avg_ranks, labels, bar_colors = [], [], []
    for m, c in zip(methods, method_colors):
        if ranks_per[m]:
            avg_ranks.append(np.mean(ranks_per[m]))
            labels.append(m)
            bar_colors.append(c)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, avg_ranks, color=bar_colors, edgecolor="#37474F", linewidth=0.8, width=0.6)

    for bar, rank in zip(bars, avg_ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{rank:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Average Rank (lower = better)", fontsize=11)
    ax.set_title("Average RMSE Rank: SOTA vs Linear+TSB-Lite", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylim(0, max(avg_ranks) + 0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "sota_average_rank.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: sota_average_rank.png")


def generate_figures(results):
    print("\nGenerating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_tsb_vs_sota(results)
    plot_average_rank(results)
    print(f"Figures saved to {FIGURES_DIR}/")


# ──────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--models", nargs="+", choices=SOTA_MODEL_NAMES)
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No saved results. Run without --analyze first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = run_experiments(args)

    analyze(results)
    generate_figures(results)
