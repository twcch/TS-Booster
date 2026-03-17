"""
Main Experiment: Vanilla Linear vs TSB-Lite across 16 datasets x 5 seeds.

Outputs:
  - results/main/results.json
  - results/main/figures/rmse_improvement.png
  - results/main/figures/mae_improvement.png
  - results/main/figures/mda_comparison.png
  - Terminal table + LaTeX table

Usage:
  cd conference
  uv run python experiments/main_experiment.py              # Full run
  uv run python experiments/main_experiment.py --analyze    # Analyze saved results only
"""

import sys
import json
import random
import argparse
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

from src.models.layers import LinearBackbone
from src.models.ts_booster import TSBoosterLite
from src.engine.trainer import train_booster, train_baseline
from src.configs.config import CONFIG
from src.utils.data_loader import prepare_data

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

SEEDS = [42, 123, 456, 789, 2024]
DATASET_DIR = Path("../dataset")
RESULTS_PATH = Path("results/main/results.json")
FIGURES_DIR = Path("results/main/figures")

FX_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
]


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


def run_pair(ds, seed, config):
    """Train Vanilla and TSB-Lite for one dataset/seed pair."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = config["seq_len"]
    pred_len = config["pred_len"]

    path = DATASET_DIR / f"{ds}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=seq_len, horizon=pred_len, batch_size=config["batch_size"]
    )

    # Vanilla
    set_seed(seed)
    vanilla = LinearBackbone(seq_len, 2, pred_len)
    vanilla = train_baseline(
        train_loader, val_loader, device, vanilla,
        num_epochs=config["epochs"], lr=config["lr"], verbose=False,
    )
    v_metrics = evaluate(vanilla, test_loader, scaler, device)

    # TSB-Lite
    set_seed(seed)
    booster = TSBoosterLite(
        base_model=LinearBackbone(seq_len, 2, pred_len),
        seq_len=seq_len, pred_len=pred_len, input_channels=2,
        decomp_kernel=config["decomp_kernel"],
        expert_kernels=config["expert_kernels"],
        expert_hidden_dim=config["expert_hidden_dim"],
        dropout_p=config["dropout"],
    )
    booster = train_booster(
        train_loader, val_loader, device, booster, config, verbose=False,
    )
    b_metrics = evaluate(booster, test_loader, scaler, device)

    del vanilla, booster
    torch.cuda.empty_cache()

    return v_metrics, b_metrics


def run_experiments():
    """Run all experiments and save results."""
    config = CONFIG.copy()
    results = {}

    for ds in ALL_DATASETS:
        results[ds] = {
            "vanilla": {"rmse": [], "mae": [], "mda": []},
            "booster": {"rmse": [], "mae": [], "mda": []},
        }
        for seed in SEEDS:
            print(f"  {ds:12s} | Seed {seed} ... ", end="", flush=True)
            pair = run_pair(ds, seed, config)
            if pair is None:
                print("SKIP")
                continue
            v_m, b_m = pair
            for key in ("rmse", "mae", "mda"):
                results[ds]["vanilla"][key].append(v_m[key])
                results[ds]["booster"][key].append(b_m[key])
            gap = (v_m["rmse"] - b_m["rmse"]) / v_m["rmse"] * 100
            tag = "WIN" if gap > 0 else "LOSS"
            print(f"V={v_m['rmse']:.4f} B={b_m['rmse']:.4f} {gap:+.2f}% {tag}")

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    return results


# ──────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────

def analyze(results):
    """Print statistical analysis and LaTeX table."""
    print("\n" + "=" * 100)
    print("VANILLA LINEAR vs TSB-LITE (5 seeds)")
    print("=" * 100)

    header = (
        f"{'Dataset':<12} | {'V_RMSE':>10} | {'B_RMSE':>10} | "
        f"{'Gap%':>8} | {'p-value':>8} | {'Sig':>5}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    dataset_gaps = []
    win_count = 0

    for ds in ALL_DATASETS:
        v = np.array(results[ds]["vanilla"]["rmse"])
        b = np.array(results[ds]["booster"]["rmse"])
        if len(v) == 0:
            continue

        gaps = (v - b) / v * 100
        mean_gap = np.mean(gaps)
        dataset_gaps.append(mean_gap)
        if mean_gap > 0:
            win_count += 1

        t_stat, p_two = stats.ttest_rel(v, b)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        sig = "**" if p_one < 0.01 else "*" if p_one < 0.05 else ""

        print(
            f"  {ds:<12} | {np.mean(v):>10.4f} | {np.mean(b):>10.4f} | "
            f"{mean_gap:>+7.2f}% | {p_one:>8.4f} | {sig:>5}"
        )

    n_ds = len(dataset_gaps)
    dataset_gaps = np.array(dataset_gaps)
    print("-" * len(header))
    print(f"  Win rate: {win_count}/{n_ds}")
    print(f"  Mean improvement: {np.mean(dataset_gaps):+.2f}%")

    # Overall tests
    if n_ds >= 2:
        w_stat, w_p = stats.wilcoxon(dataset_gaps, alternative="greater")
        print(f"\n  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.6f}")

    # LaTeX
    print(f"\n{'='*80}")
    print("LATEX TABLE")
    print("=" * 80)
    print()
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Vanilla Linear vs TSB-Lite (5-seed mean). ** $p<0.01$, * $p<0.05$.}")
    print(r"\label{tab:main}")
    print(r"\begin{tabular}{lrrrrrrl}")
    print(r"\toprule")
    print(r"Dataset & V\_RMSE & B\_RMSE & $\Delta$(\%) & V\_MAE & B\_MAE & B\_MDA(\%) & Sig. \\")
    print(r"\midrule")

    for ds in ALL_DATASETS:
        v_rmse = np.array(results[ds]["vanilla"]["rmse"])
        b_rmse = np.array(results[ds]["booster"]["rmse"])
        if len(v_rmse) == 0:
            continue
        v_mae = np.array(results[ds]["vanilla"]["mae"])
        b_mae = np.array(results[ds]["booster"]["mae"])
        b_mda = np.array(results[ds]["booster"]["mda"])

        gap = np.mean((v_rmse - b_rmse) / v_rmse * 100)
        t_stat, p_two = stats.ttest_rel(v_rmse, b_rmse)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        sig = r"$^{**}$" if p_one < 0.01 else r"$^{*}$" if p_one < 0.05 else ""
        ds_latex = ds.replace("_", r"\_")

        print(
            f"{ds_latex} & {np.mean(v_rmse):.4f} & {np.mean(b_rmse):.4f} & "
            f"{gap:+.2f} & {np.mean(v_mae):.4f} & {np.mean(b_mae):.4f} & "
            f"{np.mean(b_mda)*100:.1f} & {sig} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

def plot_improvement_bars(results):
    """Bar chart: RMSE improvement % per dataset."""
    datasets, means, stds, sigs = [], [], [], []
    for ds in ALL_DATASETS:
        v = np.array(results[ds]["vanilla"]["rmse"])
        b = np.array(results[ds]["booster"]["rmse"])
        if len(v) == 0:
            continue
        datasets.append(ds)
        gaps = (v - b) / v * 100
        means.append(np.mean(gaps))
        stds.append(np.std(gaps, ddof=1))
        t_stat, p_two = stats.ttest_rel(v, b)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        sigs.append("**" if p_one < 0.01 else "*" if p_one < 0.05 else "")

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    colors = ["#43A047" if m > 0 else "#E53935" for m in means]
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.7,
                  error_kw={"linewidth": 1, "color": "#424242"})

    for i, (bar, mean, std, sig) in enumerate(zip(bars, means, stds, sigs)):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + std + 0.5,
                f"{mean:+.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if sig:
            ax.text(bar.get_x() + bar.get_width() / 2, h + std + 2.0,
                    sig, ha="center", va="bottom", fontsize=9, color="#1565C0")

    ax.axhline(y=0, color="#9E9E9E", linewidth=0.8)

    n_fx = len([d for d in datasets if d in FX_DATASETS])
    if 0 < n_fx < len(datasets):
        ax.axvline(x=n_fx - 0.5, color="#424242", linewidth=1.5, linestyle="--", alpha=0.5)
        ax.text(n_fx / 2 - 0.5, ax.get_ylim()[1] * 0.92, "FX",
                ha="center", fontsize=10, fontweight="bold", color="#1565C0")
        ax.text(n_fx + (len(datasets) - n_fx) / 2 - 0.5, ax.get_ylim()[1] * 0.92, "Stocks",
                ha="center", fontsize=10, fontweight="bold", color="#C62828")

    overall = np.mean(means)
    ax.axhline(y=overall, color="#1565C0", linewidth=1, linestyle=":", alpha=0.7)
    ax.text(len(datasets) - 0.5, overall + 0.5,
            f"avg: {overall:+.1f}%", ha="right", fontsize=9, color="#1565C0", fontstyle="italic")

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("RMSE Improvement (%)", fontsize=11)
    ax.set_title("TSB-Lite vs Vanilla Linear: RMSE Improvement (5 seeds)\n** p < 0.01",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "rmse_improvement.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: rmse_improvement.png")


def plot_mae_improvement(results):
    """Bar chart: MAE improvement % per dataset."""
    datasets, means, stds = [], [], []
    for ds in ALL_DATASETS:
        v = np.array(results[ds]["vanilla"]["mae"])
        b = np.array(results[ds]["booster"]["mae"])
        if len(v) == 0:
            continue
        datasets.append(ds)
        gaps = (v - b) / v * 100
        means.append(np.mean(gaps))
        stds.append(np.std(gaps, ddof=1))

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    colors = ["#43A047" if m > 0 else "#E53935" for m in means]
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.7,
                  error_kw={"linewidth": 1, "color": "#424242"})
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.5,
                f"{mean:+.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(y=0, color="#9E9E9E", linewidth=0.8)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("MAE Improvement (%)", fontsize=11)
    ax.set_title("TSB-Lite vs Vanilla: MAE Improvement (5 seeds)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "mae_improvement.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: mae_improvement.png")


def plot_mda_comparison(results):
    """Grouped bar: Vanilla vs Booster MDA."""
    datasets, v_means, b_means, v_stds, b_stds = [], [], [], [], []
    for ds in ALL_DATASETS:
        v = np.array(results[ds]["vanilla"]["mda"])
        b = np.array(results[ds]["booster"]["mda"])
        if len(v) == 0:
            continue
        datasets.append(ds)
        v_means.append(np.mean(v) * 100)
        b_means.append(np.mean(b) * 100)
        v_stds.append(np.std(v, ddof=1) * 100)
        b_stds.append(np.std(b, ddof=1) * 100)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    w = 0.35
    ax.bar(x - w / 2, v_means, w, yerr=v_stds, capsize=2, label="Vanilla",
           color="#BDBDBD", edgecolor="white", error_kw={"linewidth": 0.8})
    ax.bar(x + w / 2, b_means, w, yerr=b_stds, capsize=2, label="TSB-Lite",
           color="#43A047", edgecolor="white", error_kw={"linewidth": 0.8})

    for i, (bm, bs) in enumerate(zip(b_means, b_stds)):
        ax.text(i + w / 2, bm + bs + 0.3, f"{bm:.1f}%",
                ha="center", va="bottom", fontsize=7, fontweight="bold", color="#2E7D32")

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("MDA (%)", fontsize=11)
    ax.set_title("Mean Directional Accuracy: Vanilla vs TSB-Lite (5 seeds)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(50, max(b_means) + 8)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "mda_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: mda_comparison.png")


def generate_figures(results):
    print("\nGenerating figures...")
    plot_improvement_bars(results)
    plot_mae_improvement(results)
    plot_mda_comparison(results)
    print(f"Figures saved to {FIGURES_DIR}/")


# ──────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No saved results at {RESULTS_PATH}. Run without --analyze first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = run_experiments()

    analyze(results)
    generate_figures(results)
