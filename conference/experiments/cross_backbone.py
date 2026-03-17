"""
Cross-Backbone Validation: 4 backbones x 16 datasets x 5 seeds.

Tests TSB-Lite with Linear, LSTM, GRU, Transformer backbones.

Usage:
  cd conference
  uv run python experiments/cross_backbone.py              # Full run
  uv run python experiments/cross_backbone.py --analyze    # Analyze saved results
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
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(str(Path(__file__).parent.parent))

from src.models.layers import LinearBackbone, LSTMBackbone, GRUBackbone, TransformerBackbone
from src.models.ts_booster import TSBoosterLite
from src.engine.trainer import train_booster, train_baseline
from src.configs.config import CONFIG
from src.utils.data_loader import prepare_data

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

BACKBONES = {
    "Linear": lambda sl, ic, pl: LinearBackbone(sl, ic, pl),
    "LSTM": lambda sl, ic, pl: LSTMBackbone(sl, ic, pl, hidden_dim=64, num_layers=2),
    "GRU": lambda sl, ic, pl: GRUBackbone(sl, ic, pl, hidden_dim=64, num_layers=2),
    "Transformer": lambda sl, ic, pl: TransformerBackbone(sl, ic, pl, hidden_dim=64, num_layers=2),
}

SEEDS = [42, 123, 456, 789, 2024]
DATASET_DIR = Path("../dataset")
RESULTS_PATH = Path("results/cross_backbone/results.json")
FIGURES_DIR = Path("results/cross_backbone/figures")


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


def _is_complete(results, ds):
    if ds not in results:
        return False
    for bb in BACKBONES:
        if bb not in results[ds]:
            return False
        entry = results[ds][bb]
        for metric in ["rmse", "mae", "mda"]:
            v = entry.get(f"vanilla_{metric}")
            b = entry.get(f"boosted_{metric}")
            if not (isinstance(v, list) and len(v) >= len(SEEDS)):
                return False
            if not (isinstance(b, list) and len(b) >= len(SEEDS)):
                return False
    return True


def run_cross_backbone(dataset_name, device, config):
    """Test all 4 backbones x 5 seeds on one dataset."""
    seq_len = config["seq_len"]
    pred_len = config["pred_len"]

    path = DATASET_DIR / f"{dataset_name}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=seq_len, horizon=pred_len, batch_size=config["batch_size"]
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
            vanilla = train_baseline(
                train_loader, val_loader, device, vanilla,
                num_epochs=config["epochs"], lr=config["lr"], verbose=False,
            )
            v_m = evaluate(vanilla, test_loader, scaler, device)
            for k in vanilla_metrics:
                vanilla_metrics[k].append(v_m[k])

            # TSB-Lite
            set_seed(seed)
            booster = TSBoosterLite(
                base_model=bb_fn(seq_len, 2, pred_len),
                seq_len=seq_len, pred_len=pred_len, input_channels=2,
                decomp_kernel=config["decomp_kernel"],
                expert_kernels=config["expert_kernels"],
                expert_hidden_dim=config["expert_hidden_dim"],
                dropout_p=config["dropout"],
            )
            booster = train_booster(
                train_loader, val_loader, device, booster, config, verbose=False,
            )
            b_m = evaluate(booster, test_loader, scaler, device)
            for k in boosted_metrics:
                boosted_metrics[k].append(b_m[k])

            del vanilla, booster
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        v_mean = np.mean(vanilla_metrics["rmse"])
        b_mean = np.mean(boosted_metrics["rmse"])
        gap = (v_mean - b_mean) / v_mean * 100
        win = "WIN" if gap > 0 else "LOSS"

        results[bb_name] = {
            "vanilla_rmse": vanilla_metrics["rmse"],
            "boosted_rmse": boosted_metrics["rmse"],
            "vanilla_mae": vanilla_metrics["mae"],
            "boosted_mae": boosted_metrics["mae"],
            "vanilla_mda": vanilla_metrics["mda"],
            "boosted_mda": boosted_metrics["mda"],
        }

        print(f"  {dataset_name:12s} | {bb_name:12s} | "
              f"V: {v_mean:.4f} B: {b_mean:.4f} {gap:+.2f}% {win}  ({elapsed:.0f}s)")

    return results


# ──────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────

def analyze(all_results):
    bb_names = list(BACKBONES.keys())

    print(f"\n{'='*100}")
    print("CROSS-BACKBONE SUMMARY (5-seed mean)")
    print(f"{'='*100}")

    bb_wins = {bb: 0 for bb in bb_names}
    bb_gaps = {bb: [] for bb in bb_names}

    for ds in ALL_DATASETS:
        if ds not in all_results:
            continue
        for bb in bb_names:
            if bb in all_results[ds]:
                entry = all_results[ds][bb]
                v_mean = np.mean(entry["vanilla_rmse"])
                b_mean = np.mean(entry["boosted_rmse"])
                gap = (v_mean - b_mean) / v_mean * 100
                bb_gaps[bb].append(gap)
                if gap > 0:
                    bb_wins[bb] += 1

    total_all = sum(bb_wins.values())
    total_possible = sum(len(bb_gaps[bb]) for bb in bb_names)

    for bb in bb_names:
        total = len(bb_gaps[bb])
        avg = np.mean(bb_gaps[bb]) if bb_gaps[bb] else 0
        print(f"  {bb:12s}: {bb_wins[bb]}/{total} wins, avg improvement: {avg:+.2f}%")
    print(f"\n  TOTAL: {total_all}/{total_possible} ({total_all/total_possible*100:.1f}%)")

    # LaTeX
    print(f"\n{'='*80}")
    print("LATEX TABLE")
    print("=" * 80)
    print()
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Cross-backbone: RMSE improvement (\%) of TSB-Lite (5-seed mean).}")
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
                v_mean = np.mean(entry["vanilla_rmse"])
                b_mean = np.mean(entry["boosted_rmse"])
                gap = (v_mean - b_mean) / v_mean * 100
                cells.append(f"+{gap:.1f}\\%" if gap > 0 else f"{gap:.1f}\\%")
            else:
                cells.append("--")
        print(f"{ds_latex} & " + " & ".join(cells) + r" \\")

    print(r"\midrule")
    win_cells = [f"{bb_wins[bb]}/{len(bb_gaps[bb])}" for bb in bb_names]
    avg_cells = [f"+{np.mean(bb_gaps[bb]):.1f}\\%" if bb_gaps[bb] else "--" for bb in bb_names]
    print(r"\textit{Wins} & " + " & ".join(win_cells) + r" \\")
    print(r"\textit{Avg.} & " + " & ".join(avg_cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

def plot_heatmap(all_results):
    """Heatmap: RMSE improvement % across datasets x backbones."""
    bb_names = list(BACKBONES.keys())
    datasets = [ds for ds in ALL_DATASETS if ds in all_results]

    matrix = np.zeros((len(datasets), len(bb_names)))
    for i, ds in enumerate(datasets):
        for j, bb in enumerate(bb_names):
            if bb in all_results[ds]:
                entry = all_results[ds][bb]
                v_mean = np.mean(entry["vanilla_rmse"])
                b_mean = np.mean(entry["boosted_rmse"])
                matrix[i, j] = (v_mean - b_mean) / v_mean * 100

    fig, ax = plt.subplots(figsize=(10, 12))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#EF5350", "#FFCDD2", "#FFFFFF", "#C8E6C9", "#43A047"], N=256
    )
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(bb_names)))
    ax.set_xticklabels(bb_names, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10)

    for i in range(len(datasets)):
        for j in range(len(bb_names)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_title("Cross-Backbone: TSB-Lite RMSE Improvement (%)\n(5-seed mean)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.08, label="Improvement %")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "cross_backbone_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: cross_backbone_heatmap.png")


def generate_figures(all_results):
    print("\nGenerating figures...")
    plot_heatmap(all_results)
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
            all_results = json.load(f)
        analyze(all_results)
        generate_figures(all_results)
        sys.exit(0)

    config = CONFIG.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for i, ds in enumerate(ALL_DATASETS, 1):
        print(f"\n[{i}/{len(ALL_DATASETS)}] {ds}")

        if _is_complete(all_results, ds):
            print("  SKIP (already complete)")
            continue

        result = run_cross_backbone(ds, device, config)
        if result:
            all_results[ds] = result
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2)

    analyze(all_results)
    generate_figures(all_results)
