"""
Out-of-Sample Backtest with Date-Based Split for TS-Booster Paper.

Trains on historical data up to a cutoff date, then evaluates on truly
unseen recent data. Compares Linear (vanilla) vs Linear+TSB across
16 datasets × 5 seeds.

Usage:
  uv run python experiments/oos_backtest.py                          # Full run (fetch + train + eval)
  uv run python experiments/oos_backtest.py --analyze                # Analyze saved results
  uv run python experiments/oos_backtest.py --skip-fetch             # Skip data fetching
  uv run python experiments/oos_backtest.py --train-end 2023-12-31 --val-end 2024-06-30
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

from src.models.ts_booster import TS_Booster
from src.models.layers import LinearBackbone
from src.engine.trainer import train_v11
from src.configs.model_config import BEST_STAGE3_CONFIG
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

SEEDS = [42, 123, 456, 789, 2024]
SEQ_LEN = 30
PRED_LEN = 3

RESULTS_PATH = Path("experiments/comprehensive_results/oos_backtest/oos_backtest_results.json")
FIGURES_DIR = Path("experiments/comprehensive_results/oos_backtest/figures")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fetch_all_data():
    """Fetch latest FX and stock data."""
    print("Fetching latest data...")
    from src.preprocess.get_fx import fetch_fx, TARGETS_TWD, TARGETS_INTL, _patch_headers
    from src.preprocess.get_stock import fetch_stock, TARGETS

    os_mod = __import__('os')
    os_mod.makedirs('dataset', exist_ok=True)

    for t in TARGETS_TWD:
        fetch_fx(t, "TWD", f"dataset/{t}_TWD.csv")
    for base, quote in TARGETS_INTL:
        fetch_fx(base, quote, f"dataset/{base}_{quote}.csv")
    for t in TARGETS:
        fetch_stock(t)

    print("Data fetch complete.\n")


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


def run_experiments(args):
    """Run OOS backtest across all datasets × 5 seeds."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Train end: {args.train_end}, Val end: {args.val_end}")
    print(f"Test: {args.val_end} ~ latest\n")

    config = BEST_STAGE3_CONFIG.copy()
    config['epochs'] = 100
    config['input_channels'] = 2

    # Load existing results for resume
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    # Store metadata
    results["_meta"] = {
        "train_end_date": args.train_end,
        "val_end_date": args.val_end,
        "seeds": SEEDS,
        "seq_len": SEQ_LEN,
        "pred_len": PRED_LEN,
    }

    for ds in ALL_DATASETS:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds}")
        print(f"{'='*60}")

        # Skip if already complete
        if ds in results and isinstance(results[ds], dict) and "_complete" in results[ds]:
            print(f"  SKIP (already complete)")
            continue

        path = f"dataset/{ds}.csv"
        if not Path(path).exists():
            print(f"  SKIP (file not found)")
            continue

        df = pd.read_csv(path)

        # Check if dataset has enough data after val_end
        dates = pd.to_datetime(df["Date"] if "Date" in df.columns else df.index)
        n_test_rows = (dates > args.val_end).sum()
        print(f"  Total rows: {len(df)}, Test rows (after {args.val_end}): {n_test_rows}")

        if n_test_rows < SEQ_LEN + PRED_LEN:
            print(f"  SKIP (not enough test data: need {SEQ_LEN + PRED_LEN}, have {n_test_rows})")
            continue

        train_loader, val_loader, test_loader, scaler = prepare_data(
            df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32,
            train_end_date=args.train_end, val_end_date=args.val_end,
        )

        if len(test_loader.dataset) == 0:
            print(f"  SKIP (0 test samples after sequencing)")
            continue

        vanilla_metrics = {"rmse": [], "mae": [], "mda": []}
        boosted_metrics = {"rmse": [], "mae": [], "mda": []}

        for seed in SEEDS:
            t0 = time.time()

            # Vanilla Linear
            set_seed(seed)
            vanilla = LinearBackbone(SEQ_LEN, 2, PRED_LEN)
            vanilla = train_v11(train_loader, val_loader, test_loader,
                               device=device, horizon=PRED_LEN, config=config,
                               seq_len=SEQ_LEN, verbose=False, model=vanilla)
            v_met = evaluate(vanilla, test_loader, scaler, device)
            for k in vanilla_metrics:
                vanilla_metrics[k].append(v_met[k])

            # Linear + TSB
            set_seed(seed)
            boosted = TS_Booster(
                base_model=LinearBackbone(SEQ_LEN, 2, PRED_LEN),
                seq_len=SEQ_LEN, pred_len=PRED_LEN, input_channels=2,
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
                               device=device, horizon=PRED_LEN, config=config,
                               seq_len=SEQ_LEN, verbose=False, model=boosted)
            b_met = evaluate(boosted, test_loader, scaler, device)
            for k in boosted_metrics:
                boosted_metrics[k].append(b_met[k])

            del vanilla, boosted
            torch.cuda.empty_cache()

            elapsed = time.time() - t0
            print(f"  Seed {seed}: V_RMSE={v_met['rmse']:.4f} B_RMSE={b_met['rmse']:.4f} "
                  f"gap={((v_met['rmse']-b_met['rmse'])/v_met['rmse']*100):+.1f}% ({elapsed:.0f}s)")

        results[ds] = {
            "vanilla": vanilla_metrics,
            "booster": boosted_metrics,
            "test_samples": len(test_loader.dataset),
            "_complete": True,
        }

        # Incremental save
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    return results


def analyze(results):
    """Print summary table and comparison."""
    print("\n" + "=" * 100)
    print("OOS BACKTEST RESULTS (Date-Based Split)")
    print("=" * 100)

    meta = results.get("_meta", {})
    print(f"  Train end: {meta.get('train_end_date', '?')}")
    print(f"  Val end:   {meta.get('val_end_date', '?')}")
    print(f"  Test:      after val_end ~ latest data")

    header = f"{'Dataset':<12} | {'Samples':>7} | {'V_RMSE':>8} | {'B_RMSE':>8} | {'Impr%':>7} | {'V_MAE':>8} | {'B_MAE':>8} | {'V_MDA':>7} | {'B_MDA':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    wins = {"rmse": 0, "mae": 0, "mda": 0}
    total = 0
    all_gaps = []

    for ds in ALL_DATASETS:
        if ds not in results or not isinstance(results[ds], dict) or "vanilla" not in results[ds]:
            continue

        total += 1
        entry = results[ds]
        n_samples = entry.get("test_samples", "?")
        v = entry["vanilla"]
        b = entry["booster"]

        v_rmse = np.mean(v["rmse"])
        b_rmse = np.mean(b["rmse"])
        v_mae = np.mean(v["mae"])
        b_mae = np.mean(b["mae"])
        v_mda = np.mean(v["mda"])
        b_mda = np.mean(b["mda"])

        gap = (v_rmse - b_rmse) / v_rmse * 100
        all_gaps.append(gap)

        if b_rmse < v_rmse:
            wins["rmse"] += 1
        if b_mae < v_mae:
            wins["mae"] += 1
        if b_mda > v_mda:
            wins["mda"] += 1

        tag = "W" if gap > 0 else "L"
        print(f"  {ds:<12} | {n_samples:>7} | {v_rmse:>8.4f} | {b_rmse:>8.4f} | {gap:>+6.1f}% {tag} | "
              f"{v_mae:>8.4f} | {b_mae:>8.4f} | {v_mda:>7.4f} | {b_mda:>7.4f}")

    print("-" * len(header))
    if total > 0:
        print(f"  RMSE wins: {wins['rmse']}/{total} | MAE wins: {wins['mae']}/{total} | MDA wins: {wins['mda']}/{total}")
        print(f"  Avg RMSE improvement: {np.mean(all_gaps):+.2f}%")

    # Paired t-test on RMSE
    print(f"\n{'='*60}")
    print("PAIRED T-TEST (per-dataset, 5 seeds)")
    print("=" * 60)
    sig_count = 0
    for ds in ALL_DATASETS:
        if ds not in results or not isinstance(results[ds], dict) or "vanilla" not in results[ds]:
            continue
        v_list = results[ds]["vanilla"]["rmse"]
        b_list = results[ds]["booster"]["rmse"]
        if len(v_list) < 2 or len(b_list) < 2:
            continue
        t_stat, p_two = stats.ttest_ind(b_list, v_list)
        p_val = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        gap = (np.mean(v_list) - np.mean(b_list)) / np.mean(v_list) * 100
        if p_val < 0.05 and gap > 0:
            sig_count += 1
        print(f"  {ds:<12}: gap={gap:+6.2f}%, p={p_val:.4f} {sig}")
    print(f"  Significant wins (p<0.05): {sig_count}/{total}")


def plot_comparison(results):
    """Bar chart: RMSE improvement per dataset."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets, gaps, colors = [], [], []
    for ds in ALL_DATASETS:
        if ds not in results or not isinstance(results[ds], dict) or "vanilla" not in results[ds]:
            continue
        v_rmse = np.mean(results[ds]["vanilla"]["rmse"])
        b_rmse = np.mean(results[ds]["booster"]["rmse"])
        gap = (v_rmse - b_rmse) / v_rmse * 100
        datasets.append(ds)
        gaps.append(gap)
        colors.append('#43A047' if gap > 0 else '#E53935')

    if not datasets:
        print("  No data for figures")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(datasets))
    bars = ax.barh(y, gaps, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        w = bar.get_width()
        label = f'{gap:+.1f}%'
        if w >= 0:
            ax.text(w + 0.3, i, label, ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            ax.text(w - 0.3, i, label, ha='right', va='center', fontsize=9, fontweight='bold')

    ax.axvline(x=0, color='#424242', linewidth=1)

    meta = results.get("_meta", {})
    test_start = meta.get("val_end_date", "?")
    wins = sum(1 for g in gaps if g > 0)

    ax.set_xlabel('RMSE Improvement (%)', fontsize=11)
    ax.set_title(f'OOS Backtest: Linear+TSB vs Linear (test: {test_start}~latest, {wins}/{len(gaps)} wins)',
                 fontsize=13, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'oos_rmse_improvement.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: oos_rmse_improvement.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOS backtest with date-based split")
    parser.add_argument("--analyze", action="store_true", help="Analyze saved results only")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetching")
    parser.add_argument("--train-end", default="2023-12-31", help="Train cutoff date (default: 2023-12-31)")
    parser.add_argument("--val-end", default="2024-06-30", help="Val cutoff date (default: 2024-06-30)")
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No saved results at {RESULTS_PATH}")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        analyze(results)
        plot_comparison(results)
        sys.exit(0)

    if not args.skip_fetch:
        fetch_all_data()

    results = run_experiments(args)
    analyze(results)
    plot_comparison(results)
