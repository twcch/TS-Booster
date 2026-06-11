"""
Phase 11: New SOTA DL Baselines (Autoformer, FEDformer, ETSformer, LightTS, TimesNet)

Extends experiments/sota_baselines.py with 5 additional recent/standard
top-venue models (see src/models/sota_baselines_v2.py), trained the same
way (MSE loss + Adam, 100 epochs, early stopping via ReduceLROnPlateau)
across 16 datasets x 5 seeds.

LR: fixed per-model, taken from each model's original paper/repo default
(no grid search this round -- 5 models x 16 datasets x 5 seeds is already
~400 runs; LR tuning would multiply that several-fold). If a particular
model's results look anomalously bad, re-use the revision/phase10_lr_tuning
grid-search pattern for just that model.

Usage:
  uv run python revision/phase11_new_sota/run_new_sota.py                              # Full run (resumable)
  uv run python revision/phase11_new_sota/run_new_sota.py --analyze                    # Tables from saved JSON
  uv run python revision/phase11_new_sota/run_new_sota.py --models Autoformer TimesNet # Only specific models
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

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.sota_baselines_v2 import create_sota_model_v2, SOTA_MODELS_V2
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

EXISTING_SOTA_PATH = Path("experiments/comprehensive_results/sota_baselines/sota_baselines_results.json")
RESULTS_PATH = Path("revision/phase11_new_sota/new_sota_results.json")

SEEDS = [42, 123, 456, 789, 2024]
SEQ_LEN = 30
PRED_LEN = 3
INPUT_CHANNELS = 2
NEW_MODEL_NAMES = list(SOTA_MODELS_V2.keys())  # Autoformer, FEDformer, ETSformer, LightTS, TimesNet

# Fixed per-model LRs from original papers/repos (short-sequence-friendly defaults)
NEW_MODEL_LR = {
    "Autoformer": 1e-4,   # official repo default (ETT)
    "FEDformer": 1e-4,    # official repo default (ETT)
    "ETSformer": 1e-3,    # official repo default
    "LightTS": 1e-3,      # official repo default
    "TimesNet": 1e-3,     # official repo default (short-term forecasting)
}

# All methods shown in the comparison table
COMPARISON_METHODS = ["Linear", "Linear+TSB", "DLinear", "PatchTST", "iTransformer"] + NEW_MODEL_NAMES


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


def load_existing_results():
    """Load Linear / Linear+TSB / DLinear / PatchTST / iTransformer from the
    existing 5-seed SOTA results JSON (read-only)."""
    if not EXISTING_SOTA_PATH.exists():
        print(f"WARNING: {EXISTING_SOTA_PATH} not found. Existing methods will be missing.")
        return {}

    with open(EXISTING_SOTA_PATH) as f:
        raw = json.load(f)

    existing = {}
    for ds in ALL_DATASETS:
        if ds not in raw:
            continue
        existing[ds] = {}
        for m in ["Linear", "Linear+TSB", "DLinear", "PatchTST", "iTransformer"]:
            if m in raw[ds]:
                existing[ds][m] = raw[ds][m]
    return existing


def _get_metric_mean(results, ds, method, metric="rmse"):
    val = results.get(ds, {}).get(method)
    if val is None:
        return None
    if isinstance(val, dict):
        arr = val.get(metric, [])
        return float(np.mean(arr)) if arr else None
    return None


def _get_metric_list(results, ds, method, metric="rmse"):
    val = results.get(ds, {}).get(method)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get(metric, None)
    return None


def run_single_model(model_name, ds, seed, device):
    """Train one new SOTA model for one dataset/seed. Returns metrics dict."""
    path = f"dataset/{ds}.csv"
    if not Path(path).exists():
        return None

    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32
    )

    set_seed(seed)
    model = create_sota_model_v2(model_name, seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=INPUT_CHANNELS)
    lr = NEW_MODEL_LR.get(model_name, 1e-3)
    model = train_baseline(
        train_loader, val_loader, test_loader, device=device,
        model=model, num_epochs=100, lr=lr, verbose=False
    )
    metrics = evaluate(model, test_loader, scaler, device)

    del model
    torch.cuda.empty_cache()
    return metrics


def run_experiments(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    models_to_run = args.models if args.models else NEW_MODEL_NAMES
    for m in models_to_run:
        if m not in SOTA_MODELS_V2:
            print(f"ERROR: Unknown model '{m}'. Choose from {NEW_MODEL_NAMES}")
            sys.exit(1)

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    total_runs = len(models_to_run) * len(ALL_DATASETS) * len(SEEDS)
    run_count = 0
    total_start = time.time()

    for model_name in models_to_run:
        print(f"\n{'#'*70}")
        print(f"  Model: {model_name}  (LR={NEW_MODEL_LR[model_name]:.0e})")
        print(f"{'#'*70}")

        dummy = create_sota_model_v2(model_name, SEQ_LEN, PRED_LEN, INPUT_CHANNELS)
        n_params = sum(p.numel() for p in dummy.parameters())
        print(f"  Parameters: {n_params:,}")
        del dummy

        for ds in ALL_DATASETS:
            print(f"\n  {'='*50}")
            print(f"  Dataset: {ds}")

            if ds not in results:
                results[ds] = {}

            existing_rmse = _get_metric_list(results, ds, model_name, "rmse")
            existing_mae = _get_metric_list(results, ds, model_name, "mae")
            if existing_rmse and len(existing_rmse) >= len(SEEDS) and existing_mae and len(existing_mae) >= len(SEEDS):
                run_count += len(SEEDS)
                print(f"  SKIP (already have {len(existing_rmse)} seeds with all metrics)")
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
                print(f"  {model_name}: RMSE={np.mean(seed_metrics['rmse']):.4f} "
                      f"MAE={np.mean(seed_metrics['mae']):.4f} MDA={np.mean(seed_metrics['mda']):.4f}")

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
    existing = load_existing_results()
    # Merge: existing methods (Linear, Linear+TSB, DLinear, PatchTST, iTransformer)
    # come from the read-only existing results file; new models come from
    # the phase11 results file.
    merged = {}
    for ds in ALL_DATASETS:
        merged[ds] = {}
        merged[ds].update(existing.get(ds, {}))
        merged[ds].update(results.get(ds, {}))

    methods = COMPARISON_METHODS

    print("\n" + "=" * 160)
    print("PHASE 11: NEW SOTA BASELINES BENCHMARK (5-seed mean RMSE)")
    print("=" * 160)

    header = f"{'Dataset':<12}"
    for m in methods:
        header += f" | {m:>12}"
    print(f"\n{header}")
    print("-" * len(header))

    for ds in ALL_DATASETS:
        line = f"  {ds:<12}"
        for m in methods:
            val = _get_metric_mean(merged, ds, m)
            line += f" | {val:>12.4f}" if val is not None else f" | {'N/A':>12}"
        print(line)

    # --- TSB vs each new model: win counts + paired t-tests ---
    print(f"\n{'='*80}")
    print("LINEAR+TSB vs NEW SOTA MODELS (RMSE)")
    print("=" * 80)
    for m in NEW_MODEL_NAMES:
        wins, total, gaps = 0, 0, []
        sig_wins = 0
        for ds in ALL_DATASETS:
            tsb_list = _get_metric_list(merged, ds, "Linear+TSB")
            other_list = _get_metric_list(merged, ds, m)
            if tsb_list is None or other_list is None:
                continue
            total += 1
            tsb_mean, other_mean = np.mean(tsb_list), np.mean(other_list)
            gap = (other_mean - tsb_mean) / other_mean * 100
            gaps.append(gap)
            if tsb_mean < other_mean:
                wins += 1
            if len(tsb_list) >= 2 and len(other_list) >= 2:
                t_stat, p_two = stats.ttest_ind(tsb_list, other_list)
                p_val = p_two / 2 if t_stat < 0 else 1 - p_two / 2
                if p_val < 0.05 and gap > 0:
                    sig_wins += 1
        if total > 0:
            print(f"  vs {m:<12}: {wins}/{total} wins ({sig_wins} significant, p<0.05), avg gap = {np.mean(gaps):+.2f}%")

    # --- Category breakdown ---
    print(f"\n{'='*80}")
    print("CATEGORY BREAKDOWN (mean RMSE)")
    print("=" * 80)
    for cat_name, cat_datasets in CATEGORIES.items():
        print(f"\n  {cat_name}:")
        line = "    "
        for m in methods:
            vals = [_get_metric_mean(merged, ds, m) for ds in cat_datasets]
            vals = [v for v in vals if v is not None]
            if vals:
                line += f"{m}={np.mean(vals):.4f}  "
        print(line)

    # --- Average rank ---
    print(f"\n{'='*80}")
    print("AVERAGE RANK ACROSS DATASETS")
    print("=" * 80)
    ranks_per_method = {m: [] for m in methods}
    for ds in ALL_DATASETS:
        available = {m: _get_metric_mean(merged, ds, m) for m in methods}
        available = {k: v for k, v in available.items() if v is not None}
        if not available:
            continue
        for rank, m in enumerate(sorted(available, key=lambda k: available[k]), 1):
            ranks_per_method[m].append(rank)
    for m in methods:
        if ranks_per_method[m]:
            print(f"  {m:<14}: avg rank = {np.mean(ranks_per_method[m]):.2f}")

    # --- Average RMSE row ---
    print(f"\n{'='*80}")
    print("AVERAGE RMSE / MAE / MDA ACROSS DATASETS")
    print("=" * 80)
    for metric in ["rmse", "mae", "mda"]:
        line = f"  {metric.upper():<6}: "
        for m in methods:
            vals = [_get_metric_mean(merged, ds, m, metric) for ds in ALL_DATASETS]
            vals = [v for v in vals if v is not None]
            if vals:
                line += f"{m}={np.mean(vals):.4f}  "
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="Only analyze saved results")
    parser.add_argument("--models", nargs="+", choices=NEW_MODEL_NAMES, help="Run only these models")
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No results found at {RESULTS_PATH}")
            return
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        analyze(results)
    else:
        results = run_experiments(args)
        analyze(results)


if __name__ == "__main__":
    main()
