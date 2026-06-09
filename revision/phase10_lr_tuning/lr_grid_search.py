"""
Phase 10: LR Grid Search for SOTA DL Baselines

Problem:
  Original SOTA experiment used learning rates from each model's official paper,
  which were tuned for long-sequence benchmarks (ETT, Weather, etc.).
  For our 30→3 short-window financial volatility task, these may be suboptimal,
  potentially understating baseline performance (unfair comparison).

Fix:
  For each model, grid search LR on the validation set (3 seeds for speed),
  then re-run full 5-seed evaluation with the best LR per dataset.

LR grid: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 3e-3]
Metric: validation RMSE (mean over 3 seeds)

Usage:
  # Step 1: grid search (finds best LR per model per dataset)
  uv run python revision/phase10_lr_tuning/lr_grid_search.py --step search

  # Step 2: full 5-seed re-run with best LRs
  uv run python revision/phase10_lr_tuning/lr_grid_search.py --step eval

  # Step 3: analyze and compare with original
  uv run python revision/phase10_lr_tuning/lr_grid_search.py --step analyze
"""
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import mean_squared_error

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.sota_baselines import DLinear, PatchTST, iTransformer
from src.engine.trainer import train_baseline
from src.utils.data_loader import prepare_data

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

LR_GRID    = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 3e-3]
SEARCH_SEEDS = [42, 123, 456]
EVAL_SEEDS   = [42, 123, 456, 789, 2024]
SEQ_LEN  = 30
PRED_LEN = 3
N_FEAT   = 2

ORIG_LR = {"DLinear": 1e-4, "PatchTST": 1e-4, "iTransformer": 5e-4}

OUT_DIR        = Path("revision/phase10_lr_tuning")
BEST_LR_PATH   = OUT_DIR / "best_lr.json"
EVAL_PATH      = OUT_DIR / "lr_tuned_results.json"
ORIG_SOTA_PATH = Path("experiments/comprehensive_results/sota_baselines/sota_baselines_results.json")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_model(name):
    if name == "DLinear":
        return DLinear(SEQ_LEN, PRED_LEN, N_FEAT)
    elif name == "PatchTST":
        return PatchTST(SEQ_LEN, PRED_LEN, N_FEAT, use_revin=True)
    elif name == "iTransformer":
        return iTransformer(SEQ_LEN, PRED_LEN, N_FEAT)
    raise ValueError(name)


def evaluate(model, loader, scaler, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["raw_input"].to(device)
            y = batch["target"].to(device)
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds   = np.concatenate(preds) * scaler.scale_[0] + scaler.mean_[0]
    targets = np.concatenate(targets) * scaler.scale_[0] + scaler.mean_[0]
    last_obs_all = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["raw_input"].to(device)
            last_obs_all.append((x[:, -1, 0].cpu().numpy() * scaler.scale_[0] + scaler.mean_[0]))
    last_obs = np.concatenate(last_obs_all)
    rmse = float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))
    mae  = float(np.mean(np.abs(targets.flatten() - preds.flatten())))
    mda  = float(np.mean(np.sign(preds - last_obs[:, None]) == np.sign(targets - last_obs[:, None])))
    return {"rmse": rmse, "mae": mae, "mda": mda}


# ── Step 1: Grid Search ────────────────────────────────────────────────────────

def run_search(device):
    best_lr = {}  # best_lr[model_name][dataset] = best_lr_value

    for model_name in ["DLinear", "PatchTST", "iTransformer"]:
        print(f"\n{'='*60}")
        print(f"LR Grid Search: {model_name}")
        print(f"{'='*60}")
        best_lr[model_name] = {}

        for ds in ALL_DATASETS:
            df = pd.read_csv(f"dataset/{ds}.csv")
            lr_val_rmse = {}

            for lr in LR_GRID:
                val_rmses = []
                for seed in SEARCH_SEEDS:
                    set_seed(seed)
                    train_loader, val_loader, test_loader, scaler = prepare_data(
                        df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32
                    )
                    model = make_model(model_name)
                    model = train_baseline(
                        train_loader, val_loader, test_loader,
                        device=device, model=model, num_epochs=100, lr=lr, verbose=False
                    )
                    m = evaluate(model, val_loader, scaler, device)
                    val_rmses.append(m["rmse"])
                    del model
                    torch.cuda.empty_cache()

                lr_val_rmse[lr] = float(np.mean(val_rmses))

            best = min(lr_val_rmse, key=lr_val_rmse.get)
            best_lr[model_name][ds] = best
            orig = ORIG_LR[model_name]
            print(f"  {ds:<14}  best={best:.0e}  orig={orig:.0e}  "
                  f"val_rmse: {lr_val_rmse[best]:.4f} vs {lr_val_rmse.get(orig, float('nan')):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_LR_PATH.write_text(json.dumps(best_lr, indent=2))
    print(f"\nBest LRs saved → {BEST_LR_PATH}")
    return best_lr


# ── Step 2: Full 5-seed Eval with Best LR ─────────────────────────────────────

def run_eval(device):
    if not BEST_LR_PATH.exists():
        print("Run --step search first.")
        return
    best_lr = json.loads(BEST_LR_PATH.read_text())
    results = {}

    for model_name in ["DLinear", "PatchTST", "iTransformer"]:
        print(f"\n{'='*60}")
        print(f"5-seed Eval with tuned LR: {model_name}")
        print(f"{'='*60}")

        for ds in ALL_DATASETS:
            if ds not in best_lr[model_name]:
                continue
            lr = best_lr[model_name][ds]
            df = pd.read_csv(f"dataset/{ds}.csv")
            seed_metrics = []

            for seed in EVAL_SEEDS:
                set_seed(seed)
                train_loader, val_loader, test_loader, scaler = prepare_data(
                    df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32
                )
                model = make_model(model_name)
                model = train_baseline(
                    train_loader, val_loader, test_loader,
                    device=device, model=model, num_epochs=100, lr=lr, verbose=False
                )
                m = evaluate(model, test_loader, scaler, device)
                seed_metrics.append(m)
                del model
                torch.cuda.empty_cache()

            if ds not in results:
                results[ds] = {}
            results[ds][model_name] = {
                "rmse": [m["rmse"] for m in seed_metrics],
                "mae":  [m["mae"]  for m in seed_metrics],
                "mda":  [m["mda"]  for m in seed_metrics],
                "best_lr": lr,
            }
            mean_r = np.mean([m["rmse"] for m in seed_metrics])
            print(f"  {ds:<14}  LR={lr:.0e}  RMSE={mean_r:.4f}")

    EVAL_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {EVAL_PATH}")
    return results


# ── Step 3: Analyze ────────────────────────────────────────────────────────────

def run_analyze():
    if not EVAL_PATH.exists() or not ORIG_SOTA_PATH.exists():
        print("Run search + eval first.")
        return

    tuned  = json.loads(EVAL_PATH.read_text())
    orig   = json.loads(ORIG_SOTA_PATH.read_text())
    tsb_data = json.loads(Path("experiments/comprehensive_results/main_experiment/statistical_test_results.json").read_text())

    print("\n" + "=" * 90)
    print("LR-Tuned vs Original SOTA  (mean RMSE, 5 seeds)")
    print("=" * 90)

    for model_name in ["DLinear", "PatchTST", "iTransformer"]:
        print(f"\n── {model_name} ──────────────────────────────────────────────")
        print(f"{'Dataset':<14} {'Orig RMSE':>12} {'Tuned RMSE':>12} {'Δ%':>8} {'TSB RMSE':>12} {'TSB wins?':>10}")
        print("-" * 75)

        improvements = []
        tsb_wins_orig = 0
        tsb_wins_tuned = 0

        for ds in ALL_DATASETS:
            if ds not in tuned or model_name not in tuned[ds]:
                continue
            t_rmse = float(np.mean(tuned[ds][model_name]["rmse"]))
            o_rmse = float(np.mean(orig[ds][model_name]["rmse"])) if ds in orig and model_name in orig[ds] else None
            tsb_r  = float(np.mean(tsb_data[ds]["booster"]["rmse"])) if ds in tsb_data else None

            delta = ((o_rmse - t_rmse) / o_rmse * 100) if o_rmse else 0
            improvements.append(delta)

            if tsb_r and tsb_r < o_rmse: tsb_wins_orig  += 1
            if tsb_r and tsb_r < t_rmse: tsb_wins_tuned += 1

            lr_str = f"(LR={tuned[ds][model_name]['best_lr']:.0e})"
            print(f"{ds:<14} {o_rmse:>12.4f} {t_rmse:>12.4f} {delta:>+7.1f}% {tsb_r:>12.4f}  "
                  f"{'✓' if tsb_r and tsb_r < t_rmse else '✗':>4} {lr_str}")

        n = sum(1 for ds in ALL_DATASETS if ds in tuned and model_name in tuned[ds])
        avg_impr = np.mean(improvements) if improvements else 0
        print(f"\n  Avg tuning gain: {avg_impr:+.2f}%  |  "
              f"TSB vs orig: {tsb_wins_orig}/{n}  |  TSB vs tuned: {tsb_wins_tuned}/{n}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["search", "eval", "analyze"], default="search")
    args = parser.parse_args()

    if args.step == "analyze":
        run_analyze()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.step == "search":
        run_search(device)
    elif args.step == "eval":
        run_eval(device)


if __name__ == "__main__":
    main()
