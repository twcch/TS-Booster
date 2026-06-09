"""
Phase 4: PatchTST without RevIN — Fair Comparison Re-run

Problem:
  Original sota_baselines.py ran PatchTST WITH RevIN (instance-level normalization),
  while DLinear, iTransformer, and TSB all use global Z-score normalization.
  This confounds architecture comparison with normalization method.

  Note: iTransformer in this codebase already has NO RevIN (verified).
  Only PatchTST needs to be re-run.

Fix:
  Re-run PatchTST with use_revin=False (global Z-score only, same as all other models).
  Compare new results against original (with RevIN) to quantify the RevIN effect.

Usage:
  uv run python revision/phase4_revin_fix/rerun_patchtst_no_revin.py
"""
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import mean_squared_error

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.sota_baselines import PatchTST
from src.engine.trainer import train_baseline
from src.utils.data_loader import prepare_data

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

SEEDS    = [42, 123, 456, 789, 2024]
SEQ_LEN  = 30
PRED_LEN = 3
LR       = 1e-4

ORIG_SOTA_PATH = Path("experiments/comprehensive_results/sota_baselines/sota_baselines_results.json")
DL_PATH        = Path("experiments/comprehensive_results/main_experiment/statistical_test_results.json")
OUT_PATH       = Path("revision/phase4_revin_fix/patchtst_no_revin_results.json")


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
            pred = model(x)[0]
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            last_obs_list.append(x[:, -1, 0].cpu().numpy())

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    last_obs = np.concatenate(last_obs_list)

    # Inverse-transform
    preds    = preds    * scaler.scale_[0] + scaler.mean_[0]
    targets  = targets  * scaler.scale_[0] + scaler.mean_[0]
    last_obs = last_obs * scaler.scale_[0] + scaler.mean_[0]

    rmse = float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))
    mae  = float(np.mean(np.abs(targets.flatten() - preds.flatten())))
    mda  = float(np.mean(np.sign(preds - last_obs[:, None]) == np.sign(targets - last_obs[:, None])))
    return {"rmse": rmse, "mae": mae, "mda": mda}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    orig = json.loads(ORIG_SOTA_PATH.read_text())
    dl   = json.loads(DL_PATH.read_text())

    results = {}

    for ds in ALL_DATASETS:
        print(f"\n── {ds} ──────────────────────────────")
        df = pd.read_csv(f"dataset/{ds}.csv")

        seed_metrics = []
        for seed in SEEDS:
            set_seed(seed)
            train_loader, val_loader, test_loader, scaler = prepare_data(
                df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32
            )
            model = PatchTST(
                seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=2, use_revin=False
            )
            model = train_baseline(
                train_loader, val_loader, test_loader,
                device=device, model=model, num_epochs=100, lr=LR, verbose=False,
            )
            m = evaluate(model, test_loader, scaler, device)
            seed_metrics.append(m)
            del model
            torch.cuda.empty_cache()
            print(f"  seed {seed}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  MDA={m['mda']:.4f}")

        new_rmse = float(np.mean([m["rmse"] for m in seed_metrics]))
        new_mae  = float(np.mean([m["mae"]  for m in seed_metrics]))
        new_mda  = float(np.mean([m["mda"]  for m in seed_metrics]))

        # Original PatchTST (with RevIN)
        orig_rmse = float(np.mean(orig[ds]["PatchTST"]["rmse"])) if ds in orig else None
        tsb_rmse  = float(np.mean(dl[ds]["booster"]["rmse"]))    if ds in dl   else None

        results[ds] = {
            "patchtst_no_revin": {"rmse": [m["rmse"] for m in seed_metrics],
                                  "mae":  [m["mae"]  for m in seed_metrics],
                                  "mda":  [m["mda"]  for m in seed_metrics]},
            "patchtst_orig_rmse": orig_rmse,
            "tsb_rmse":           tsb_rmse,
        }

        delta = ((orig_rmse - new_rmse) / orig_rmse * 100) if orig_rmse else None
        tsb_vs_new = ((new_rmse - tsb_rmse) / new_rmse * 100) if tsb_rmse else None
        print(f"  PatchTST w/ RevIN : {orig_rmse:.4f}" if orig_rmse else "  PatchTST w/ RevIN : N/A")
        print(f"  PatchTST no RevIN : {new_rmse:.4f}  (Δ={delta:+.1f}% vs orig)" if delta else f"  PatchTST no RevIN : {new_rmse:.4f}")
        print(f"  TSB               : {tsb_rmse:.4f}  TSB vs new PatchTST: {tsb_vs_new:+.1f}%" if tsb_vs_new else "")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: PatchTST w/ RevIN  vs  PatchTST no RevIN  vs  TSB")
    print("=" * 80)
    print(f"{'Dataset':<14} {'w/RevIN':>10} {'no RevIN':>10} {'RevIN effect':>13} {'TSB wins?':>10}")
    print("-" * 80)

    tsb_wins_orig = 0
    tsb_wins_new  = 0
    revin_helps   = 0

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        r = results[ds]
        new_r  = float(np.mean(r["patchtst_no_revin"]["rmse"]))
        orig_r = r["patchtst_orig_rmse"]
        tsb_r  = r["tsb_rmse"]

        effect = f"{(orig_r-new_r)/orig_r*100:+.1f}%" if orig_r else "N/A"
        if orig_r and new_r < orig_r:
            revin_helps += 1  # RevIN was actually hurting (removing it helped)

        tsb_o = "✓" if tsb_r and tsb_r < orig_r else "✗"
        tsb_n = "✓" if tsb_r and tsb_r < new_r  else "✗"
        if tsb_r and tsb_r < orig_r: tsb_wins_orig += 1
        if tsb_r and tsb_r < new_r:  tsb_wins_new  += 1

        print(f"{ds:<14} {orig_r:>10.4f} {new_r:>10.4f} {effect:>13} {tsb_n:>5}(new) {tsb_o:>5}(orig)")

    n = len(results)
    print("-" * 80)
    print(f"\nTSB beats PatchTST (original, w/ RevIN) : {tsb_wins_orig}/{n}")
    print(f"TSB beats PatchTST (no RevIN, fair)     : {tsb_wins_new}/{n}")
    print(f"Removing RevIN improved PatchTST        : {revin_helps}/{n} datasets")
    print(f"  (positive = RevIN was hurting, confirming 學長's hypothesis)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
