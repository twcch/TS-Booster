"""
Phase 9b: Lookback Length Robustness

Tests whether TSB's gains hold across different lookback (input window) lengths.
Companion to phase9_temporal (which fixed L=30 and varied H); this fixes H and varies L.

Config:
  - Lookback L ∈ {30, 60, 90}  (L=30 is the paper's main setting)
  - Horizon H = 3 (fixed — matches the paper's main setting and phase9's anchor)
  - 3 seeds (robustness check, not full 5-seed; matches phase9 for comparability)
  - Models: Vanilla Linear vs TSB

Note: LinearBackbone's input dim depends on seq_len, so it must be reconstructed per L.

Usage:
  uv run python revision/phase9b_lookback/lookback_robustness.py
  uv run python revision/phase9b_lookback/lookback_robustness.py --analyze
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

from src.models.layers import LinearBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.engine.trainer import train_v11
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

LOOKBACKS = [30, 60, 90]
HORIZON   = 3
SEEDS     = [42, 123, 456]
OUT_DIR   = Path("revision/phase9b_lookback")
OUT_PATH  = OUT_DIR / "lookback_results.json"


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
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            last_obs_list.append(x[:, -1, 0].cpu().numpy())

    preds    = np.concatenate(preds) * scaler.scale_[0] + scaler.mean_[0]
    targets  = np.concatenate(targets) * scaler.scale_[0] + scaler.mean_[0]
    last_obs = np.concatenate(last_obs_list) * scaler.scale_[0] + scaler.mean_[0]

    rmse = float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))
    mae  = float(np.mean(np.abs(targets.flatten() - preds.flatten())))
    mda  = float(np.mean(np.sign(preds - last_obs[:, None]) == np.sign(targets - last_obs[:, None])))
    return {"rmse": rmse, "mae": mae, "mda": mda}


def run_all(device):
    config = BEST_STAGE3_CONFIG.copy()
    results = {}

    for L in LOOKBACKS:
        print(f"\n{'='*70}")
        print(f"Lookback L={L}")
        print(f"{'='*70}")
        results[L] = {}

        base_kwargs = dict(
            seq_len=L, pred_len=HORIZON, input_channels=2,
            decomp_kernels=config["decomp_kernels"],
            expert_kernels=config["expert_kernels"],
            expert_hidden_dim=config["expert_hidden_dim"],
            dropout_p=config["dropout"],
            n_boost_rounds=config["n_boost_rounds"],
            expert_type=config["expert_type"],
        )

        for ds in ALL_DATASETS:
            df = pd.read_csv(f"dataset/{ds}.csv")
            van_rmse, tsb_rmse = [], []

            for seed in SEEDS:
                set_seed(seed)
                train_loader, val_loader, test_loader, scaler = prepare_data(
                    df, lookback=L, horizon=HORIZON, batch_size=32
                )

                # Vanilla
                van = FlexibleTSBooster(
                    base_model=LinearBackbone(L, 2, HORIZON),
                    use_decomp=False, use_experts=False, **base_kwargs
                ).to(device)
                van = train_v11(train_loader, val_loader, test_loader, device=device,
                                horizon=HORIZON, config=config, model=van, verbose=False)
                mv = evaluate(van, test_loader, scaler, device)
                van_rmse.append(mv["rmse"])
                del van; torch.cuda.empty_cache()

                # TSB
                set_seed(seed)
                tsb = FlexibleTSBooster(
                    base_model=LinearBackbone(L, 2, HORIZON),
                    use_decomp=True, use_experts=True,
                    expert_gate="snr", enrich_mode="trend", **base_kwargs
                ).to(device)
                tsb = train_v11(train_loader, val_loader, test_loader, device=device,
                                horizon=HORIZON, config=config, model=tsb, verbose=False)
                mt = evaluate(tsb, test_loader, scaler, device)
                tsb_rmse.append(mt["rmse"])
                del tsb; torch.cuda.empty_cache()

            results[L][ds] = {"vanilla": van_rmse, "tsb": tsb_rmse}
            v_m = np.mean(van_rmse)
            t_m = np.mean(tsb_rmse)
            impr = (v_m - t_m) / v_m * 100
            print(f"  {ds:<14}  Van={v_m:.4f}  TSB={t_m:.4f}  Δ={impr:+.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT_PATH}")
    return results


def analyze(results):
    print("\n" + "=" * 80)
    print("Lookback Robustness: TSB improvement % by lookback length (avg over 16 datasets)")
    print("=" * 80)
    print(f"{'Dataset':<14}", end="")
    for L in LOOKBACKS:
        print(f"  L={L:>3}", end="")
    print()
    print("-" * (14 + 8 * len(LOOKBACKS)))

    for ds in ALL_DATASETS:
        row = f"{ds:<14}"
        for L in LOOKBACKS:
            key = L if L in results else str(L)
            if key not in results or ds not in results[key]:
                row += "     —  "
                continue
            v = np.mean(results[key][ds]["vanilla"])
            t = np.mean(results[key][ds]["tsb"])
            impr = (v - t) / v * 100
            row += f"  {impr:>+5.1f}%"
        print(row)

    print("-" * (14 + 8 * len(LOOKBACKS)))
    avg_row = f"{'Average':<14}"
    for L in LOOKBACKS:
        key = L if L in results else str(L)
        impr_list = [
            (np.mean(results[key][ds]["vanilla"]) - np.mean(results[key][ds]["tsb"]))
            / np.mean(results[key][ds]["vanilla"]) * 100
            for ds in ALL_DATASETS if ds in results.get(key, {})
        ]
        avg_row += f"  {np.mean(impr_list):>+5.1f}%" if impr_list else "     —  "
    print(avg_row)

    print()
    print("Win counts:")
    for L in LOOKBACKS:
        key = L if L in results else str(L)
        wins = sum(
            1 for ds in ALL_DATASETS
            if ds in results.get(key, {}) and
            np.mean(results[key][ds]["tsb"]) < np.mean(results[key][ds]["vanilla"])
        )
        n = len([ds for ds in ALL_DATASETS if ds in results.get(key, {})])
        print(f"  L={L:>3}: {wins}/{n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.analyze:
        if not OUT_PATH.exists():
            print(f"No results at {OUT_PATH}. Run without --analyze first.")
            return
        results = json.loads(OUT_PATH.read_text())
        analyze(results)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    results = run_all(device)
    analyze(results)


if __name__ == "__main__":
    main()
