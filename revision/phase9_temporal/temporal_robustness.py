"""
Phase 9: Temporal Scale Robustness

Tests whether TSB's gains hold across different forecast horizons.

Config:
  - Horizon H ∈ {1, 3, 5, 10}  (H=3 is the paper's main setting)
  - Lookback L = 30 (fixed — horizon is the more critical dimension)
  - 3 seeds (robustness check, not full 5-seed)
  - Models: Vanilla Linear vs TSB

For H > 1: both models output H steps directly (same training objective).
For H = 1: single-step prediction.

Note: sensitivity analysis already covers n_boost_rounds and decomp kernels.
This experiment focuses specifically on H variation.

Usage:
  uv run python revision/phase9_temporal/temporal_robustness.py
  uv run python revision/phase9_temporal/temporal_robustness.py --analyze
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

HORIZONS  = [1, 3, 5, 10]
SEQ_LEN   = 30
SEEDS     = [42, 123, 456]
OUT_DIR   = Path("revision/phase9_temporal")
OUT_PATH  = OUT_DIR / "temporal_results.json"


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

    for H in HORIZONS:
        print(f"\n{'='*70}")
        print(f"Horizon H={H}")
        print(f"{'='*70}")
        results[H] = {}

        base_kwargs = dict(
            seq_len=SEQ_LEN, pred_len=H, input_channels=2,
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
                    df, lookback=SEQ_LEN, horizon=H, batch_size=32
                )

                # Vanilla
                van = FlexibleTSBooster(
                    base_model=LinearBackbone(SEQ_LEN, 2, H),
                    use_decomp=False, use_experts=False, **base_kwargs
                ).to(device)
                van = train_v11(train_loader, val_loader, test_loader, device=device,
                                horizon=H, config=config, model=van, verbose=False)
                mv = evaluate(van, test_loader, scaler, device)
                van_rmse.append(mv["rmse"])
                del van; torch.cuda.empty_cache()

                # TSB
                set_seed(seed)
                tsb = FlexibleTSBooster(
                    base_model=LinearBackbone(SEQ_LEN, 2, H),
                    use_decomp=True, use_experts=True,
                    expert_gate="snr", enrich_mode="trend", **base_kwargs
                ).to(device)
                tsb = train_v11(train_loader, val_loader, test_loader, device=device,
                                horizon=H, config=config, model=tsb, verbose=False)
                mt = evaluate(tsb, test_loader, scaler, device)
                tsb_rmse.append(mt["rmse"])
                del tsb; torch.cuda.empty_cache()

            results[H][ds] = {"vanilla": van_rmse, "tsb": tsb_rmse}
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
    print("Temporal Robustness: TSB improvement % by horizon (avg over 16 datasets)")
    print("=" * 80)
    print(f"{'Dataset':<14}", end="")
    for H in HORIZONS:
        print(f"  H={H:>2}", end="")
    print()
    print("-" * (14 + 7 * len(HORIZONS)))

    for ds in ALL_DATASETS:
        row = f"{ds:<14}"
        for H in HORIZONS:
            if str(H) not in results and H not in results:
                row += "    —  "
                continue
            key = H if H in results else str(H)
            if ds not in results[key]:
                row += "    —  "
                continue
            v = np.mean(results[key][ds]["vanilla"])
            t = np.mean(results[key][ds]["tsb"])
            impr = (v - t) / v * 100
            row += f"  {impr:>+4.1f}%"
        print(row)

    print("-" * (14 + 7 * len(HORIZONS)))
    avg_row = f"{'Average':<14}"
    for H in HORIZONS:
        key = H if H in results else str(H)
        impr_list = [
            (np.mean(results[key][ds]["vanilla"]) - np.mean(results[key][ds]["tsb"]))
            / np.mean(results[key][ds]["vanilla"]) * 100
            for ds in ALL_DATASETS if ds in results.get(key, {})
        ]
        avg_row += f"  {np.mean(impr_list):>+4.1f}%" if impr_list else "    —  "
    print(avg_row)

    print()
    print("Win counts:")
    for H in HORIZONS:
        key = H if H in results else str(H)
        wins = sum(
            1 for ds in ALL_DATASETS
            if ds in results.get(key, {}) and
            np.mean(results[key][ds]["tsb"]) < np.mean(results[key][ds]["vanilla"])
        )
        n = len([ds for ds in ALL_DATASETS if ds in results.get(key, {})])
        print(f"  H={H:>2}: {wins}/{n}")


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
