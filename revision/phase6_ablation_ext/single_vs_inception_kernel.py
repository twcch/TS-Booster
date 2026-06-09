"""
Phase 6 (cont.) — Single Conv Kernel vs Inception Multi-Kernel Expert

Point 7 of the revision plan flagged this as missing: does the Inception-style
multi-scale design (parallel branches with kernels [3, 5, 7], concatenated)
actually matter, or would a plain single-kernel CNN with the same total
capacity (expert_hidden_dim) do just as well?

Config:
  - Inception (paper default): kernels=[3, 5, 7], branch_dim = 32 // 3 ≈ 10
  - Single-kernel variants: kernels=[k] for k ∈ {3, 5, 7}, branch_dim = 32
    (one branch gets the FULL hidden_dim — capacity-matched, scale-unmatched)
  - L=30, H=3 (paper's main setting), 3 seeds
  - All other config = BEST_STAGE3_CONFIG (full TS-Booster)

Rationale for testing all three single-kernel sizes (not just one): if Inception
beats every individual kernel choice, that's strong evidence the *combination*
of receptive-field scales is what matters — not just lucky kernel-size selection.

This is a TSB-internal architecture ablation (no Vanilla baseline needed).

Usage:
  uv run python revision/phase6_ablation_ext/single_vs_inception_kernel.py
  uv run python revision/phase6_ablation_ext/single_vs_inception_kernel.py --analyze
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

# variant name -> kernel set. "inception" is the paper's default.
VARIANTS = {
    "single_k3":  [3],
    "single_k5":  [5],
    "single_k7":  [7],
    "inception":  [3, 5, 7],
}
SEQ_LEN  = 30
HORIZON  = 3
SEEDS    = [42, 123, 456]
OUT_DIR  = Path("revision/phase6_ablation_ext")
OUT_PATH = OUT_DIR / "single_vs_inception_results.json"


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


def build_tsb(config, kernels, device):
    return FlexibleTSBooster(
        base_model=LinearBackbone(SEQ_LEN, 2, HORIZON),
        seq_len=SEQ_LEN, pred_len=HORIZON, input_channels=2,
        use_decomp=True, use_experts=True,
        expert_gate="snr", enrich_mode="trend",
        decomp_kernels=config["decomp_kernels"],
        expert_kernels=kernels,
        expert_hidden_dim=config["expert_hidden_dim"],
        dropout_p=config["dropout"],
        n_boost_rounds=config["n_boost_rounds"],
        expert_type=config["expert_type"],
    ).to(device)


def run_all(device):
    config = BEST_STAGE3_CONFIG.copy()
    results = {}

    for name, kernels in VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"{name}  kernels={kernels}")
        print(f"{'='*70}")
        results[name] = {}

        for ds in ALL_DATASETS:
            df = pd.read_csv(f"dataset/{ds}.csv")
            rmse_list, mae_list, mda_list = [], [], []

            for seed in SEEDS:
                set_seed(seed)
                train_loader, val_loader, test_loader, scaler = prepare_data(
                    df, lookback=SEQ_LEN, horizon=HORIZON, batch_size=32
                )
                tsb = build_tsb(config, kernels, device)
                tsb = train_v11(train_loader, val_loader, test_loader, device=device,
                                horizon=HORIZON, config=config, model=tsb, verbose=False)
                m = evaluate(tsb, test_loader, scaler, device)
                rmse_list.append(m["rmse"]); mae_list.append(m["mae"]); mda_list.append(m["mda"])
                del tsb; torch.cuda.empty_cache()

            results[name][ds] = {"rmse": rmse_list, "mae": mae_list, "mda": mda_list}
            print(f"  {ds:<14}  RMSE={np.mean(rmse_list):.4f}  "
                  f"MAE={np.mean(mae_list):.4f}  MDA={np.mean(mda_list):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT_PATH}")
    return results


def analyze(results):
    names = list(VARIANTS.keys())
    ds_list = list(results[names[0]].keys())

    print("\n" + "=" * 90)
    print("Single Kernel vs Inception: avg RMSE across 16 datasets")
    print("=" * 90)
    print(f"{'Dataset':<14}", end="")
    for name in names:
        print(f"  {name:>11}", end="")
    print()
    print("-" * (14 + 13 * len(names)))

    inception_wins = 0
    for ds in ds_list:
        row = f"{ds:<14}"
        rmses = {}
        for name in names:
            r = np.mean(results[name][ds]["rmse"])
            rmses[name] = r
            row += f"  {r:>11.4f}"
        print(row)
        if rmses["inception"] == min(rmses.values()):
            inception_wins += 1

    print("-" * (14 + 13 * len(names)))
    avg_row = f"{'Average':<14}"
    for name in names:
        avg = np.mean([np.mean(results[name][ds]["rmse"]) for ds in ds_list])
        avg_row += f"  {avg:>11.4f}"
    print(avg_row)

    print()
    print(f"Inception [3,5,7] beats ALL single-kernel variants on: {inception_wins}/{len(ds_list)} datasets")
    for name in names:
        if name == "inception":
            continue
        wins = sum(
            1 for ds in ds_list
            if np.mean(results["inception"][ds]["rmse"]) < np.mean(results[name][ds]["rmse"])
        )
        print(f"  Inception vs {name:<11}: {wins}/{len(ds_list)} wins")


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
