"""
Phase 6 (cont.) — Loss Component Ablation

Point 7 of the revision plan flagged this as missing: the paper's loss is

    Loss = (0.5 x Huber + 0.5 x Directional)        <- main loss on y_final
         + 0.3 x MSE(y_expert, y_true - y_base.detach())  <- aux (expert residual)
         + 0.1 x MSE(y_base, y_true)                       <- base (backbone accuracy)

This ablates each component individually (one-at-a-time removal) to show what
each contributes:

  full          : dir_weight=0.5, aux_weight=0.3, base_weight=0.1  (paper default)
  no_directional: dir_weight=0.0  (pure Huber main loss; tests DirectionalLoss's
                  contribution — expect MDA to drop most)
  no_aux        : fixed_aux_weight=0.0  (expert no longer explicitly trained on
                  the residual; tests whether the aux residual loss matters)
  no_base       : fixed_base_weight=0.0  (backbone not explicitly anchored to
                  accuracy; tests whether the base auxiliary loss matters)

Config: L=30, H=3 (paper's main setting), 3 seeds, full TS-Booster architecture
(only the loss config changes between variants).

This is a TSB-internal loss ablation (no Vanilla baseline needed — the question
is "what does each loss term contribute to TSB", not "does TSB beat Vanilla").

Usage:
  uv run python revision/phase6_ablation_ext/loss_component_ablation.py
  uv run python revision/phase6_ablation_ext/loss_component_ablation.py --analyze
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

# variant name -> config overrides (relative to BEST_STAGE3_CONFIG)
VARIANTS = {
    "full":           {},
    "no_directional": {"dir_weight": 0.0},
    "no_aux":         {"fixed_aux_weight": 0.0},
    "no_base":        {"fixed_base_weight": 0.0},
}
SEQ_LEN  = 30
HORIZON  = 3
SEEDS    = [42, 123, 456]
OUT_DIR  = Path("revision/phase6_ablation_ext")
OUT_PATH = OUT_DIR / "loss_component_results.json"


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


def build_tsb(config, device):
    return FlexibleTSBooster(
        base_model=LinearBackbone(SEQ_LEN, 2, HORIZON),
        seq_len=SEQ_LEN, pred_len=HORIZON, input_channels=2,
        use_decomp=True, use_experts=True,
        expert_gate="snr", enrich_mode="trend",
        decomp_kernels=config["decomp_kernels"],
        expert_kernels=config["expert_kernels"],
        expert_hidden_dim=config["expert_hidden_dim"],
        dropout_p=config["dropout"],
        n_boost_rounds=config["n_boost_rounds"],
        expert_type=config["expert_type"],
    ).to(device)


def run_all(device):
    base_config = BEST_STAGE3_CONFIG.copy()
    results = {}

    for name, overrides in VARIANTS.items():
        config = base_config.copy()
        config.update(overrides)
        print(f"\n{'='*70}")
        print(f"{name}  overrides={overrides}")
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
                tsb = build_tsb(config, device)
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

    for metric, label, better in [("rmse", "RMSE (lower better)", "min"),
                                   ("mda", "MDA (higher better)", "max")]:
        print("\n" + "=" * 90)
        print(f"Loss Component Ablation: avg {label} across 16 datasets")
        print("=" * 90)
        print(f"{'Dataset':<14}", end="")
        for name in names:
            print(f"  {name:>14}", end="")
        print()
        print("-" * (14 + 16 * len(names)))

        full_wins = 0
        for ds in ds_list:
            row = f"{ds:<14}"
            vals = {}
            for name in names:
                v = np.mean(results[name][ds][metric])
                vals[name] = v
                row += f"  {v:>14.4f}"
            print(row)
            best = min(vals, key=vals.get) if better == "min" else max(vals, key=vals.get)
            if best == "full":
                full_wins += 1

        print("-" * (14 + 16 * len(names)))
        avg_row = f"{'Average':<14}"
        for name in names:
            avg = np.mean([np.mean(results[name][ds][metric]) for ds in ds_list])
            avg_row += f"  {avg:>14.4f}"
        print(avg_row)
        print(f"\n'full' is best on: {full_wins}/{len(ds_list)} datasets ({metric.upper()})")

    print("\n" + "=" * 90)
    print("Component contribution (Δ from removing each component, avg over 16 datasets)")
    print("=" * 90)
    full_rmse = np.mean([np.mean(results["full"][ds]["rmse"]) for ds in ds_list])
    full_mda  = np.mean([np.mean(results["full"][ds]["mda"]) for ds in ds_list])
    for name in names:
        if name == "full":
            continue
        r = np.mean([np.mean(results[name][ds]["rmse"]) for ds in ds_list])
        m = np.mean([np.mean(results[name][ds]["mda"]) for ds in ds_list])
        print(f"  Removing → {name:<16}  ΔRMSE = {r - full_rmse:+.4f}   ΔMDA = {m - full_mda:+.4f}")


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
