"""
Phase 6 (cont.) — Expert Branch Count K Scan

Point 7 of the revision plan flagged this as missing: how sensitive is TSB to
the number of parallel multi-scale CNN branches (K) inside MultiScaleExpert?

MultiScaleExpert splits expert_hidden_dim across K parallel Conv1d branches
(branch_dim = hidden_dim // K), each with a different kernel size, then
concatenates and projects. The paper's default is K=3 with kernels=[3, 5, 7].

Config:
  - K ∈ {1, 2, 3, 4} via kernel sets [5], [3, 7], [3, 5, 7] (default), [3, 5, 7, 9]
  - expert_hidden_dim fixed at 32 (so total expert capacity ≈ matched, just
    split across a different number of receptive-field scales)
  - L=30, H=3 (paper's main setting), 3 seeds
  - All other config = BEST_STAGE3_CONFIG (full TS-Booster, decomp + experts + snr gate)

This is purely a TSB-internal architecture ablation (no Vanilla baseline needed —
the question is "which K is best for TSB", not "does TSB beat Vanilla").

Usage:
  uv run python revision/phase6_ablation_ext/expert_k_scan.py
  uv run python revision/phase6_ablation_ext/expert_k_scan.py --analyze
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

# K -> kernel set. K=3 ([3,5,7]) is the paper's default (Inception-style).
K_KERNELS = {
    1: [5],
    2: [3, 7],
    3: [3, 5, 7],
    4: [3, 5, 7, 9],
}
SEQ_LEN  = 30
HORIZON  = 3
SEEDS    = [42, 123, 456]
OUT_DIR  = Path("revision/phase6_ablation_ext")
OUT_PATH = OUT_DIR / "expert_k_scan_results.json"


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

    for K, kernels in K_KERNELS.items():
        print(f"\n{'='*70}")
        print(f"K={K}  kernels={kernels}")
        print(f"{'='*70}")
        results[K] = {}

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

            results[K][ds] = {"rmse": rmse_list, "mae": mae_list, "mda": mda_list}
            print(f"  {ds:<14}  RMSE={np.mean(rmse_list):.4f}  "
                  f"MAE={np.mean(mae_list):.4f}  MDA={np.mean(mda_list):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT_PATH}")
    return results


def analyze(results):
    Ks = sorted(int(k) for k in results.keys())
    print("\n" + "=" * 80)
    print("Expert K Scan: avg RMSE across 16 datasets, by K (kernel branch count)")
    print("=" * 80)
    print(f"{'Dataset':<14}", end="")
    for K in Ks:
        print(f"   K={K}", end="")
    print()
    print("-" * (14 + 7 * len(Ks)))

    ds_list = list(results[str(Ks[0])].keys()) if str(Ks[0]) in results else list(results[Ks[0]].keys())
    for ds in ds_list:
        row = f"{ds:<14}"
        for K in Ks:
            key = K if K in results else str(K)
            r = np.mean(results[key][ds]["rmse"])
            row += f"  {r:>5.3f}"
        print(row)

    print("-" * (14 + 7 * len(Ks)))
    avg_row = f"{'Avg RMSE':<14}"
    best_counts = {K: 0 for K in Ks}
    for ds in ds_list:
        rmses = {}
        for K in Ks:
            key = K if K in results else str(K)
            rmses[K] = np.mean(results[key][ds]["rmse"])
        best_K = min(rmses, key=rmses.get)
        best_counts[best_K] += 1
    for K in Ks:
        key = K if K in results else str(K)
        avg = np.mean([np.mean(results[key][ds]["rmse"]) for ds in ds_list])
        avg_row += f"  {avg:>5.3f}"
    print(avg_row)

    print()
    print("Best-K win counts (lowest avg RMSE per dataset):")
    for K in Ks:
        print(f"  K={K} (kernels={K_KERNELS.get(K, '?')}): {best_counts[K]}/{len(ds_list)}")


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
