"""
Ablation: AdaptiveScheduler vs Fixed Loss Weights.

Compares:
  1. Adaptive (default) — dynamic aux/base weights via AdaptiveWeightScheduler
  2. Fixed — constant aux_weight=0.3, base_weight=0.1 throughout training

Runs on all 16 datasets with 3 seeds for speed.

Usage:
  uv run python experiments/ablation_adaptive_loss.py
"""
import sys, random, json
import numpy as np
import torch
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.layers import LinearBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.engine.trainer import train_v11
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data
from sklearn.metrics import mean_squared_error

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

SEEDS = [42, 123, 456]
RESULTS_PATH = Path("experiments/comprehensive_results/ablation_adaptive_loss.json")


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
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["raw_input"].to(device)
            y = batch["target"].to(device)
            output = model(x)
            pred = output[0] if isinstance(output, tuple) else output
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    if scaler:
        preds = preds * scaler.scale_[0] + scaler.mean_[0]
        targets = targets * scaler.scale_[0] + scaler.mean_[0]
    return np.sqrt(mean_squared_error(targets.flatten(), preds.flatten()))


def run_one(ds, seed, use_adaptive=True, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BEST_STAGE3_CONFIG.copy()
    config['epochs'] = epochs
    config['input_channels'] = 2

    if not use_adaptive:
        config['fixed_loss_weights'] = True
        config['fixed_aux_weight'] = 0.3
        config['fixed_base_weight'] = 0.1

    seq_len, pred_len = 30, 3
    path = f"dataset/{ds}.csv"
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=seq_len, horizon=pred_len, batch_size=32
    )

    base_kwargs = dict(
        seq_len=seq_len, pred_len=pred_len, input_channels=2,
        decomp_kernels=[7, 13],
        expert_kernels=config.get('expert_kernels', [3, 5, 7]),
        expert_hidden_dim=config.get('expert_hidden_dim', 32),
        dropout_p=config.get('dropout', 0.3),
        n_boost_rounds=config.get('n_boost_rounds', 2),
        expert_type=config.get('expert_type', 'cnn'),
    )

    set_seed(seed)
    model = FlexibleTSBooster(
        base_model=LinearBackbone(seq_len, 2, pred_len),
        use_decomp=True, use_experts=True,
        expert_gate='snr', enrich_mode='trend',
        **base_kwargs
    )
    model = train_v11(
        train_loader, val_loader, test_loader, device=device,
        horizon=pred_len, config=config, seq_len=seq_len, verbose=False, model=model
    )
    return evaluate(model, test_loader, scaler, device)


def main():
    results = {}

    for ds in ALL_DATASETS:
        results[ds] = {"adaptive": [], "fixed": []}
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")

        for seed in SEEDS:
            # Adaptive
            print(f"  Seed {seed} | Adaptive ... ", end="", flush=True)
            r_adapt = run_one(ds, seed, use_adaptive=True)
            print(f"RMSE={r_adapt:.6f}")

            # Fixed
            print(f"  Seed {seed} | Fixed    ... ", end="", flush=True)
            r_fixed = run_one(ds, seed, use_adaptive=False)
            print(f"RMSE={r_fixed:.6f}")

            if r_adapt is not None:
                results[ds]["adaptive"].append(r_adapt)
            if r_fixed is not None:
                results[ds]["fixed"].append(r_fixed)

        # Save incrementally
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    # === Summary ===
    print(f"\n\n{'='*80}")
    print("ABLATION: Adaptive Scheduler vs Fixed Weights")
    print(f"{'='*80}")
    print(f"{'Dataset':<12} | {'Adaptive RMSE':>14} | {'Fixed RMSE':>14} | {'Diff':>8} | {'Better':>8}")
    print("-" * 65)

    adapt_wins = 0
    for ds in ALL_DATASETS:
        a = np.mean(results[ds]["adaptive"])
        f = np.mean(results[ds]["fixed"])
        diff = (f - a) / f * 100  # positive = adaptive is better
        better = "Adaptive" if a < f else "Fixed"
        if a < f:
            adapt_wins += 1
        print(f"  {ds:<12} | {a:>14.6f} | {f:>14.6f} | {diff:>+7.2f}% | {better:>8}")

    print("-" * 65)
    print(f"  Adaptive wins: {adapt_wins}/{len(ALL_DATASETS)}")
    avg_a = np.mean([np.mean(results[ds]["adaptive"]) for ds in ALL_DATASETS])
    avg_f = np.mean([np.mean(results[ds]["fixed"]) for ds in ALL_DATASETS])
    print(f"  Overall avg — Adaptive: {avg_a:.6f}, Fixed: {avg_f:.6f}")
    diff_overall = (avg_f - avg_a) / avg_f * 100
    print(f"  Overall diff: {diff_overall:+.2f}% ({'Adaptive' if diff_overall > 0 else 'Fixed'} is better)")


if __name__ == "__main__":
    main()
