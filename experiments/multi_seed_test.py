"""
Multi-seed evaluation to assess robustness.
Run K_7_13 + SNR + trend config with 5 seeds on critical datasets.
Report mean ± std gap.
"""
import sys, random, numpy as np, torch, pandas as pd
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

SEEDS = [42, 123, 456, 789, 2024]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def evaluate(model, test_loader, scaler, device):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["raw_input"].to(device); y = batch["target"].to(device)
            output = model(x)
            pred = output[0] if isinstance(output, tuple) else output
            preds.append(pred.cpu().numpy()); targets.append(y.cpu().numpy())
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    if scaler: preds = preds * scaler.scale_[0] + scaler.mean_[0]; targets = targets * scaler.scale_[0] + scaler.mean_[0]
    return np.sqrt(mean_squared_error(targets.flatten(), preds.flatten()))

def run_pair(ds, seed, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BEST_STAGE3_CONFIG.copy()
    config['epochs'] = epochs; config['input_channels'] = 2
    seq_len, pred_len = 30, 3

    path = f"dataset/{ds}.csv"
    if not Path(path).exists(): return None
    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(df, lookback=seq_len, horizon=pred_len, batch_size=32)

    base_kwargs = dict(
        seq_len=seq_len, pred_len=pred_len, input_channels=2,
        decomp_kernels=[7, 13],
        expert_kernels=config.get('expert_kernels', [3, 5, 7]),
        expert_hidden_dim=config.get('expert_hidden_dim', 32),
        dropout_p=config.get('dropout', 0.3),
        n_boost_rounds=config.get('n_boost_rounds', 2),
        expert_type=config.get('expert_type', 'cnn'),
    )

    # +Expert
    set_seed(seed)
    em = FlexibleTSBooster(base_model=LinearBackbone(seq_len, 2, pred_len),
                           use_decomp=False, use_experts=True, **base_kwargs)
    em = train_v11(train_loader, val_loader, test_loader, device=device,
                   horizon=pred_len, config=config, seq_len=seq_len, verbose=False, model=em)
    er = evaluate(em, test_loader, scaler, device)

    # TS-Booster
    set_seed(seed)
    bm = FlexibleTSBooster(base_model=LinearBackbone(seq_len, 2, pred_len),
                           use_decomp=True, use_experts=True,
                           expert_gate='snr', enrich_mode='trend',
                           **base_kwargs)
    bm = train_v11(train_loader, val_loader, test_loader, device=device,
                   horizon=pred_len, config=config, seq_len=seq_len, verbose=False, model=bm)
    br = evaluate(bm, test_loader, scaler, device)

    gap = (er - br) / er * 100
    return gap

if __name__ == "__main__":
    # {dataset: [gaps per seed]}
    results = {ds: [] for ds in ALL_DATASETS}

    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"Seed: {seed}")
        print(f"{'='*70}")
        for ds in ALL_DATASETS:
            gap = run_pair(ds, seed)
            if gap is not None:
                results[ds].append(gap)
                win = "WIN" if gap > 0 else "LOSS"
                print(f"  {ds:12s} | Seed {seed}: {gap:+6.2f}% {win}")

    # Summary
    print(f"\n\n{'='*80}")
    print(f"MULTI-SEED SUMMARY (K_7_13 + SNR + trend, {len(SEEDS)} seeds)")
    print(f"{'='*80}")
    print(f"{'Dataset':<12} | {'Mean':>7} | {'Std':>6} | {'Min':>7} | {'Max':>7} | {'Wins':>5} | {'Result':>6}")
    print("-" * 70)

    total_wins = 0
    for ds in ALL_DATASETS:
        gaps = results[ds]
        if gaps:
            mean_g = np.mean(gaps)
            std_g = np.std(gaps)
            min_g = np.min(gaps)
            max_g = np.max(gaps)
            wins = sum(1 for g in gaps if g > 0)
            # "WIN" if mean > 0 OR wins >= 3/5
            result = "WIN" if mean_g > 0 else "LOSS"
            if mean_g > 0: total_wins += 1
            print(f"  {ds:<12} | {mean_g:>+6.2f}% | {std_g:>5.2f}% | {min_g:>+6.2f}% | {max_g:>+6.2f}% | {wins}/{len(gaps)} | {result}")

    print(f"\nTotal: {total_wins}/{len(ALL_DATASETS)} datasets with positive mean gap")
    avg_mean = np.mean([np.mean(results[ds]) for ds in ALL_DATASETS if results[ds]])
    print(f"Overall avg gap: {avg_mean:+.2f}%")
