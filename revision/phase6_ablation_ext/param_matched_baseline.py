"""
Phase 6a: Parameter-Matched Baseline

Motivation (from 學長 feedback):
  TSB-Linear adds ~6,790 parameters over vanilla Linear (366→7,156 params).
  To prove the gain comes from ARCHITECTURE (decomp + expert + gate),
  not merely from parameter count, we compare against a vanilla MLP with ~7K params.

Design:
  - MLP-7K: 2-hidden-layer MLP on flattened input (30×2=60 features)
    - Layer sizes: 60 → 64 → 48 → 3
    - Params: (60×64+64) + (64×48+48) + (48×3+3) = 3904 + 3120 + 147 = 7,171
  - Same training setup as vanilla Linear (MSE loss, Adam, 100 epochs, lr=1e-3)
  - Compare: Vanilla Linear (366p) vs MLP-7K (7,171p) vs TSB-Linear (7,156p)

Key hypothesis: If TSB wins over MLP-7K, the gains are architectural, not parametric.

Usage:
  uv run python revision/phase6_ablation_ext/param_matched_baseline.py
"""
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
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

SEEDS    = [42, 123, 456, 789, 2024]
SEQ_LEN  = 30
PRED_LEN = 3
N_FEAT   = 2
OUT_DIR  = Path("revision/phase6_ablation_ext")


# ── MLP-7K Model ──────────────────────────────────────────────────────────────

class MLP7K(nn.Module):
    """2-hidden-layer MLP with ~7K parameters. Flattens input and predicts directly."""
    def __init__(self, seq_len: int = 30, n_feat: int = 2, pred_len: int = 3,
                 h1: int = 64, h2: int = 48):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        in_dim = seq_len * n_feat
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, pred_len),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, S, C] → flatten → [B, S*C]
        B = x.shape[0]
        out = self.net(x.reshape(B, -1))  # [B, pred_len]
        # Return tuple compatible with FlexibleTSBooster format so train_v11 works:
        # (y_final, y_base, gate_weights, trend, seasonal, y_expert_out, expert_rounds)
        dummy = torch.zeros(3, device=x.device)
        return out, out, dummy, x, x, None, None


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ── Seed / Evaluate ───────────────────────────────────────────────────────────

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
            x  = batch["raw_input"].to(device)
            y  = batch["target"].to(device)
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            last_obs_list.append(x[:, -1, 0].cpu().numpy())

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    last_obs = np.concatenate(last_obs_list)

    preds    = preds    * scaler.scale_[0] + scaler.mean_[0]
    targets  = targets  * scaler.scale_[0] + scaler.mean_[0]
    last_obs = last_obs * scaler.scale_[0] + scaler.mean_[0]

    rmse = float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))
    mae  = float(np.mean(np.abs(targets.flatten() - preds.flatten())))
    mda  = float(np.mean(np.sign(preds - last_obs[:, None]) == np.sign(targets - last_obs[:, None])))
    return {"rmse": rmse, "mae": mae, "mda": mda}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BEST_STAGE3_CONFIG.copy()
    base_kwargs = dict(
        seq_len=SEQ_LEN, pred_len=PRED_LEN, input_channels=N_FEAT,
        decomp_kernels=config["decomp_kernels"],
        expert_kernels=config["expert_kernels"],
        expert_hidden_dim=config["expert_hidden_dim"],
        dropout_p=config["dropout"],
        n_boost_rounds=config["n_boost_rounds"],
        expert_type=config["expert_type"],
    )

    # Verify param counts
    mlp7k_sample = MLP7K(SEQ_LEN, N_FEAT, PRED_LEN)
    vanilla_sample = FlexibleTSBooster(
        base_model=LinearBackbone(SEQ_LEN, N_FEAT, PRED_LEN),
        use_decomp=False, use_experts=False, **base_kwargs
    )
    tsb_sample = FlexibleTSBooster(
        base_model=LinearBackbone(SEQ_LEN, N_FEAT, PRED_LEN),
        use_decomp=True, use_experts=True,
        expert_gate="snr", enrich_mode="trend", **base_kwargs
    )
    print(f"Model parameter counts:")
    print(f"  Vanilla Linear : {count_params(vanilla_sample):,}")
    print(f"  MLP-7K         : {count_params(mlp7k_sample):,}")
    print(f"  TSB-Linear     : {count_params(tsb_sample):,}")
    print()

    results = {}

    for ds in ALL_DATASETS:
        print(f"\n── {ds} ──────────────────────────────")
        df = pd.read_csv(f"dataset/{ds}.csv")

        van_rmse, mlp_rmse, tsb_rmse = [], [], []
        van_mda,  mlp_mda,  tsb_mda  = [], [], []

        for seed in SEEDS:
            set_seed(seed)
            train_loader, val_loader, test_loader, scaler = prepare_data(
                df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32
            )

            # 1. Vanilla Linear
            van = FlexibleTSBooster(
                base_model=LinearBackbone(SEQ_LEN, N_FEAT, PRED_LEN),
                use_decomp=False, use_experts=False, **base_kwargs
            ).to(device)
            van = train_v11(train_loader, val_loader, test_loader, device=device,
                            horizon=PRED_LEN, config=config, model=van, verbose=False)
            mv = evaluate(van, test_loader, scaler, device)
            van_rmse.append(mv["rmse"]); van_mda.append(mv["mda"])
            del van; torch.cuda.empty_cache()

            # 2. MLP-7K
            set_seed(seed)
            mlp = MLP7K(SEQ_LEN, N_FEAT, PRED_LEN).to(device)
            mlp = train_v11(train_loader, val_loader, test_loader, device=device,
                            horizon=PRED_LEN, config=config, model=mlp, verbose=False)
            mm = evaluate(mlp, test_loader, scaler, device)
            mlp_rmse.append(mm["rmse"]); mlp_mda.append(mm["mda"])
            del mlp; torch.cuda.empty_cache()

            # 3. TSB
            set_seed(seed)
            tsb = FlexibleTSBooster(
                base_model=LinearBackbone(SEQ_LEN, N_FEAT, PRED_LEN),
                use_decomp=True, use_experts=True,
                expert_gate="snr", enrich_mode="trend", **base_kwargs
            ).to(device)
            tsb = train_v11(train_loader, val_loader, test_loader, device=device,
                            horizon=PRED_LEN, config=config, model=tsb, verbose=False)
            mt = evaluate(tsb, test_loader, scaler, device)
            tsb_rmse.append(mt["rmse"]); tsb_mda.append(mt["mda"])
            del tsb; torch.cuda.empty_cache()

            print(f"  seed {seed}: Vanilla={mv['rmse']:.4f}  MLP7K={mm['rmse']:.4f}  TSB={mt['rmse']:.4f}")

        results[ds] = {
            "vanilla": {"rmse": van_rmse, "mda": van_mda},
            "mlp7k":   {"rmse": mlp_rmse, "mda": mlp_mda},
            "tsb":     {"rmse": tsb_rmse, "mda": tsb_mda},
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("Parameter-Matched Baseline: Vanilla (366p) vs MLP-7K (7K) vs TSB (7K)")
    print("=" * 90)
    print(f"{'Dataset':<14} {'Vanilla':>10} {'MLP-7K':>10} {'TSB':>10} {'TSB>MLP':>10} {'TSB>Van':>10}")
    print("-" * 70)

    tsb_beats_mlp = 0
    tsb_beats_van = 0

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        r = results[ds]
        van_r = np.mean(r["vanilla"]["rmse"])
        mlp_r = np.mean(r["mlp7k"]["rmse"])
        tsb_r = np.mean(r["tsb"]["rmse"])

        beat_mlp = tsb_r < mlp_r
        beat_van = tsb_r < van_r
        if beat_mlp: tsb_beats_mlp += 1
        if beat_van: tsb_beats_van += 1

        print(f"{ds:<14} {van_r:>10.4f} {mlp_r:>10.4f} {tsb_r:>10.4f} "
              f"{'✓' if beat_mlp else '✗':>10} {'✓' if beat_van else '✗':>10}")

    n = len(results)
    print("-" * 70)
    print(f"\nTSB beats Vanilla: {tsb_beats_van}/{n}")
    print(f"TSB beats MLP-7K:  {tsb_beats_mlp}/{n}  ← KEY RESULT")
    print(f"\nInterpretation: If TSB > MLP-7K on most datasets, gains are architectural,")
    print(f"not from parameter count.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "param_matched_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
