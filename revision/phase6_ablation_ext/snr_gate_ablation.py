"""
Phase 6b: SNR Gate Ablation

Isolates the contribution of the SNR gate by comparing 4 gate variants,
all with the same decomp-enriched expert and same backbone:

  1. No gate    (gate=1.0 always) — expert_gate=None in FlexibleTSBooster
  2. Const-0.5  (gate=0.5 always) — monkey-patched after init
  3. Learned    (single learnable scalar) — expert_gate='learned'
  4. SNR gate   (per-sample SNR-based)   — expert_gate='snr'  [current TSB]

All configs: use_decomp=True, use_experts=True, enrich_mode='trend'
Result helps isolate gate contribution from decomp enrichment.

Usage:
  uv run python revision/phase6_ablation_ext/snr_gate_ablation.py
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


# ── Const-0.5 Gate Subclass ───────────────────────────────────────────────────

class ConstGateBooster(FlexibleTSBooster):
    """FlexibleTSBooster subclass with a fixed constant gate value."""
    def __init__(self, *args, gate_value: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._const_gate_value = gate_value

    def _compute_expert_gate(self, x, seasonal_part):
        return torch.tensor(self._const_gate_value, device=x.device)


# ── Helpers ───────────────────────────────────────────────────────────────────

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

    preds    = np.concatenate(preds) * scaler.scale_[0] + scaler.mean_[0]
    targets  = np.concatenate(targets) * scaler.scale_[0] + scaler.mean_[0]
    last_obs = np.concatenate(last_obs_list) * scaler.scale_[0] + scaler.mean_[0]

    rmse = float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))
    mae  = float(np.mean(np.abs(targets.flatten() - preds.flatten())))
    mda  = float(np.mean(np.sign(preds - last_obs[:, None]) == np.sign(targets - last_obs[:, None])))
    return {"rmse": rmse, "mae": mae, "mda": mda}


def make_config(gate_type):
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
    if gate_type == "no_gate":
        return FlexibleTSBooster(
            base_model=LinearBackbone(SEQ_LEN, N_FEAT, PRED_LEN),
            use_decomp=True, use_experts=True,
            expert_gate=None, enrich_mode="trend", **base_kwargs
        )
    elif gate_type == "const_0.5":
        return ConstGateBooster(
            base_model=LinearBackbone(SEQ_LEN, N_FEAT, PRED_LEN),
            use_decomp=True, use_experts=True,
            expert_gate=None, enrich_mode="trend",
            gate_value=0.5, **base_kwargs
        )
    elif gate_type == "learned":
        return FlexibleTSBooster(
            base_model=LinearBackbone(SEQ_LEN, N_FEAT, PRED_LEN),
            use_decomp=True, use_experts=True,
            expert_gate="learned", enrich_mode="trend", **base_kwargs
        )
    elif gate_type == "snr":
        return FlexibleTSBooster(
            base_model=LinearBackbone(SEQ_LEN, N_FEAT, PRED_LEN),
            use_decomp=True, use_experts=True,
            expert_gate="snr", enrich_mode="trend", **base_kwargs
        )
    raise ValueError(gate_type)


GATE_CONFIGS = ["no_gate", "const_0.5", "learned", "snr"]
GATE_LABELS  = ["No gate (=1)", "Const (=0.5)", "Learned scalar", "SNR gate (TSB)"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BEST_STAGE3_CONFIG.copy()
    results = {}

    for ds in ALL_DATASETS:
        print(f"\n── {ds} ──────────────────────────────")
        df = pd.read_csv(f"dataset/{ds}.csv")
        ds_r = {g: {"rmse": [], "mae": [], "mda": []} for g in GATE_CONFIGS}

        for seed in SEEDS:
            set_seed(seed)
            train_loader, val_loader, test_loader, scaler = prepare_data(
                df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32
            )
            row = []
            for gt in GATE_CONFIGS:
                set_seed(seed)
                m = make_config(gt).to(device)
                m = train_v11(train_loader, val_loader, test_loader, device=device,
                              horizon=PRED_LEN, config=config, model=m, verbose=False)
                r = evaluate(m, test_loader, scaler, device)
                for k in ["rmse", "mae", "mda"]:
                    ds_r[gt][k].append(r[k])
                row.append(f"{gt}={r['rmse']:.4f}")
                del m; torch.cuda.empty_cache()

            print(f"  seed {seed}: " + "  ".join(row))

        results[ds] = ds_r

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SNR Gate Ablation: all configs with decomp + expert + trend enrichment")
    print("=" * 90)
    print(f"{'Dataset':<14}" + "".join(f"{lb:>17}" for lb in GATE_LABELS))
    print("-" * 84)

    snr_wins = 0
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        means = [np.mean(results[ds][g]["rmse"]) for g in GATE_CONFIGS]
        best  = np.argmin(means)
        if best == GATE_CONFIGS.index("snr"):
            snr_wins += 1
        row = f"{ds:<14}"
        for i, m in enumerate(means):
            marker = "*" if i == best else " "
            row += f"{m:>16.4f}{marker}"
        print(row)

    n = len([ds for ds in ALL_DATASETS if ds in results])
    print("-" * 84)
    avg_means = [np.mean([np.mean(results[ds][g]["rmse"])
                          for ds in ALL_DATASETS if ds in results])
                 for g in GATE_CONFIGS]
    best_avg = np.argmin(avg_means)
    avg_row = f"{'Average':<14}"
    for i, m in enumerate(avg_means):
        marker = "*" if i == best_avg else " "
        avg_row += f"{m:>16.4f}{marker}"
    print(avg_row)

    print(f"\nSNR gate (TSB) is best on {snr_wins}/{n} datasets")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "snr_gate_ablation_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
