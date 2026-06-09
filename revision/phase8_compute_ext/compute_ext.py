"""
Phase 8: Compute Cost Extension

Extends existing compute_cost.py with:
  1. FLOPs (Multiply-Accumulate operations) via torchinfo
  2. Peak GPU memory during inference (torch.cuda.max_memory_allocated)
  3. Parameter count comparison (verify matches existing results)
  4. Compact table for paper

Usage:
  uv run python revision/phase8_compute_ext/compute_ext.py
"""
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.layers import LinearBackbone, LSTMBackbone, GRUBackbone, TransformerBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.models.sota_baselines import DLinear, PatchTST, iTransformer
from src.configs.model_config import BEST_STAGE3_CONFIG
from torchinfo import summary

SEQ_LEN  = 30
PRED_LEN = 3
N_FEAT   = 2
BATCH    = 32
N_WARMUP = 20
N_RUNS   = 200

OUT_DIR = Path("revision/phase8_compute_ext")


def make_vanilla_and_tsb(backbone_type: str):
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

    def make_backbone():
        if backbone_type == "linear":
            return LinearBackbone(SEQ_LEN, N_FEAT, PRED_LEN)
        elif backbone_type == "lstm":
            return LSTMBackbone(SEQ_LEN, N_FEAT, PRED_LEN, hidden_dim=64, num_layers=2)
        elif backbone_type == "gru":
            return GRUBackbone(SEQ_LEN, N_FEAT, PRED_LEN, hidden_dim=64, num_layers=2)
        elif backbone_type == "transformer":
            return TransformerBackbone(SEQ_LEN, N_FEAT, PRED_LEN, hidden_dim=64, num_layers=2)
        raise ValueError(backbone_type)

    vanilla = FlexibleTSBooster(
        base_model=make_backbone(), use_decomp=False, use_experts=False, **base_kwargs
    )
    tsb = FlexibleTSBooster(
        base_model=make_backbone(), use_decomp=True, use_experts=True,
        expert_gate="snr", enrich_mode="trend", **base_kwargs
    )
    return vanilla, tsb


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model: torch.nn.Module, device: torch.device) -> int:
    """Return total MACs (Multiply-Accumulate Ops) via torchinfo."""
    model = model.to(device).eval()
    # FlexibleTSBooster.forward(x) takes raw tensor [B, S, C]
    info = summary(model, input_size=(1, SEQ_LEN, N_FEAT), verbose=0, device=str(device))
    return info.total_mult_adds


def measure_inference(model: torch.nn.Module, device: torch.device) -> tuple[float, float]:
    """Return (mean_ms, peak_memory_MB) for batch inference."""
    model = model.to(device).eval()
    x = torch.randn(BATCH, SEQ_LEN, N_FEAT, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)  # ms

    mean_ms = float(np.mean(times))

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    return mean_ms, peak_mb


def measure_sota(model: torch.nn.Module, device: torch.device) -> dict:
    model = model.to(device).eval()
    x = torch.randn(1, SEQ_LEN, N_FEAT, device=device)
    info = summary(model, input_size=(1, SEQ_LEN, N_FEAT), verbose=0, device=str(device))
    flops = info.total_mult_adds

    xb = torch.randn(BATCH, SEQ_LEN, N_FEAT, device=device)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(xb)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(xb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0
    return {"params": count_params(model), "flops": flops,
            "latency_ms": float(np.mean(times)), "peak_mem_mb": peak_mb}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = {}
    BACKBONES = ["linear", "lstm", "gru", "transformer"]

    print(f"{'Model':<26} {'Params':>10} {'FLOPs(M)':>10} {'Latency(ms)':>13} {'PeakMem(MB)':>13}")
    print("-" * 80)

    for bb in BACKBONES:
        van, tsb = make_vanilla_and_tsb(bb)

        v_params = count_params(van)
        t_params = count_params(tsb)

        v_flops  = measure_flops(van, device)
        t_flops  = measure_flops(tsb, device)

        v_lat, v_mem = measure_inference(van, device)
        t_lat, t_mem = measure_inference(tsb, device)

        overhead_pct = (t_params - v_params) / v_params * 100

        label_v = f"Linear-{bb}" if bb == "linear" else bb.upper()
        print(f"Vanilla-{bb:<16} {v_params:>10,} {v_flops/1e6:>10.3f} {v_lat:>12.3f}ms {v_mem:>12.1f}MB")
        print(f"TSB-{bb:<20} {t_params:>10,} {t_flops/1e6:>10.3f} {t_lat:>12.3f}ms {t_mem:>12.1f}MB"
              f"  [+{overhead_pct:.0f}% params, +{(t_flops-v_flops)/v_flops*100:.0f}% FLOPs]")
        print()

        results[bb] = {
            "vanilla": {"params": v_params, "flops": v_flops, "latency_ms": v_lat, "peak_mem_mb": v_mem},
            "tsb":     {"params": t_params, "flops": t_flops, "latency_ms": t_lat, "peak_mem_mb": t_mem},
        }

    # SOTA DL baselines
    print("\n── SOTA DL Baselines ──────────────────────────────────────────────────")
    for name, model in [
        ("DLinear",       DLinear(SEQ_LEN, PRED_LEN, N_FEAT)),
        ("PatchTST",      PatchTST(SEQ_LEN, PRED_LEN, N_FEAT, use_revin=True)),
        ("iTransformer",  iTransformer(SEQ_LEN, PRED_LEN, N_FEAT)),
    ]:
        m = measure_sota(model, device)
        print(f"{name:<20} params={m['params']:>8,}  FLOPs={m['flops']/1e6:>7.3f}M  "
              f"lat={m['latency_ms']:.3f}ms  mem={m['peak_mem_mb']:.1f}MB")
        results[name] = m

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "compute_ext_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
