"""
Compute cost analysis: parameter count, training time, inference latency.

Usage:
  cd conference
  uv run python experiments/compute_cost.py
"""

import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.layers import LinearBackbone
from src.models.ts_booster import TSBoosterLite
from src.engine.trainer import train_booster, train_baseline
from src.configs.config import CONFIG
from src.utils.data_loader import prepare_data

DATASET_DIR = Path("../dataset")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    config = CONFIG.copy()
    seq_len, pred_len = config["seq_len"], config["pred_len"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parameter count
    vanilla = LinearBackbone(seq_len, 2, pred_len)
    booster = TSBoosterLite(
        base_model=LinearBackbone(seq_len, 2, pred_len),
        seq_len=seq_len, pred_len=pred_len, input_channels=2,
        decomp_kernel=config["decomp_kernel"],
        expert_kernels=config["expert_kernels"],
        expert_hidden_dim=config["expert_hidden_dim"],
        dropout_p=config["dropout"],
    )

    v_params = count_params(vanilla)
    b_params = count_params(booster)
    overhead = b_params - v_params

    print(f"\n{'='*50}")
    print(f"PARAMETER COUNT")
    print(f"{'='*50}")
    print(f"  Vanilla Linear:  {v_params:>8,} params")
    print(f"  TSB-Lite:        {b_params:>8,} params")
    print(f"  Overhead:        {overhead:>8,} params (+{overhead})")

    # Training time comparison
    ds = "USD_TWD"
    path = DATASET_DIR / f"{ds}.csv"
    if not path.exists():
        print(f"\n  {path} not found, skipping timing test.")
        return

    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=seq_len, horizon=pred_len, batch_size=config["batch_size"]
    )

    # Vanilla training time
    vanilla = LinearBackbone(seq_len, 2, pred_len)
    t0 = time.time()
    vanilla = train_baseline(
        train_loader, val_loader, device, vanilla,
        num_epochs=config["epochs"], lr=config["lr"], verbose=False,
    )
    t_vanilla = time.time() - t0

    # TSB-Lite training time
    booster = TSBoosterLite(
        base_model=LinearBackbone(seq_len, 2, pred_len),
        seq_len=seq_len, pred_len=pred_len, input_channels=2,
        decomp_kernel=config["decomp_kernel"],
        expert_kernels=config["expert_kernels"],
        expert_hidden_dim=config["expert_hidden_dim"],
        dropout_p=config["dropout"],
    )
    t0 = time.time()
    booster = train_booster(
        train_loader, val_loader, device, booster, config, verbose=False,
    )
    t_booster = time.time() - t0

    print(f"\n{'='*50}")
    print(f"TRAINING TIME ({ds})")
    print(f"{'='*50}")
    print(f"  Vanilla:  {t_vanilla:.1f}s")
    print(f"  TSB-Lite: {t_booster:.1f}s")
    print(f"  Ratio:    {t_booster/t_vanilla:.2f}x")

    # Inference latency
    booster.eval()
    vanilla.eval()
    booster = booster.to(device)
    vanilla = vanilla.to(device)

    n_runs = 100
    dummy = torch.randn(1, seq_len, 2).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            vanilla(dummy)
            booster(dummy)

    # Vanilla inference
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            vanilla(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_v_inf = (time.time() - t0) / n_runs * 1000

    # TSB inference
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            booster(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_b_inf = (time.time() - t0) / n_runs * 1000

    print(f"\n{'='*50}")
    print(f"INFERENCE LATENCY (single sample, {n_runs} runs)")
    print(f"{'='*50}")
    print(f"  Vanilla:  {t_v_inf:.3f} ms")
    print(f"  TSB-Lite: {t_b_inf:.3f} ms")
    print(f"  Ratio:    {t_b_inf/t_v_inf:.2f}x")


if __name__ == "__main__":
    main()
