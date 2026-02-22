"""
Computational Cost Analysis: Vanilla vs TS-Booster.

Measures parameter count, training time per epoch, and inference time
across all 4 backbone types.

Usage:
  uv run python experiments/compute_cost.py
"""

import sys, time, random
import numpy as np
import torch
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.factory import create_model_pair
from src.models.layers import (
    LinearBackbone,
    LSTMBackbone,
    GRUBackbone,
    TransformerBackbone,
)
from src.models.ts_booster import TS_Booster
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data

SEQ_LEN = 30
PRED_LEN = 3
INPUT_CH = 2
WARMUP_ITERS = 5
MEASURE_ITERS = 20


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference(model, device, batch_size=32, seq_len=SEQ_LEN, input_ch=INPUT_CH):
    """Measure average inference time (ms) over multiple iterations."""
    model.eval()
    x = torch.randn(batch_size, seq_len, input_ch, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(MEASURE_ITERS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return np.mean(times), np.std(times)


def measure_train_epoch(model, train_loader, device, config):
    """Measure one training epoch time (seconds)."""
    from src.utils.metrics import HybridDirectionalLoss
    import torch.nn.functional as F

    model.train()
    model = model.to(device)
    criterion = HybridDirectionalLoss(dir_weight=config.get("dir_weight", 0.5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Warmup 1 batch
    for batch in train_loader:
        x = batch["raw_input"].to(device)
        y = batch["target"].to(device)
        optimizer.zero_grad()
        out = model(x)
        pred = out[0] if isinstance(out, tuple) else out
        loss = criterion(pred, y, x[:, -1, 0:1])
        loss.backward()
        optimizer.step()
        break

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure full epoch
    t0 = time.perf_counter()
    for batch in train_loader:
        x = batch["raw_input"].to(device)
        y = batch["target"].to(device)
        optimizer.zero_grad()
        out = model(x)
        pred = out[0] if isinstance(out, tuple) else out
        prev_val = x[:, -1, 0:1]
        loss = criterion(pred, y, prev_val)

        # Match trainer's multi-objective loss for TS-Booster
        if isinstance(out, tuple) and len(out) > 5:
            y_base = out[1]
            y_expert = out[5]
            if y_expert is not None and y_base is not None:
                residual = y - y_base.detach()
                loss = (
                    loss
                    + 0.1 * F.mse_loss(y_expert, residual)
                    + 0.1 * F.mse_loss(y_base, y)
                )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return t1 - t0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    config = BEST_STAGE3_CONFIG.copy()
    config["epochs"] = 100
    config["input_channels"] = INPUT_CH

    # Load one dataset for training time measurement
    df = pd.read_csv("dataset/USD_TWD.csv")
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32
    )
    n_train_batches = len(train_loader)
    print(f"Training batches per epoch: {n_train_batches}")

    backbones = ["linear", "lstm", "gru", "transformer"]
    rows = []

    for bb in backbones:
        print(f"\n{'=' * 60}")
        print(f"Backbone: {bb.upper()}")
        print(f"{'=' * 60}")

        config["backbone_type"] = bb
        set_seed(42)
        vanilla, booster = create_model_pair(config, SEQ_LEN, PRED_LEN, INPUT_CH)
        vanilla = vanilla.to(device)
        booster = booster.to(device)

        # Parameter count
        v_total, v_train = count_params(vanilla)
        b_total, b_train = count_params(booster)
        overhead = b_train - v_train
        overhead_pct = overhead / v_train * 100

        print(
            f"  Params — Vanilla: {v_train:,}  |  Booster: {b_train:,}  |  Overhead: +{overhead:,} (+{overhead_pct:.1f}%)"
        )

        # Inference time
        v_inf_mean, v_inf_std = measure_inference(vanilla, device)
        b_inf_mean, b_inf_std = measure_inference(booster, device)
        inf_ratio = b_inf_mean / v_inf_mean

        print(
            f"  Inference — Vanilla: {v_inf_mean:.2f}±{v_inf_std:.2f}ms  |  Booster: {b_inf_mean:.2f}±{b_inf_std:.2f}ms  |  Ratio: {inf_ratio:.2f}x"
        )

        # Training time per epoch
        set_seed(42)
        vanilla2, _ = create_model_pair(config, SEQ_LEN, PRED_LEN, INPUT_CH)
        v_epoch = measure_train_epoch(vanilla2, train_loader, device, config)

        set_seed(42)
        _, booster2 = create_model_pair(config, SEQ_LEN, PRED_LEN, INPUT_CH)
        b_epoch = measure_train_epoch(booster2, train_loader, device, config)
        train_ratio = b_epoch / v_epoch

        print(
            f"  Epoch — Vanilla: {v_epoch:.3f}s  |  Booster: {b_epoch:.3f}s  |  Ratio: {train_ratio:.2f}x"
        )

        rows.append(
            {
                "Backbone": bb.capitalize(),
                "Vanilla Params": v_train,
                "Booster Params": b_train,
                "Overhead": overhead,
                "Overhead %": overhead_pct,
                "Vanilla Inf (ms)": v_inf_mean,
                "Booster Inf (ms)": b_inf_mean,
                "Inf Ratio": inf_ratio,
                "Vanilla Epoch (s)": v_epoch,
                "Booster Epoch (s)": b_epoch,
                "Epoch Ratio": train_ratio,
            }
        )

    # === Summary Table ===
    print(f"\n\n{'=' * 90}")
    print("SUMMARY TABLE")
    print(f"{'=' * 90}")
    print(
        f"{'Backbone':<13} | {'V.Params':>9} | {'B.Params':>9} | {'Overhead':>12} | "
        f"{'V.Inf(ms)':>9} | {'B.Inf(ms)':>9} | {'Inf.×':>5} | "
        f"{'V.Epoch(s)':>10} | {'B.Epoch(s)':>10} | {'Epo.×':>5}"
    )
    print("-" * 110)
    for r in rows:
        print(
            f"  {r['Backbone']:<11} | {r['Vanilla Params']:>9,} | {r['Booster Params']:>9,} | "
            f"+{r['Overhead']:>6,} ({r['Overhead %']:>4.1f}%) | "
            f"{r['Vanilla Inf (ms)']:>9.2f} | {r['Booster Inf (ms)']:>9.2f} | {r['Inf Ratio']:>5.2f} | "
            f"{r['Vanilla Epoch (s)']:>10.3f} | {r['Booster Epoch (s)']:>10.3f} | {r['Epoch Ratio']:>5.2f}"
        )

    # === LaTeX Table ===
    print(f"\n{'=' * 90}")
    print("LATEX TABLE")
    print(f"{'=' * 90}\n")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(
        r"\caption{Computational overhead of TS-Booster across backbone architectures.}"
    )
    print(r"\label{tab:compute_cost}")
    print(r"\begin{tabular}{lrrcrrc}")
    print(r"\toprule")
    print(
        r"Backbone & \multicolumn{2}{c}{Parameters} & Overhead & \multicolumn{2}{c}{Inference (ms)} & Train \\"
    )
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){5-6}")
    print(r" & Vanilla & Booster & & Vanilla & Booster & Ratio \\")
    print(r"\midrule")
    for r in rows:
        print(
            f"{r['Backbone']} & {r['Vanilla Params']:,} & {r['Booster Params']:,} & "
            f"+{r['Overhead']:,} ({r['Overhead %']:.1f}\\%) & "
            f"{r['Vanilla Inf (ms)']:.2f} & {r['Booster Inf (ms)']:.2f} & "
            f"{r['Epoch Ratio']:.2f}$\\times$ \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
