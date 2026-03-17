"""
TS-Booster Lite — Simplified conference version.

Architecture:
  Input x
    ├── Backbone(x)          → y_base
    ├── MA(x)                → trend
    └── Expert([x, trend])   → y_expert

  y_final = y_base + y_expert

Simplifications vs thesis version:
  - Single fixed moving average (no learnable multi-scale)
  - Single expert round (no multi-round boosting)
  - No SNR gate (direct addition)
  - Simpler loss: MSE + aux residual loss
"""

import torch
import torch.nn as nn
from src.models.layers import MovingAvgDecomp, MultiScaleExpert


class TSBoosterLite(nn.Module):
    """
    Lightweight Time-Series Booster (Adapter Pattern).
    Wraps any backbone and enhances it with:
    1. Moving Average Decomposition (fixed kernel)
    2. Trend-Enriched Expert (learns residuals)
    3. Additive Fusion: y_final = y_base + y_expert
    """

    def __init__(
        self,
        base_model,
        seq_len,
        pred_len,
        input_channels,
        decomp_kernel=7,
        expert_kernels=[3, 5, 7],
        expert_hidden_dim=32,
        dropout_p=0.3,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_channels = input_channels

        # Backbone on raw input
        self.backbone = base_model

        # Fixed MA decomposition (for trend enrichment)
        self.decomp = MovingAvgDecomp(kernel_size=decomp_kernel)

        # Single expert on [raw, trend]
        expert_input_ch = input_channels * 2  # [raw, trend]
        self.expert = MultiScaleExpert(
            seq_len=seq_len,
            pred_len=pred_len,
            input_channels=expert_input_ch,
            hidden_dim=expert_hidden_dim,
            kernels=expert_kernels,
            dropout_p=dropout_p,
        )

    def forward(self, x):
        # 1. Decomposition
        trend, seasonal = self.decomp(x)

        # 2. Backbone prediction on raw
        y_base = self.backbone(x)
        if isinstance(y_base, tuple):
            y_base = y_base[0]
        if y_base.dim() == 3:
            y_base = y_base.squeeze(-1) if y_base.size(-1) == 1 else y_base[..., 0]

        # 3. Expert on [raw, trend]
        x_enriched = torch.cat([x, trend], dim=-1)
        y_expert = self.expert(x_enriched)

        # 4. Additive fusion
        y_final = y_base + y_expert

        return y_final, y_base, y_expert
