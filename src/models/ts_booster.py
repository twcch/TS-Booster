import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import DNAS_SeriesDecomp, MultiScaleExpert

class TS_Booster(nn.Module):
    """
    Time-Series Booster Module (Adapter Pattern).
    Wraps any base model (backbone) and enhances it with:
    1. Adaptive Decomposition (DNAS)
    2. Multi-Scale Experts (Inception-style)
    3. Simple Additive Fusion (gate=1)

    Formula: Y_final = Y_Base + Y_Expert
    """
    def __init__(
        self,
        base_model,
        seq_len,
        pred_len,
        input_channels,
        # DNAS Config (decomp only)
        decomp_dnas_kernels=[7, 13, 25, 49, 'identity'],
        # Expert Config
        expert_kernels=[3, 5, 7],
        trend_hidden_dim=32,
        seasonal_hidden_dim=32,
        dropout_p=0.3,
        # Legacy params (ignored, kept for backward compat)
        trend_dnas_kernels=None,
        seasonal_dnas_kernels=None,
    ):
        super(TS_Booster, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_channels = input_channels

        # Backbone(s): Always use 2 instances for DLinear-style decomposition
        self.backbone_trend = copy.deepcopy(base_model)
        self.backbone_seasonal = copy.deepcopy(base_model)

        # 1. Adaptive Decomposition (DNAS)
        self.decomp = DNAS_SeriesDecomp(
            input_channels=input_channels,
            candidate_kernels=decomp_dnas_kernels,
            use_learnable=True
        )

        # 2. Multi-Scale Experts (Inception-style, no DNAS)
        self.trend_expert = MultiScaleExpert(
            seq_len=seq_len, pred_len=pred_len, input_channels=input_channels,
            hidden_dim=trend_hidden_dim, kernels=expert_kernels, dropout_p=dropout_p
        )
        self.seasonal_expert = MultiScaleExpert(
            seq_len=seq_len, pred_len=pred_len, input_channels=input_channels,
            hidden_dim=seasonal_hidden_dim, kernels=expert_kernels, dropout_p=dropout_p
        )

        # DNAS Interface
        self.dnas_tau = 1.0
        self.gating_fixed = True  # For compatibility with trainer

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, S, C = x.shape

        # A. Decomposition (DNAS)
        trend_part, seasonal_part, decomp_weights = self.decomp(x)

        # B. Backbone Prediction (System 1) - DLinear-style: model(trend) + model(seasonal)
        y_trend_base = self.backbone_trend(trend_part)
        y_seasonal_base = self.backbone_seasonal(seasonal_part)

        # Handle tuple outputs and dimension mismatches
        if isinstance(y_trend_base, tuple):
            y_trend_base = y_trend_base[0]
        if isinstance(y_seasonal_base, tuple):
            y_seasonal_base = y_seasonal_base[0]

        if y_trend_base.dim() == 3:
            y_trend_base = y_trend_base.squeeze(-1) if y_trend_base.size(-1) == 1 else y_trend_base[..., 0]
        if y_seasonal_base.dim() == 3:
            y_seasonal_base = y_seasonal_base.squeeze(-1) if y_seasonal_base.size(-1) == 1 else y_seasonal_base[..., 0]

        # Combine backbone predictions
        y_base = y_trend_base + y_seasonal_base

        # C. Expert Prediction (System 2)
        y_trend_expert = self.trend_expert(trend_part)
        y_seasonal_expert = self.seasonal_expert(seasonal_part)
        y_expert = y_trend_expert + y_seasonal_expert

        # D. Simple Additive Fusion (gate=1)
        y_final = y_base + y_expert

        # Aux returns for logging (fixed gate=1 for compatibility)
        one = torch.tensor(1.0, device=x.device)
        zero = torch.tensor(0.0, device=x.device)
        weights_to_return = torch.stack([one, zero, zero])

        return y_final, y_base, weights_to_return, trend_part, seasonal_part, y_expert

    # --- DNAS Interface (decomp only) ---
    def set_tau(self, tau):
        self.dnas_tau = tau
        self.decomp.set_tau(tau)

    def set_gating_status(self, fixed=True):
        """Set gating status (kept for compatibility, no-op since we use fixed gate=1)"""
        self.gating_fixed = fixed

    def toggle_dnas_mode(self, mode="warmup"):
        self.decomp.toggle_dnas_mode(mode)

    def get_dnas_status(self):
        return {
            "decomp": self.decomp.get_dnas_status(),
        }

    def get_arch_parameters(self):
        yield from self.decomp.get_arch_parameters()
