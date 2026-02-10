"""
Flexible TS_Booster with Component Toggles
===========================================
支援靈活的組件開關,用於 Ablation Study
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import DNAS_SeriesDecomp, MultiScaleExpert, SeriesDecomp, MultiScaleDecomp

class FlexibleTSBooster(nn.Module):
    """
    Flexible Time-Series Booster Module.

    支援組件開關:
    - use_decomp: 是否使用分解
    - use_experts: 是否使用 Multi-Scale Experts
    - use_gating: 是否使用自適應 Gating
    - use_dnas: 是否使用 DNAS (僅影響 decomposition)
    - use_multiscale_decomp: 是否使用 Multi-Scale Decomposition (Global Weights)
    """
    def __init__(
        self,
        base_model,
        seq_len,
        pred_len,
        input_channels,
        # Component Toggles
        use_decomp=True,
        use_experts=True,
        use_gating=True,
        use_dnas=False,  # Changed default to False
        use_multiscale_decomp=False,  # NEW: Multi-Scale Decomposition
        # Fixed Params (when use_dnas=False and use_multiscale_decomp=False)
        fixed_decomp_kernel=25,
        # Multi-Scale Config (when use_multiscale_decomp=True)
        multiscale_kernels=[7, 13, 25, 49],
        # DNAS Config (when use_dnas=True, only for decomp)
        decomp_dnas_kernels=[7, 13, 25, 49, 'identity'],
        # Expert Config
        expert_kernels=[3, 5, 7],
        trend_hidden_dim=32,
        seasonal_hidden_dim=32,
        dropout_p=0.3,
        # Legacy params (ignored, kept for backward compat)
        trend_dnas_kernels=None,
        seasonal_dnas_kernels=None,
        fixed_trend_kernel=None,
        fixed_seasonal_kernel=None,
    ):
        super(FlexibleTSBooster, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_channels = input_channels

        # Component flags
        self.use_decomp = use_decomp
        self.use_experts = use_experts
        self.use_gating = use_gating
        self.use_dnas = use_dnas
        self.use_multiscale_decomp = use_multiscale_decomp
        self.gating_fixed = True  # For compatibility with trainer

        # Sanity check: cannot use both DNAS and MultiScale at the same time
        if self.use_dnas and self.use_multiscale_decomp:
            raise ValueError("Cannot use both use_dnas=True and use_multiscale_decomp=True simultaneously")

        # Backbone(s): Create 2 instances when decomp enabled (DLinear-style)
        if self.use_decomp:
            # Two separate backbones for trend and seasonal components
            self.backbone_trend = copy.deepcopy(base_model)
            self.backbone_seasonal = copy.deepcopy(base_model)
            self.base_model = None  # Not used when decomp enabled
        else:
            # Single backbone for vanilla configuration
            self.base_model = base_model
            self.backbone_trend = None
            self.backbone_seasonal = None

        # 1. Decomposition (DNAS / MultiScale / Fixed)
        if self.use_decomp:
            if self.use_dnas:
                # DNAS: Differentiable Architecture Search for kernel selection
                self.decomp = DNAS_SeriesDecomp(
                    input_channels=input_channels,
                    candidate_kernels=decomp_dnas_kernels,
                    use_learnable=True
                )
            elif self.use_multiscale_decomp:
                # Multi-Scale: Learnable weighted combination of multiple kernels
                self.decomp = MultiScaleDecomp(
                    input_channels=input_channels,
                    kernels=multiscale_kernels,
                    use_learnable=True
                )
            else:
                # Fixed: Single fixed kernel
                self.decomp = SeriesDecomp(
                    kernel_size=fixed_decomp_kernel,
                    input_channels=input_channels,
                    use_learnable=True
                )
        else:
            self.decomp = None

        # 2. Multi-Scale Experts (always Inception-style, no DNAS)
        if self.use_experts:
            self.trend_expert = MultiScaleExpert(
                seq_len=seq_len, pred_len=pred_len, input_channels=input_channels,
                hidden_dim=trend_hidden_dim, kernels=expert_kernels, dropout_p=dropout_p
            )
            self.seasonal_expert = MultiScaleExpert(
                seq_len=seq_len, pred_len=pred_len, input_channels=input_channels,
                hidden_dim=seasonal_hidden_dim, kernels=expert_kernels, dropout_p=dropout_p
            )
        else:
            self.trend_expert = None
            self.seasonal_expert = None

        # DNAS Interface
        self.dnas_tau = 1.0

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, S, C = x.shape

        # A. Decomposition + Backbone Prediction (DLinear-style when decomp enabled)
        if self.use_decomp and self.decomp is not None:
            # Step 1: Decompose into trend and seasonal
            if self.use_dnas or self.use_multiscale_decomp:
                # Both DNAS and MultiScale return weights
                trend_part, seasonal_part, decomp_weights = self.decomp(x)
            else:
                # Fixed decomposition doesn't return weights
                trend_part, seasonal_part = self.decomp(x)
                decomp_weights = None

            # Step 2: Feed decomposed components to separate backbones
            y_trend_base = self.backbone_trend(trend_part)
            y_seasonal_base = self.backbone_seasonal(seasonal_part)

            # Handle tuple outputs
            if isinstance(y_trend_base, tuple):
                y_trend_base = y_trend_base[0]
            if isinstance(y_seasonal_base, tuple):
                y_seasonal_base = y_seasonal_base[0]

            # Keep 3D shape for compatibility with both univariate and multivariate
            # Ensure outputs are 3D: (Batch, Pred_Len, Channels)
            if y_trend_base.dim() == 2:
                y_trend_base = y_trend_base.unsqueeze(-1)  # (B, P) -> (B, P, 1)
            if y_seasonal_base.dim() == 2:
                y_seasonal_base = y_seasonal_base.unsqueeze(-1)  # (B, P) -> (B, P, 1)

            # Step 3: Combine backbone predictions (DLinear: model(A) + model(B))
            y_base = y_trend_base + y_seasonal_base
        else:
            # Vanilla: Single backbone on original x
            y_base = self.base_model(x)
            if isinstance(y_base, tuple):
                y_base = y_base[0]
            # Ensure 3D shape
            if y_base.dim() == 2:
                y_base = y_base.unsqueeze(-1)  # (B, P) -> (B, P, 1)

            # No decomposition
            trend_part = x
            seasonal_part = torch.zeros_like(x)
            decomp_weights = None

        # C. Expert Prediction (MultiScaleExpert returns single tensor, no weights)
        if self.use_experts and self.trend_expert is not None:
            y_trend = self.trend_expert(trend_part)
            y_seasonal = self.seasonal_expert(seasonal_part)
            y_expert = y_trend + y_seasonal
        else:
            y_expert = torch.zeros_like(y_base)

        # D. Simple Additive Fusion (gate=1)
        if self.use_experts:
            y_final = y_base + y_expert
            gate_val = torch.tensor(1.0, device=x.device)
        else:
            y_final = y_base
            gate_val = torch.tensor(0.0, device=x.device)

        # Handle dimension for compatibility (always squeeze to 2D for univariate output)
        # If single-channel: squeeze; if multi-channel: take first channel
        if y_final.dim() == 3:
            if y_final.size(-1) == 1:
                y_final = y_final.squeeze(-1)  # [B, P, 1] -> [B, P]
            else:
                y_final = y_final[..., 0]  # [B, P, C] -> [B, P], take first channel

        if y_base.dim() == 3:
            if y_base.size(-1) == 1:
                y_base = y_base.squeeze(-1)
            else:
                y_base = y_base[..., 0]

        if y_expert.dim() == 3:
            if y_expert.size(-1) == 1:
                y_expert = y_expert.squeeze(-1)
            else:
                y_expert = y_expert[..., 0]

        # Prepare return values (keep format for compatibility)
        zero = torch.tensor(0.0, device=x.device)
        weights_to_return = torch.stack([gate_val, zero, zero])

        return y_final, y_base, weights_to_return, trend_part, seasonal_part, y_expert

    # --- DNAS Interface (decomp only) ---
    def set_tau(self, tau):
        self.dnas_tau = tau
        if self.use_dnas and self.decomp is not None and hasattr(self.decomp, 'set_tau'):
            self.decomp.set_tau(tau)

    def set_gating_status(self, fixed=True):
        """Set gating status (kept for compatibility, no-op since we use fixed gate=1)"""
        self.gating_fixed = fixed

    def toggle_dnas_mode(self, mode="warmup"):
        if self.use_dnas and self.decomp is not None and hasattr(self.decomp, 'toggle_dnas_mode'):
            self.decomp.toggle_dnas_mode(mode)

    def get_dnas_status(self):
        if not self.use_dnas:
            return {}
        status = {}
        if self.decomp is not None and hasattr(self.decomp, 'get_dnas_status'):
            status["decomp"] = self.decomp.get_dnas_status()
        return status

    def get_arch_parameters(self):
        if not self.use_dnas:
            return []
        params = []
        if self.decomp is not None and hasattr(self.decomp, 'get_arch_parameters'):
            params.extend(list(self.decomp.get_arch_parameters()))
        return params
