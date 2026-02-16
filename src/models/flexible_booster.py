"""
Flexible TS_Booster with Component Toggles for Ablation Study.

4 configs:
  1. Vanilla      — backbone only (not wrapped, used directly)
  2. +Decomp      — multi-scale decomp + 2 backbones (DLinear-style, no expert)
  3. +Expert      — single backbone + multi-round expert on raw input
  4. TS-Booster   — single backbone + decomp-enriched multi-round expert

Key design: Decomposition enriches the EXPERT, not the backbone.
TS-Booster backbone is identical to +Expert backbone (single, on raw).
"""

import copy
import torch
import torch.nn as nn
from src.models.layers import MultiScaleDecomp, create_expert


class FlexibleTSBooster(nn.Module):
    def __init__(
        self,
        base_model,
        seq_len,
        pred_len,
        input_channels,
        # Component Toggles
        use_decomp=True,
        use_experts=True,
        # Config
        decomp_kernels=[7, 13],
        expert_kernels=[3, 5, 7],
        expert_hidden_dim=32,
        dropout_p=0.3,
        n_boost_rounds=1,
        expert_type='cnn',
        # Legacy params (ignored)
        n_freq=None,
        decomp_kernel=None, trend_hidden_dim=None, seasonal_hidden_dim=None,
        use_multiscale_decomp=None, use_gating=None, use_dnas=None,
        fixed_decomp_kernel=None, multiscale_kernels=None,
        decomp_dnas_kernels=None, trend_dnas_kernels=None,
        seasonal_dnas_kernels=None, fixed_trend_kernel=None,
        fixed_seasonal_kernel=None,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_channels = input_channels
        self.use_decomp = use_decomp
        self.use_experts = use_experts

        # Legacy compat
        if trend_hidden_dim is not None and expert_hidden_dim == 32:
            expert_hidden_dim = trend_hidden_dim

        # === Backbone Setup ===
        if self.use_decomp and not self.use_experts:
            self.backbone_trend = copy.deepcopy(base_model)
            self.backbone_seasonal = copy.deepcopy(base_model)
            self.base_model = None
        else:
            self.base_model = base_model
            self.backbone_trend = None
            self.backbone_seasonal = None

        # === Decomposition (Learnable Multi-Scale) ===
        if self.use_decomp:
            self.decomp = MultiScaleDecomp(kernels=decomp_kernels)
        else:
            self.decomp = None

        # === Expert Setup ===
        self.experts = None
        self.n_boost_rounds = 0

        if self.use_experts:
            self.n_boost_rounds = n_boost_rounds
            expert_input_ch = input_channels * 2 if self.use_decomp else input_channels
            self.experts = nn.ModuleList()
            for _ in range(n_boost_rounds):
                self.experts.append(create_expert(
                    expert_type=expert_type,
                    seq_len=seq_len, pred_len=pred_len,
                    input_channels=expert_input_ch,
                    hidden_dim=expert_hidden_dim, kernels=expert_kernels,
                    dropout_p=dropout_p
                ))

    def forward(self, x):
        B, S, C = x.shape

        # === Base Prediction ===
        if self.use_decomp and not self.use_experts:
            trend_part, seasonal_part = self.decomp(x)

            y_trend_base = self.backbone_trend(trend_part)
            y_seasonal_base = self.backbone_seasonal(seasonal_part)

            if isinstance(y_trend_base, tuple): y_trend_base = y_trend_base[0]
            if isinstance(y_seasonal_base, tuple): y_seasonal_base = y_seasonal_base[0]
            if y_trend_base.dim() == 3:
                y_trend_base = y_trend_base.squeeze(-1) if y_trend_base.size(-1) == 1 else y_trend_base[..., 0]
            if y_seasonal_base.dim() == 3:
                y_seasonal_base = y_seasonal_base.squeeze(-1) if y_seasonal_base.size(-1) == 1 else y_seasonal_base[..., 0]

            y_base = y_trend_base + y_seasonal_base
        else:
            y_base = self.base_model(x)
            if isinstance(y_base, tuple): y_base = y_base[0]
            if y_base.dim() == 3:
                y_base = y_base.squeeze(-1) if y_base.size(-1) == 1 else y_base[..., 0]
            trend_part = x
            seasonal_part = torch.zeros_like(x)

        # === Decomposition for Expert Enrichment (TS-Booster only) ===
        if self.use_decomp and self.use_experts:
            trend_part, seasonal_part = self.decomp(x)

        # === Expert Prediction ===
        if self.use_experts and self.experts is not None:
            if self.use_decomp:
                x_expert = torch.cat([x, seasonal_part], dim=-1)
            else:
                x_expert = x

            y_expert = torch.zeros_like(y_base)
            expert_rounds = []
            for expert in self.experts:
                y_round = expert(x_expert)
                expert_rounds.append(y_round)
                y_expert = y_expert + y_round

            y_final = y_base + y_expert
            y_expert_out = y_expert
        else:
            y_final = y_base
            y_expert_out = None
            expert_rounds = None

        gate_val = torch.tensor(1.0 if self.use_experts else 0.0, device=x.device)
        zero = torch.tensor(0.0, device=x.device)
        weights_to_return = torch.stack([gate_val, zero, zero])

        return y_final, y_base, weights_to_return, trend_part, seasonal_part, y_expert_out, expert_rounds

    # Compatibility stubs for trainer
    def set_tau(self, tau): pass
    def set_gating_status(self, fixed=True): pass
    def toggle_dnas_mode(self, mode="warmup"): pass
    def get_dnas_status(self): return {}
    def get_arch_parameters(self): return []
    def get_model_parameters(self): return list(self.parameters())

    def get_multiscale_weights(self):
        if self.decomp is not None and hasattr(self.decomp, '_last_weights') and self.decomp._last_weights is not None:
            w = self.decomp._last_weights.cpu().tolist()
            return {str(k): v for k, v in zip(self.decomp.kernels, w)}
        return None
