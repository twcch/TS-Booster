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
        # Enrichment options
        seasonal_dropout_p=0.0,
        seasonal_scale=None,      # None, 'learned'
        expert_gate=None,         # None, 'learned', 'snr', 'adaptive'
        enrich_mode='seasonal',   # 'seasonal', 'trend', 'both'
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
        self.seasonal_dropout_p = seasonal_dropout_p
        self.enrich_mode = enrich_mode
        self.expert_gate_mode = expert_gate

        # Legacy compat
        if trend_hidden_dim is not None and expert_hidden_dim == 32:
            expert_hidden_dim = trend_hidden_dim

        # === Learned Seasonal Scale ===
        self.seasonal_scale_param = None
        if seasonal_scale == 'learned' and self.use_decomp and self.use_experts:
            self.seasonal_scale_param = nn.Parameter(torch.tensor(0.0))

        # === Expert Gate Variants ===
        self.expert_gate_param = None
        self.snr_gate = None
        self.adaptive_gate = None

        if self.use_experts and self.use_decomp:
            if expert_gate == 'learned':
                self.expert_gate_param = nn.Parameter(torch.tensor(1.0))  # sigmoid(1)≈0.73
            elif expert_gate == 'snr':
                # Gate based on seasonal/raw signal ratio: gate = sigmoid(w*snr + b)
                self.snr_gate_w = nn.Parameter(torch.tensor(1.0))
                self.snr_gate_b = nn.Parameter(torch.tensor(0.0))
                self.snr_gate = True
            elif expert_gate == 'input_snr':
                # Same as 'snr' but applied to enrichment INPUT, not expert output
                self.snr_gate_w = nn.Parameter(torch.tensor(1.0))
                self.snr_gate_b = nn.Parameter(torch.tensor(0.0))
                self.snr_gate = True  # reuse same gate computation
                # Flag to distinguish input vs output gating
            elif expert_gate == 'conservative_snr':
                # SNR gate with conservative init (gate starts nearly off)
                self.snr_gate_w = nn.Parameter(torch.tensor(1.0))
                self.snr_gate_b = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12
                self.snr_gate = True
            elif expert_gate == 'adaptive':
                # Gate computed from input statistics (mean, std per channel)
                # 2 channels → 4 stats (mean+std per channel) → 1 gate
                self.adaptive_gate = nn.Sequential(
                    nn.Linear(input_channels * 2, 1),
                )

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
            if self.use_decomp:
                if enrich_mode == 'both':
                    expert_input_ch = input_channels * 3
                else:
                    expert_input_ch = input_channels * 2
            else:
                expert_input_ch = input_channels
            self.experts = nn.ModuleList()
            for _ in range(n_boost_rounds):
                self.experts.append(create_expert(
                    expert_type=expert_type,
                    seq_len=seq_len, pred_len=pred_len,
                    input_channels=expert_input_ch,
                    hidden_dim=expert_hidden_dim, kernels=expert_kernels,
                    dropout_p=dropout_p
                ))

    def _compute_expert_gate(self, x, seasonal_part):
        """Compute expert gate value based on gate mode."""
        if self.expert_gate_param is not None:
            return torch.sigmoid(self.expert_gate_param)
        elif self.snr_gate is not None:
            # SNR: ratio of seasonal variation to raw variation
            seasonal_std = seasonal_part.std(dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
            raw_std = x.std(dim=1).mean(dim=-1, keepdim=True)  # [B, 1]
            snr = seasonal_std / (raw_std + 1e-8)  # [B, 1]
            gate = torch.sigmoid(self.snr_gate_w * snr + self.snr_gate_b)  # [B, 1]
            return gate.squeeze(-1)  # [B]
        elif self.adaptive_gate is not None:
            # Input statistics: mean and std per channel
            x_mean = x.mean(dim=1)  # [B, C]
            x_std = x.std(dim=1)    # [B, C]
            stats = torch.cat([x_mean, x_std], dim=-1)  # [B, 2C]
            gate = torch.sigmoid(self.adaptive_gate(stats))  # [B, 1]
            return gate.squeeze(-1)  # [B]
        return None

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
                seasonal_scaled = seasonal_part
                trend_scaled = trend_part

                # Apply seasonal scaling if learned
                if self.seasonal_scale_param is not None:
                    scale = torch.sigmoid(self.seasonal_scale_param)
                    seasonal_scaled = seasonal_part * scale
                    trend_scaled = trend_part * scale

                # Apply seasonal dropout during training
                if self.training and self.seasonal_dropout_p > 0:
                    mask = (torch.rand(B, 1, 1, device=x.device) > self.seasonal_dropout_p).float()
                    seasonal_scaled = seasonal_scaled * mask
                    trend_scaled = trend_scaled * mask

                # Apply input-level gating if configured
                if self.expert_gate_mode == 'input_snr':
                    input_gate = self._compute_expert_gate(x, seasonal_part)
                    if input_gate is not None:
                        if input_gate.dim() == 1:
                            input_gate = input_gate.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                        elif input_gate.dim() == 2:
                            input_gate = input_gate.unsqueeze(-1)  # [B, 1, 1]
                        seasonal_scaled = seasonal_scaled * input_gate
                        trend_scaled = trend_scaled * input_gate

                # Build enriched input based on mode
                if self.enrich_mode == 'seasonal':
                    x_expert = torch.cat([x, seasonal_scaled], dim=-1)
                elif self.enrich_mode == 'trend':
                    x_expert = torch.cat([x, trend_scaled], dim=-1)
                elif self.enrich_mode == 'both':
                    x_expert = torch.cat([x, trend_scaled, seasonal_scaled], dim=-1)
                else:
                    x_expert = torch.cat([x, seasonal_scaled], dim=-1)
            else:
                x_expert = x

            y_expert = torch.zeros_like(y_base)
            expert_rounds = []
            for expert in self.experts:
                y_round = expert(x_expert)
                expert_rounds.append(y_round)
                y_expert = y_expert + y_round

            # Apply expert gate (skip if input_snr - already applied at input level)
            gate = self._compute_expert_gate(x, seasonal_part) if (self.use_decomp and self.expert_gate_mode != 'input_snr') else None
            if gate is not None:
                if gate.dim() == 1:
                    gate = gate.unsqueeze(-1)  # [B, 1] for broadcasting
                y_expert = gate * y_expert

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
