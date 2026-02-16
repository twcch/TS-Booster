import torch
import torch.nn as nn
from src.models.layers import MultiScaleDecomp, create_expert


class TS_Booster(nn.Module):
    """
    Time-Series Booster Module (Adapter Pattern).
    Wraps any base model (backbone) and enhances it with:
    1. Learnable Multi-Scale Decomposition (weighted MA ensemble)
    2. Single Backbone on raw input
    3. Decomposition-Enriched Expert (raw + seasonal features)
    4. Multi-Round Boosting
    5. Additive Fusion: Y_final = Y_Base + Y_Expert
    """
    def __init__(
        self,
        base_model,
        seq_len,
        pred_len,
        input_channels,
        decomp_kernels=[7, 13],
        expert_kernels=[3, 5, 7],
        expert_hidden_dim=32,
        dropout_p=0.3,
        n_boost_rounds=1,
        expert_type='cnn',
        # Legacy/unused params
        n_freq=None,
        decomp_kernel=None, trend_hidden_dim=None, seasonal_hidden_dim=None,
        multiscale_kernels=None, decomp_dnas_kernels=None,
        trend_dnas_kernels=None, seasonal_dnas_kernels=None,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_channels = input_channels
        self.n_boost_rounds = n_boost_rounds

        # Single backbone on raw input
        self.backbone = base_model

        # Multi-Scale Decomposition (for expert enrichment)
        self.decomp = MultiScaleDecomp(kernels=decomp_kernels)

        # Multi-Round Boosting experts
        self.experts = nn.ModuleList()
        for _ in range(n_boost_rounds):
            self.experts.append(create_expert(
                expert_type=expert_type,
                seq_len=seq_len, pred_len=pred_len,
                input_channels=input_channels * 2,
                hidden_dim=expert_hidden_dim, kernels=expert_kernels,
                dropout_p=dropout_p
            ))

    def forward(self, x):
        B, S, C = x.shape

        # 1. Multi-Scale Decomposition (for expert enrichment)
        trend_part, seasonal_part = self.decomp(x)

        # 2. Backbone prediction (single backbone on raw)
        y_base = self.backbone(x)
        if isinstance(y_base, tuple): y_base = y_base[0]
        if y_base.dim() == 3:
            y_base = y_base.squeeze(-1) if y_base.size(-1) == 1 else y_base[..., 0]

        # 3. Enriched expert input: [raw, seasonal]
        x_enriched = torch.cat([x, seasonal_part], dim=-1)

        # 4. Multi-Round Boosting
        y_expert_total = torch.zeros_like(y_base)
        expert_rounds = []
        for expert in self.experts:
            y_round = expert(x_enriched)
            expert_rounds.append(y_round)
            y_expert_total = y_expert_total + y_round

        # 5. Additive fusion
        y_final = y_base + y_expert_total

        one = torch.tensor(1.0, device=x.device)
        zero = torch.tensor(0.0, device=x.device)
        weights_to_return = torch.stack([one, zero, zero])

        return y_final, y_base, weights_to_return, trend_part, seasonal_part, y_expert_total, expert_rounds

    # Compatibility stubs for trainer
    def set_tau(self, tau): pass
    def set_gating_status(self, fixed=True): pass
    def toggle_dnas_mode(self, mode="warmup"): pass
    def get_dnas_status(self): return {}
    def get_arch_parameters(self): return []
    def get_model_parameters(self): return list(self.parameters())

    def get_multiscale_weights(self):
        if hasattr(self.decomp, '_last_weights') and self.decomp._last_weights is not None:
            w = self.decomp._last_weights.cpu().tolist()
            return {str(k): v for k, v in zip(self.decomp.kernels, w)}
        return None
