"""
Flexible TS_Booster with Component Toggles
===========================================
支援靈活的組件開關,用於 Ablation Study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import DNAS_SeriesDecomp, DNAS_CNNExpert, SeriesDecomp, CNNExpert

class FlexibleTSBooster(nn.Module):
    """
    Flexible Time-Series Booster Module.
    
    支援組件開關:
    - use_decomp: 是否使用分解
    - use_experts: 是否使用 CNN Experts
    - use_gating: 是否使用自適應 Gating
    - use_dnas: 是否使用 DNAS (否則使用固定參數)
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
        use_dnas=True,
        # Fixed Params (when use_dnas=False)
        fixed_decomp_kernel=25,
        fixed_trend_kernel=5,
        fixed_seasonal_kernel=5,
        # DNAS Config (when use_dnas=True)
        trend_dnas_kernels=[3, 5, 7, 9],
        seasonal_dnas_kernels=[3, 5, 7, 9],
        decomp_dnas_kernels=[7, 13, 25, 49, 'identity'],
        # Expert Config
        trend_hidden_dim=32,
        seasonal_hidden_dim=32,
        dropout_p=0.3
    ):
        super(FlexibleTSBooster, self).__init__()
        
        self.base_model = base_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_channels = input_channels
        
        # Component flags
        self.use_decomp = use_decomp
        self.use_experts = use_experts
        self.use_gating = use_gating
        self.use_dnas = use_dnas
        
        # 1. Decomposition
        if self.use_decomp:
            if self.use_dnas:
                self.decomp = DNAS_SeriesDecomp(
                    input_channels=input_channels,
                    candidate_kernels=decomp_dnas_kernels,
                    use_learnable=True
                )
            else:
                self.decomp = SeriesDecomp(
                    kernel_size=fixed_decomp_kernel,
                    input_channels=input_channels,
                    use_learnable=True
                )
        else:
            self.decomp = None
        
        # 2. CNN Experts
        if self.use_experts:
            if self.use_dnas:
                self.trend_expert = DNAS_CNNExpert(
                    seq_len=seq_len, pred_len=pred_len, input_channels=input_channels,
                    hidden_dim=trend_hidden_dim, candidate_kernels=trend_dnas_kernels, 
                    dropout_p=dropout_p
                )
                self.seasonal_expert = DNAS_CNNExpert(
                    seq_len=seq_len, pred_len=pred_len, input_channels=input_channels,
                    hidden_dim=seasonal_hidden_dim, candidate_kernels=seasonal_dnas_kernels, 
                    dropout_p=dropout_p
                )
            else:
                self.trend_expert = CNNExpert(
                    seq_len=seq_len, pred_len=pred_len, input_channels=input_channels,
                    hidden_dim=trend_hidden_dim, cnnExpert_KernelSize=fixed_trend_kernel,
                    dropout_p=dropout_p
                )
                self.seasonal_expert = CNNExpert(
                    seq_len=seq_len, pred_len=pred_len, input_channels=input_channels,
                    hidden_dim=seasonal_hidden_dim, cnnExpert_KernelSize=fixed_seasonal_kernel,
                    dropout_p=dropout_p
                )
        else:
            self.trend_expert = None
            self.seasonal_expert = None
        
        # 3. Gating Network
        if self.use_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(seq_len * input_channels, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            self.gate_net = None
        
        # Interfaces
        self.dnas_tau = 1.0
        self.gating_fixed = False

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, S, C = x.shape
        x_flat = x.reshape(B, -1)
        
        # A. Base Model Prediction
        y_base = self.base_model(x)
        if isinstance(y_base, tuple):
            y_base = y_base[0]
        
        # Handle dimension mismatch
        if y_base.dim() == 3:
            if y_base.size(-1) == 1:
                y_base = y_base.squeeze(-1)
            else:
                y_base = y_base[..., 0]
        
        # B. Expert Prediction
        if self.use_decomp and self.decomp is not None:
            # Decomposition
            if self.use_dnas:
                trend_part, seasonal_part, decomp_weights = self.decomp(x)
            else:
                trend_part, seasonal_part = self.decomp(x)
                decomp_weights = None
        else:
            # No decomposition: trend = x, seasonal = 0
            trend_part = x
            seasonal_part = torch.zeros_like(x)
            decomp_weights = None
        
        # Run Experts
        if self.use_experts and self.trend_expert is not None:
            if self.use_dnas:
                y_trend, w_trend = self.trend_expert(trend_part)
                y_seasonal, w_seasonal = self.seasonal_expert(seasonal_part)
            else:
                y_trend = self.trend_expert(trend_part)
                y_seasonal = self.seasonal_expert(seasonal_part)
                w_trend = None
                w_seasonal = None
            
            y_expert = y_trend + y_seasonal
        else:
            y_expert = torch.zeros_like(y_base)
            w_trend = None
            w_seasonal = None
        
        # C. Fusion (Gating)
        if self.use_gating and self.gate_net is not None:
            if self.gating_fixed:
                gate = torch.tensor(0.5, device=x.device)
            else:
                raw_gate = self.gate_net(x_flat)
                gate = 0.1 + 0.9 * raw_gate
            
            y_final = gate * y_expert + (1 - gate) * y_base
            gate_val = gate.mean()
        else:
            # No gating: simple addition or just base
            if self.use_experts:
                y_final = y_base + y_expert  # Fixed fusion
                gate_val = torch.tensor(0.5, device=x.device)
            else:
                y_final = y_base
                gate_val = torch.tensor(0.0, device=x.device)
        
        # Prepare return values
        t_dnas_max = w_trend.max() if w_trend is not None else torch.tensor(0.0, device=x.device)
        s_dnas_max = w_seasonal.max() if w_seasonal is not None else torch.tensor(0.0, device=x.device)
        
        weights_to_return = torch.stack([gate_val, t_dnas_max, s_dnas_max])
        
        return y_final, y_base, weights_to_return, trend_part, seasonal_part

    # --- DNAS Interface ---
    def set_tau(self, tau):
        self.dnas_tau = tau
        if self.use_dnas:
            if self.decomp is not None and hasattr(self.decomp, 'set_tau'):
                self.decomp.set_tau(tau)
            if self.trend_expert is not None and hasattr(self.trend_expert, 'set_tau'):
                self.trend_expert.set_tau(tau)
            if self.seasonal_expert is not None and hasattr(self.seasonal_expert, 'set_tau'):
                self.seasonal_expert.set_tau(tau)
    
    def set_gating_status(self, fixed=True):
        self.gating_fixed = fixed
    
    def toggle_dnas_mode(self, mode="warmup"):
        if self.use_dnas:
            if self.decomp is not None and hasattr(self.decomp, 'toggle_dnas_mode'):
                self.decomp.toggle_dnas_mode(mode)
            if self.trend_expert is not None and hasattr(self.trend_expert, 'toggle_dnas_mode'):
                self.trend_expert.toggle_dnas_mode(mode)
            if self.seasonal_expert is not None and hasattr(self.seasonal_expert, 'toggle_dnas_mode'):
                self.seasonal_expert.toggle_dnas_mode(mode)
    
    def get_dnas_status(self):
        if not self.use_dnas:
            return {}
        
        status = {}
        if self.decomp is not None and hasattr(self.decomp, 'get_dnas_status'):
            status["decomp"] = self.decomp.get_dnas_status()
        if self.trend_expert is not None and hasattr(self.trend_expert, 'get_dnas_status'):
            status["trend_expert"] = self.trend_expert.get_dnas_status()
        if self.seasonal_expert is not None and hasattr(self.seasonal_expert, 'get_dnas_status'):
            status["seasonal_expert"] = self.seasonal_expert.get_dnas_status()
        return status
    
    def get_arch_parameters(self):
        if not self.use_dnas:
            return []
        
        params = []
        if self.decomp is not None and hasattr(self.decomp, 'get_arch_parameters'):
            params.extend(list(self.decomp.get_arch_parameters()))
        if self.trend_expert is not None and hasattr(self.trend_expert, 'get_arch_parameters'):
            params.extend(list(self.trend_expert.get_arch_parameters()))
        if self.seasonal_expert is not None and hasattr(self.seasonal_expert, 'get_arch_parameters'):
            params.extend(list(self.seasonal_expert.get_arch_parameters()))
        return params
