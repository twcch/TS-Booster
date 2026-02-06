import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import DNAS_SeriesDecomp, DNAS_CNNExpert

class TS_Booster(nn.Module):
    """
    Time-Series Booster Module (Adapter Pattern).
    Wraps any base model (backbone) and enhances it with:
    1. Adaptive Decomposition (DNAS)
    2. Contextual Experts (CNN/DNAS)
    3. Adaptive Gating (Fusion)
    
    Formula: Y_final = Gate * Y_Expert + (1 - Gate) * Y_Base
    """
    def __init__(
        self, 
        base_model,
        seq_len, 
        pred_len, 
        input_channels,
        # DNAS Config
        trend_dnas_kernels=[3, 5, 7, 9],
        seasonal_dnas_kernels=[3, 5, 7, 9],
        decomp_dnas_kernels=[7, 13, 25, 49, 'identity'],  # 包含 identity
        # Expert Config
        trend_hidden_dim=32,
        seasonal_hidden_dim=32,
        dropout_p=0.3
    ):
        super(TS_Booster, self).__init__()
        
        self.base_model = base_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_channels = input_channels
        
        # 1. Adaptive Decomposition
        # Includes IdentityDecomp if 'identity' in kernels, so model can choose NOT to decompose
        self.decomp = DNAS_SeriesDecomp(
            input_channels=input_channels,
            candidate_kernels=decomp_dnas_kernels,
            use_learnable=True
        )
        
        # 2. Experts (System 2)
        # Process the decomposed components
        self.trend_expert = DNAS_CNNExpert(
            seq_len=seq_len, pred_len=pred_len, input_channels=input_channels,
            hidden_dim=trend_hidden_dim, candidate_kernels=trend_dnas_kernels, dropout_p=dropout_p
        )
        
        self.seasonal_expert = DNAS_CNNExpert(
            seq_len=seq_len, pred_len=pred_len, input_channels=input_channels,
            hidden_dim=seasonal_hidden_dim, candidate_kernels=seasonal_dnas_kernels, dropout_p=dropout_p
        )
        
        # 3. Gating Network (Fusion Weight)
        # Maps input x -> scalar gate [0, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(seq_len * input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Interfaces
        self.dnas_tau = 1.0
        self.gating_fixed = False

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, S, C = x.shape
        x_flat = x.reshape(B, -1)
        
        # A. Base Model Prediction (System 1)
        # Adapt: Ensure y_base is [Batch, Pred_Len]
        y_base = self.base_model(x)
        if isinstance(y_base, tuple):
            y_base = y_base[0] # Assume first element is the main prediction
        
        # B. Expert Prediction (System 2)
        # 1. Adaptive Decomp
        trend_part, seasonal_part, decomp_weights = self.decomp(x)
        
        # 2. Run Experts
        # trend_part might be x (if Identity), seasonal might be 0
        y_trend, w_trend = self.trend_expert(trend_part)
        y_seasonal, w_seasonal = self.seasonal_expert(seasonal_part)
        
        y_expert = y_trend + y_seasonal
        
        # C. Fusion (Gating)
        if self.gating_fixed:
            gate = torch.tensor(0.5, device=x.device)
        else:
            raw_gate = self.gate_net(x_flat) # [B, 1]
            # Floor Protection (min 10% expert)
            gate = 0.1 + 0.9 * raw_gate 
            
        # shape check: y_base might be [B, P, C], y_expert is [B, P]
        if y_base.dim() == 3:
            # Simple projection: flatten last dim if it matches Pred_Len?
            # Or assume channel 0 is target?
            # Usually for forecasting y is [B, P]. If base returns [B, P, 1], squeeze it.
            if y_base.size(-1) == 1:
                y_base = y_base.squeeze(-1)
            else:
                 # If multiple channels, maybe we want just the target channel (usually 0)?
                 # Or use an adapter? For now, let's take mean or channel 0.
                 # Safest generic way: Take first channel
                 y_base = y_base[..., 0] 
                 
        # C. Fusion (Gating)
        if self.gating_fixed:
            gate = torch.tensor(0.5, device=x.device)
        else:
            raw_gate = self.gate_net(x_flat) # [B, 1]
            # Floor Protection (min 10% expert)
            # Logic: 0.1 + 0.9 * Sigmoid
            # Note: gate_net ends with Sigmoid, so raw_gate is in [0, 1]
            gate = 0.1 + 0.9 * raw_gate 
            
        y_final = gate * y_expert + (1 - gate) * y_base
        
        # Aux returns for logging
        # [Gate_Value, Max_Trend_DNAS, Max_Seas_DNAS]
        t_dnas_max = w_trend.max() if w_trend is not None else 0.0
        s_dnas_max = w_seasonal.max() if w_seasonal is not None else 0.0
        
        # Return gate mean
        gate_val = gate.mean()
        
        weights_to_return = torch.stack([gate_val, t_dnas_max, s_dnas_max])
        
        return y_final, y_base, weights_to_return, trend_part, seasonal_part

    # --- DNAS Interface Pass-through ---
    def set_tau(self, tau):
        self.dnas_tau = tau
        self.decomp.set_tau(tau)
        self.trend_expert.set_tau(tau)
        self.seasonal_expert.set_tau(tau)
        
    def set_gating_status(self, fixed=True):
        self.gating_fixed = fixed
        
    def toggle_dnas_mode(self, mode="warmup"):
        self.decomp.toggle_dnas_mode(mode)
        self.trend_expert.toggle_dnas_mode(mode)
        self.seasonal_expert.toggle_dnas_mode(mode)
        
    def get_dnas_status(self):
        return {
            "decomp": self.decomp.get_dnas_status(),
            "trend_expert": self.trend_expert.get_dnas_status(),
            "seasonal_expert": self.seasonal_expert.get_dnas_status()
        }
    
    def get_arch_parameters(self):
        yield from self.decomp.get_arch_parameters()
        yield from self.trend_expert.get_arch_parameters()
        yield from self.seasonal_expert.get_arch_parameters()
