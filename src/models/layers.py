import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableMovingAvg(nn.Module):
    def __init__(self, kernel_size, input_channels):
        super(LearnableMovingAvg, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=input_channels,
            bias=False,
            padding_mode='replicate'

        )
        nn.init.constant_(self.conv.weight, 1.0 / kernel_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class FixedMovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super(FixedMovingAvg, self).__init__()
        # 使用 AvgPool1d，它本身就沒有可學習參數
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        # 假設輸入 x 為 (Batch, Time, Channel)
        x = x.permute(0, 2, 1)
        x = self.avg_pool(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size, input_channels, use_learnable=True):
        super(SeriesDecomp, self).__init__()

        if use_learnable:
            print("Learnable Moving Average Decomposition Enabled")
            self.moving_avg = LearnableMovingAvg(kernel_size, input_channels)
        else:
            print("Fixed Moving Average Decomposition Enabled")
            self.moving_avg = FixedMovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal

class IdentityDecomp(nn.Module):
    """
    Identity Decomposition Path.
    Trend = Input
    Seasonal = 0
    """
    def __init__(self):
        super(IdentityDecomp, self).__init__()

    def forward(self, x):
        return x, torch.zeros_like(x)

class CNNExpert(nn.Module):
    def __init__(self, seq_len, pred_len, input_channels, hidden_dim=64, cnnExpert_KernelSize=5, dropout_p=0.35):
        super().__init__()

        # Params needed for tuning
        k_size = cnnExpert_KernelSize  # <--- Kernel Size (3, 5, 7...)
        # dropout_p 现在是参数传入

        # 自動計算 Padding (保證輸入輸出長度一致：30 -> 30)
        # 公式：(kernel_size - 1) // 2
        # 例如：k=3 -> p=1; k=5 -> p=2; k=7 -> p=3
        padding = (k_size - 1) // 2

        self.net = nn.Sequential(
            # 第一層
            nn.Conv1d(input_channels, hidden_dim, kernel_size=k_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),

            # 第二層 (為了簡化，我們讓第二層也跟著變，或者你可以固定第二層為 k=3)
            # 這裡示範兩層都改大，讓感受野最大化
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=k_size, padding=padding),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p)
        )

        # 只要 Padding 設定正確，長度 seq_len (30) 就不會變
        self.flatten_dim = seq_len * (hidden_dim * 2)
        self.fc = nn.Linear(self.flatten_dim, pred_len)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels] -> [Batch, Channels, Seq_Len]
        x = x.permute(0, 2, 1)
        feat = self.net(x)
        # Flatten
        feat = feat.reshape(feat.size(0), -1)
        out = self.fc(feat)
        return out


class LinearExpert(nn.Module):
    """
    Linear Expert (Identity-like path for Dimension Change).
    Flatten -> Linear Projection.
    Acts as the 'Zero-Operation' or 'Simple' path in DNAS.
    """
    def __init__(self, seq_len, pred_len, input_channels):
        super(LinearExpert, self).__init__()
        self.flatten_dim = seq_len * input_channels
        self.fc = nn.Linear(self.flatten_dim, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, S, C = x.shape
        x_flat = x.reshape(B, -1)
        return self.fc(x_flat)



class LSTMBackbone(nn.Module):
    """LSTM Backbone for sequence modeling"""
    def __init__(self, seq_len, input_channels, pred_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Output layer
        self.fc = nn.Linear(seq_len * hidden_dim, pred_len)
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]
        # LSTM expects [Batch, Seq_Len, Features]
        lstm_out, _ = self.lstm(x)  # [Batch, Seq_Len, Hidden_Dim]
        
        # Flatten
        feat = lstm_out.reshape(lstm_out.size(0), -1)  # [Batch, Seq_Len * Hidden_Dim]
        out = self.fc(feat)  # [Batch, Pred_Len]
        
        # Standardize return for trainer compatibility:
        # (final_out, base_out, gate_weights, trend_part, seasonal_part)
        return out, out, None, None, None


class GRUBackbone(nn.Module):
    """GRU Backbone for sequence modeling"""
    def __init__(self, seq_len, input_channels, pred_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Output layer
        self.fc = nn.Linear(seq_len * hidden_dim, pred_len)
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]
        # GRU expects [Batch, Seq_Len, Features]
        gru_out, _ = self.gru(x)  # [Batch, Seq_Len, Hidden_Dim]
        
        # Flatten
        feat = gru_out.reshape(gru_out.size(0), -1)  # [Batch, Seq_Len * Hidden_Dim]
        out = self.fc(feat)  # [Batch, Pred_Len]
        
        return out, out, None, None, None


class TransformerBackbone(nn.Module):
    """Transformer Backbone for sequence modeling"""
    def __init__(self, seq_len, input_channels, pred_len, d_model=64, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(seq_len * d_model, pred_len)
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]
        
        # Project input to d_model dimension
        x = self.input_projection(x)  # [Batch, Seq_Len, d_model]
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)  # [Batch, Seq_Len, d_model]
        
        # Flatten
        feat = transformer_out.reshape(transformer_out.size(0), -1)  # [Batch, Seq_Len * d_model]
        out = self.fc(feat)  # [Batch, Pred_Len]
        
        return out, out, None, None, None


class DNAS_SeriesDecomp(nn.Module):
    """
    DNAS-enabled Series Decomposition with Gumbel-Softmax & Optional Identity Path.
    
    Args:
        candidate_kernels: List of kernel sizes. Can include 'identity' string to add Identity path.
                          Example: [7, 13, 25, 49, 'identity'] or [7, 13, 25, 49]
    """
    def __init__(self, input_channels, candidate_kernels=[7, 13, 25, 49, 'identity'], use_learnable=True):
        super(DNAS_SeriesDecomp, self).__init__()
        
        self.input_channels = input_channels
        self.tau = 1.0  # Temperature for Gumbel-Softmax
        
        # 分離數值 kernel 和 identity 標記
        self.numeric_kernels = [k for k in candidate_kernels if isinstance(k, int)]
        self.has_identity = 'identity' in candidate_kernels
        
        # 建立候選分解操作池
        self.decomps = nn.ModuleList([
            SeriesDecomp(
                kernel_size=k, 
                input_channels=input_channels, 
                use_learnable=use_learnable
            ) 
            for k in self.numeric_kernels
        ])
        
        # 如果需要,添加 Identity Decomp
        if self.has_identity:
            self.decomps.append(IdentityDecomp())
        
        # 架構參數
        num_candidates = len(self.numeric_kernels) + (1 if self.has_identity else 0)
        self.alpha_decomp = nn.Parameter(torch.randn(num_candidates) * 1e-3)

    def set_tau(self, tau):
        self.tau = tau

    def get_arch_parameters(self):
        yield self.alpha_decomp

    def get_model_parameters(self):
        return self.decomps.parameters()

    def toggle_dnas_mode(self, mode="warmup"):
        """
        Switch DNAS mode.
        mode="warmup": Freeze architecture parameters (alpha).
        mode="search": Unfreeze architecture parameters.
        """
        if mode == "warmup":
            self.alpha_decomp.requires_grad = False
        elif mode == "search":
            self.alpha_decomp.requires_grad = True
            
    def get_dnas_status(self):
        """
        Get current architecture probability distribution.
        """
        with torch.no_grad():
            if self.training:
                # Use current tau for a realistic view of what gumbel is seeing, 
                # or just softmax for 'true' preference? 
                # Usually Softmax(alpha) shows the underlying preference.
                probs = F.softmax(self.alpha_decomp, dim=0)
            else:
                probs = F.softmax(self.alpha_decomp, dim=0)
        probs_list = probs.cpu().numpy().tolist()
        
        # Construct Dictionary with Labels
        status_dict = {}
        for i, k in enumerate(self.numeric_kernels):
            status_dict[f"MA_{k}"] = probs_list[i]
        
        # 只有當有 Identity 時才添加
        if self.has_identity:
            status_dict["Identity"] = probs_list[-1]
        
        return status_dict

    def forward(self, x):
        # 1. 計算權重 (Gumbel-Softmax)
        if self.training:
            # Hard=False: 可微，但接近 One-hot
            weights = F.gumbel_softmax(self.alpha_decomp, tau=self.tau, hard=False, dim=0)
        else:
            # 推論時直接用 Softmax
            weights = F.softmax(self.alpha_decomp, dim=0)
        
        trends = []
        seasonals = []
        
        # 2. 並行計算
        for decomp in self.decomps:
            t, s = decomp(x)
            trends.append(t)
            seasonals.append(s)
            
        trends_stack = torch.stack(trends, dim=-1)     # [B, S, C, K]
        seasonals_stack = torch.stack(seasonals, dim=-1) # [B, S, C, K]
        
        # 3. 加權總和
        # weights shape: [K] -> broadcast to [1, 1, 1, K]
        w = weights.view(1, 1, 1, -1)
        final_trend = (trends_stack * w).sum(dim=-1)
        final_seasonal = (seasonals_stack * w).sum(dim=-1)
        
        return final_trend, final_seasonal, weights


class DNAS_CNNExpert(nn.Module):
    """
    DNAS-enabled CNN Expert with Gumbel-Softmax, Identity Path & Path Dropout.
    """
    def __init__(self, seq_len, pred_len, input_channels, hidden_dim=64, 
                 candidate_kernels=[3, 5, 7, 9], dropout_p=0.35):
        super(DNAS_CNNExpert, self).__init__()
        
        self.candidate_kernels = candidate_kernels
        self.tau = 1.0
        
        experts_list = []
        # 1. 一般 CNN Experts
        for k in candidate_kernels:
            experts_list.append(
                CNNExpert(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    input_channels=input_channels,
                    hidden_dim=hidden_dim,
                    cnnExpert_KernelSize=k,
                    dropout_p=dropout_p
                )
            )
        
        # 2. Identity Path (Linear Expert)
        # 模擬 "Skip Connection" 或 "不處理" (但需轉維度)
        experts_list.append(
            LinearExpert(seq_len, pred_len, input_channels)
        )
        
        self.experts = nn.ModuleList(experts_list)
        
        # 架構參數 (多了 Identity 一個選項)
        num_candidates = len(candidate_kernels) + 1
        self.alpha_cnn = nn.Parameter(torch.randn(num_candidates) * 1e-3)

    def set_tau(self, tau):
        self.tau = tau

    def get_arch_parameters(self):
        yield self.alpha_cnn

    def get_model_parameters(self):
        return self.experts.parameters()
        
    def toggle_dnas_mode(self, mode="warmup"):
        """
        Switch DNAS mode.
        mode="warmup": Freeze architecture parameters (alpha).
        mode="search": Unfreeze architecture parameters.
        """
        if mode == "warmup":
            self.alpha_cnn.requires_grad = False
        elif mode == "search":
            self.alpha_cnn.requires_grad = True
            
    def get_dnas_status(self):
        """
        Get current architecture probability distribution.
        """
        with torch.no_grad():
            probs = F.softmax(self.alpha_cnn, dim=0)
        probs_list = probs.cpu().numpy().tolist()
        
        # Construct Dictionary with Labels
        status_dict = {}
        for i, k in enumerate(self.candidate_kernels):
            status_dict[f"K_{k}"] = probs_list[i]
        status_dict["Identity"] = probs_list[-1]
        
        return status_dict

    def forward(self, x):
        # 1. 計算權重 Gumbel-Softmax
        if self.training:
            weights = F.gumbel_softmax(self.alpha_cnn, tau=self.tau, hard=False, dim=0)
        else:
            weights = F.softmax(self.alpha_cnn, dim=0)
            
        outputs = []
        
        # 2. 並行計算
        for expert in self.experts:
            out = expert(x) 
            outputs.append(out)
            
        outputs_stack = torch.stack(outputs, dim=-1) # [B, Pred, K]
        
        # 3. 加權總和
        w = weights.view(1, 1, -1)
        final_output = (outputs_stack * w).sum(dim=-1)
        
        return final_output, weights
class LinearBackbone(nn.Module):
    """
    Simple Linear Backbone.
    Flatten -> Linear -> Output.
    """
    def __init__(self, seq_len, input_channels, pred_len, hidden_dim=64, num_layers=1):
        super().__init__()
        # hidden_dim & num_layers ignored for pure linear, or we can use them for MLP?
        # User request typically implies "Linear" as in DLinear (single layer).
        # But factory passes hidden_dim. Let's make it a simple single layer linear for "Linear".
        self.flatten_dim = seq_len * input_channels
        self.fc = nn.Linear(self.flatten_dim, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        out = self.fc(x_flat) # [Batch, Pred_Len]
        return out, out, None, None, None
