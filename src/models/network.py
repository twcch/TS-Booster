import torch
import torch.nn as nn

from src.models.layers import CNNExpert, SeriesDecomp, LSTMBackbone, GRUBackbone, TransformerBackbone, DNAS_SeriesDecomp, DNAS_CNNExpert


class EnhancedDLinear(nn.Module):
    """
    Enhanced DLinear with DNAS (Differentiable Neural Architecture Search).
    
    主要變更:
    1. 引入 DNAS_SeriesDecomp: 自動搜尋最佳分解視窗。
    2. 引入 DNAS_CNNExpert: 自動搜尋最佳卷積核大小。
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        input_channels,
        use_seasonal_cnn=True,
        use_trend_cnn=False,
        use_decomp=True,
        use_learnable=True,
        # CNN Expert params
        trend_hidden_dim=32,
        seasonal_hidden_dim=32,
        trend_dropout=0.35,
        seasonal_dropout=0.35,
        # DNAS Candidate Pools (取代原本單一 Kernel Size)
        trend_dnas_kernels=[3, 5, 7, 9],            # Trend CNN 候選池
        seasonal_dnas_kernels=[3, 5, 7, 9],         # Seasonal CNN 候選池
        decomp_dnas_kernels=[7, 13, 25, 49],        # 分解視窗候選池
        # Gating mechanism params
        trend_base_init=0.0,
        seasonal_base_init=0.0,
        delta_net_hidden_dim=8,
        # Backbone params
        backbone_type='linear',
        backbone_hidden_dim=64,
        backbone_num_layers=2,
        # Legacy params (為了兼容舊代碼，雖然 DNAS 版本可能不用，但保留以免報錯)
        trendCNNExpert_KernelSize=None,
        seasonalCNNExpert_KernelSize=None,
        seriesDecomposition_KernelSize=None,
    ):
        super().__init__()

        # 1. 保存設定參數
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_decomp = use_decomp
        self.input_channels = input_channels
        self.use_seasonal_cnn = use_seasonal_cnn
        self.use_trend_cnn = use_trend_cnn
        self.use_learnable = use_learnable
        self.backbone_type = backbone_type.lower()
        self.backbone_hidden_dim = backbone_hidden_dim
        self.backbone_num_layers = backbone_num_layers
        self.backbone_num_layers = backbone_num_layers
        self.trend_hidden_dim = trend_hidden_dim
        self.seasonal_hidden_dim = seasonal_hidden_dim
        self.delta_net_hidden_dim = delta_net_hidden_dim
        
        # DNAS Temperature
        self.dnas_tau = 1.0
        
        # Gating Status
        self.gating_fixed = False # If True, force gates to 0.5

        # 2. Series Decomposition (DNAS 版本)
        if self.use_decomp:
            # 使用 DNAS 模組自動尋找最佳分解視窗
            self.decomp = DNAS_SeriesDecomp(
                input_channels=input_channels,
                candidate_kernels=decomp_dnas_kernels,
                use_learnable=use_learnable
            )
        else:
            self.decomp = None

        # 3. Backbone Selection (Linear, LSTM, GRU, or Transformer)
        # (這部分邏輯不變，負責處理 Trend 和 Seasonal 的主要預測)
        if self.backbone_type == 'linear':
            self.linear_trend = nn.Linear(seq_len * input_channels, pred_len)
            self.linear_seasonal = nn.Linear(seq_len * input_channels, pred_len)
            self.backbone_trend = None
            self.backbone_seasonal = None
        elif self.backbone_type == 'lstm':
            self.backbone_trend = LSTMBackbone(seq_len, input_channels, pred_len, backbone_hidden_dim, backbone_num_layers)
            self.backbone_seasonal = LSTMBackbone(seq_len, input_channels, pred_len, backbone_hidden_dim, backbone_num_layers)
            self.linear_trend = None
            self.linear_seasonal = None
        elif self.backbone_type == 'gru':
            self.backbone_trend = GRUBackbone(seq_len, input_channels, pred_len, backbone_hidden_dim, backbone_num_layers)
            self.backbone_seasonal = GRUBackbone(seq_len, input_channels, pred_len, backbone_hidden_dim, backbone_num_layers)
            self.linear_trend = None
            self.linear_seasonal = None
        elif self.backbone_type == 'transformer':
            self.backbone_trend = TransformerBackbone(seq_len, input_channels, pred_len, backbone_hidden_dim, nhead=4, num_layers=backbone_num_layers)
            self.backbone_seasonal = TransformerBackbone(seq_len, input_channels, pred_len, backbone_hidden_dim, nhead=4, num_layers=backbone_num_layers)
            self.linear_trend = None
            self.linear_seasonal = None
        else:
            raise ValueError(f"Unknown backbone_type: {self.backbone_type}")

        # 4. CNN Booster & Hybrid Gating (DNAS 版本)

        if self.use_trend_cnn:
            # 改用 DNAS_CNNExpert
            self.cnn_trend = DNAS_CNNExpert(
                seq_len=seq_len, 
                pred_len=pred_len, 
                input_channels=input_channels, 
                hidden_dim=trend_hidden_dim, 
                candidate_kernels=trend_dnas_kernels,  # 使用候選池
                dropout_p=trend_dropout
            )
            self.trend_base = nn.Parameter(torch.tensor(trend_base_init))
            self.trend_delta_net = nn.Sequential(
                nn.Linear(seq_len * input_channels, delta_net_hidden_dim),
                nn.ReLU(),
                nn.Linear(delta_net_hidden_dim, 1),
                nn.Tanh(),
            )
        else:
            self.cnn_trend = None
            self.trend_base = None
            self.trend_delta_net = None

        if self.use_seasonal_cnn:
            # 改用 DNAS_CNNExpert
            self.cnn_seasonal = DNAS_CNNExpert(
                seq_len=seq_len,
                pred_len=pred_len,
                input_channels=input_channels,
                hidden_dim=seasonal_hidden_dim,
                candidate_kernels=seasonal_dnas_kernels, # 使用候選池
                dropout_p=seasonal_dropout
            )

            self.seasonal_base = nn.Parameter(torch.tensor(seasonal_base_init))

            self.seasonal_delta_net = nn.Sequential(
                nn.Linear(seq_len * input_channels, delta_net_hidden_dim),
                nn.ReLU(),
                nn.Linear(delta_net_hidden_dim, 1),
                nn.Tanh(),
            )
        else:
            self.cnn_seasonal = None
            self.seasonal_base = None
            self.seasonal_delta_net = None

            self.seasonal_delta_net = None

    def set_dnas_tau(self, tau):
        """
        設定 DNAS Gumbel-Softmax 的溫度係數
        """
        self.dnas_tau = tau
        if self.use_decomp and self.decomp is not None:
            self.decomp.set_tau(tau)
        if self.use_trend_cnn and self.cnn_trend is not None:
            self.cnn_trend.set_tau(tau)
        if self.use_seasonal_cnn and self.cnn_seasonal is not None:
            self.cnn_seasonal.set_tau(tau)

    def set_gating_status(self, fixed=True):
        """
        設定是否固定門控權重 (Gating Warmup)
        fixed=True: 強制將 Gate 設為 0.5，不進行梯度更新
        fixed=False: 開放模型學習 Gate
        """
        self.gating_fixed = fixed

    def get_arch_parameters(self):
        """
        取得所有架構參數 (Alpha)
        """
        arch_params = []
        if self.use_decomp and self.decomp is not None:
            arch_params.extend(list(self.decomp.get_arch_parameters()))
        if self.use_trend_cnn and self.cnn_trend is not None:
            arch_params.extend(list(self.cnn_trend.get_arch_parameters()))
        if self.use_seasonal_cnn and self.cnn_seasonal is not None:
            arch_params.extend(list(self.cnn_seasonal.get_arch_parameters()))
        return arch_params
    
    def toggle_dnas_mode(self, mode="warmup"):
        """
        Switch DNAS mode for all DNAS components.
        mode="warmup": Freeze architecture parameters.
        mode="search": Unfreeze architecture parameters.
        """
        if self.use_decomp and self.decomp is not None:
            self.decomp.toggle_dnas_mode(mode)
        if self.use_trend_cnn and self.cnn_trend is not None:
            self.cnn_trend.toggle_dnas_mode(mode)
        if self.use_seasonal_cnn and self.cnn_seasonal is not None:
            self.cnn_seasonal.toggle_dnas_mode(mode)

    def get_dnas_status(self):
        """
        Get diagnostics information about DNAS weights.
        """
        status = {}
        if self.use_decomp and self.decomp is not None:
            status["decomp"] = self.decomp.get_dnas_status()
        if self.use_trend_cnn and self.cnn_trend is not None:
            status["trend_cnn"] = self.cnn_trend.get_dnas_status()
        if self.use_seasonal_cnn and self.cnn_seasonal is not None:
            status["seasonal_cnn"] = self.cnn_seasonal.get_dnas_status()
        return status
            
    def get_model_parameters(self):
        """
        取得所有模型權重 (排除架構參數)
        通常用 `net.parameters()` 減去 `arch_parameters` 也可以，但這裡提供明確接口
        """
        # 簡單實作：回傳全部，讓 Optimizer 透過 param_groups 區分
        # 或者使用 yield 過濾
        for name, param in self.named_parameters():
             # 假設架構參數都有特定命名模式 (alpha_)
             if 'alpha_' not in name:
                 yield param

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels]

        # === 1. Series Decomposition (DNAS) ===
        decomp_weights = None
        if self.use_decomp and self.decomp is not None:
            # 這裡 decomp 會自動做 DNAS 加權總和，回傳最好的 Trend 與 Seasonal 組合
            # 更新: 回傳 (trend, seasonal, weights)
            trend_part, seasonal_part, decomp_weights = self.decomp(x)
        else:
            seasonal_part = torch.zeros_like(x)
            trend_part = x

        # 準備 Flatten 特徵
        B, S, C = x.shape
        trend_flat = trend_part.reshape(B, -1)
        seasonal_flat = seasonal_part.reshape(B, -1)

        # === 2. Backbone Prediction (基底預測) ===
        if self.backbone_type == 'linear':
            trend_out_linear = self.linear_trend(trend_flat)
            seasonal_out_linear = self.linear_seasonal(seasonal_flat)
        else:
            trend_out_linear = self.backbone_trend(trend_part)
            seasonal_out_linear = self.backbone_seasonal(seasonal_part)

        base_output_norm = trend_out_linear + seasonal_out_linear

        # === 3. CNN Expert Correction (DNAS + Gating) ===
        cnn_correction = 0.0

        # 用於 Log 監控 Gating 權重
        t_weight_val = torch.tensor(0.0, device=x.device)
        s_weight_val = torch.tensor(0.0, device=x.device)
        
        # DNAS 權重監控
        trend_cnn_weights = None
        seasonal_cnn_weights = None

        # Part A: Trend CNN
        if self.use_trend_cnn and self.cnn_trend is not None:
            # DNAS Expert 自動加權多尺度特徵
            # 更新: 回傳 (output, weights)
            trend_out_cnn, trend_cnn_weights = self.cnn_trend(trend_part)
            
            if self.gating_fixed:
                t_gate = torch.tensor(0.5, device=x.device)
            else:
                # Gating 機制
                delta_t = self.trend_delta_net(trend_flat)
                raw_gate = torch.sigmoid(self.trend_base + delta_t)
                
                # [Gating Floor Protection]
                # Linear Mapping: [0, 1] -> [0.2, 1.0]
                t_gate = 0.2 + 0.8 * raw_gate

            cnn_correction = cnn_correction + (t_gate * trend_out_cnn)
            t_weight_val = t_gate.mean()

        # Part B: Seasonal CNN
        if self.use_seasonal_cnn and self.cnn_seasonal is not None:
            # DNAS Expert 自動加權多尺度特徵
            # 更新: 回傳 (output, weights)
            seasonal_out_cnn, seasonal_cnn_weights = self.cnn_seasonal(seasonal_part)

            if self.gating_fixed:
                s_gate = torch.tensor(0.5, device=x.device)
            else:
                # Gating 機制
                delta_s = self.seasonal_delta_net(seasonal_flat)
                raw_gate = torch.sigmoid(self.seasonal_base + delta_s)
                
                # [Gating Floor Protection]
                # Linear Mapping: [0, 1] -> [0.2, 1.0]
                s_gate = 0.2 + 0.8 * raw_gate

            cnn_correction = cnn_correction + (s_gate * seasonal_out_cnn)
            s_weight_val = s_gate.mean()

        # === 4. Final Output ===
        final_output_norm = base_output_norm + cnn_correction

        # 回傳: [Prediction, Base_Pred, [Trend_Gate, Seasonal_Gate], Trend_Component, Seasonal_Component]
        # 更新 return weights 為: [Trend_Gate, Seasonal_Gate, Max_Trend_DNAS, Max_Seasonal_DNAS]
        
        # 取得 DNAS 最大權重 (Confidence)
        t_dnas_max = torch.tensor(0.0, device=x.device)
        s_dnas_max = torch.tensor(0.0, device=x.device)
        
        if trend_cnn_weights is not None:
            t_dnas_max = trend_cnn_weights.max()
        if seasonal_cnn_weights is not None:
            s_dnas_max = seasonal_cnn_weights.max()
        
        weights_to_return = torch.stack([t_weight_val, s_weight_val, t_dnas_max, s_dnas_max])
        
        # 暫時 Hack: 將 DNAS 權重存成 member variable 供外部讀取 (僅供 debug/logging)
        self.last_decomp_weights = decomp_weights
        self.last_trend_cnn_weights = trend_cnn_weights
        self.last_seasonal_cnn_weights = seasonal_cnn_weights

        return final_output_norm, base_output_norm, weights_to_return, trend_part, seasonal_part
