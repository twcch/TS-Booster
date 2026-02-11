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
class MultiScaleDecomp(nn.Module):
    """
    Multi-Scale Decomposition with Global Learnable Weights.

    Concept: Learn a weighted combination of multiple fixed MA decompositions.
    - Maintains MA structure (strong inductive bias)
    - Only 4 learnable scalar parameters (global weights)
    - "Plug-in" design: can replace any SeriesDecomp

    Formula:
        trend = w1*MA7(x) + w2*MA13(x) + w3*MA25(x) + w4*MA49(x)
        seasonal = x - trend
        where w = softmax(learnable_weights)

    Args:
        input_channels: Number of input channels
        kernels: List of kernel sizes (e.g., [7, 13, 25, 49])
        use_learnable: Whether to use learnable or fixed MA (passed to SeriesDecomp)
    """
    def __init__(self, input_channels, kernels=[7, 13, 25, 49], use_learnable=True):
        super(MultiScaleDecomp, self).__init__()

        self.input_channels = input_channels
        self.kernels = kernels

        # Fixed MA decompositions for each kernel size
        self.decomps = nn.ModuleList([
            SeriesDecomp(
                kernel_size=k,
                input_channels=input_channels,
                use_learnable=use_learnable
            )
            for k in kernels
        ])

        # Global learnable weights (one per kernel)
        # Initialize uniformly: [1/K, 1/K, ..., 1/K]
        num_kernels = len(kernels)
        self.weights = nn.Parameter(torch.ones(num_kernels) / num_kernels)

        print(f"[MultiScaleDecomp] Initialized with kernels {kernels}")
        print(f"[MultiScaleDecomp] Global weights shape: {self.weights.shape}")

    def get_weights_status(self):
        """Get current weight distribution (for logging)"""
        with torch.no_grad():
            w = F.softmax(self.weights, dim=0)
        w_list = w.cpu().numpy().tolist()

        status_dict = {}
        for i, k in enumerate(self.kernels):
            status_dict[f"MA_{k}"] = w_list[i]

        return status_dict

    def forward(self, x):
        """
        Args:
            x: [Batch, Seq_Len, Channels]

        Returns:
            trend: [Batch, Seq_Len, Channels]
            seasonal: [Batch, Seq_Len, Channels]
            weights: [num_kernels] (for logging)
        """
        # 1. Compute softmax weights
        w = F.softmax(self.weights, dim=0)  # [num_kernels]

        # 2. Compute all decompositions in parallel
        trends = []
        seasonals = []

        for decomp in self.decomps:
            t, s = decomp(x)
            trends.append(t)
            seasonals.append(s)

        # Stack: [B, S, C, num_kernels]
        trends_stack = torch.stack(trends, dim=-1)
        seasonals_stack = torch.stack(seasonals, dim=-1)

        # 3. Weighted combination
        # Broadcast weights: [1, 1, 1, num_kernels]
        w_broadcast = w.view(1, 1, 1, -1)

        final_trend = (trends_stack * w_broadcast).sum(dim=-1)      # [B, S, C]
        final_seasonal = (seasonals_stack * w_broadcast).sum(dim=-1)  # [B, S, C]

        return final_trend, final_seasonal, w


class FFTGuidedGaussianDecomp(nn.Module):
    """
    FFT-Guided Gaussian Decomposition with Learnable Smoothing.

    Concept: Use continuous Gaussian smoothing with learnable σ (scale parameter).
    - FFT-guided initialization: Estimate dominant period from frequency domain
    - Learnable σ: Continuous optimization (more gradient-friendly than discrete kernels)
    - Gaussian kernel: Smooth trend extraction with mathematical properties

    Formula:
        trend = GaussianSmooth(x, σ)
        seasonal = x - trend
        where σ ∈ ℝ+ is learnable

    Args:
        input_channels: Number of input channels
        seq_len: Sequence length (for FFT analysis)
        init_sigma: Initial σ value (if None, will be estimated from FFT)
        min_sigma: Minimum allowed σ (for stability)
        max_sigma: Maximum allowed σ (to prevent over-smoothing)
    """
    def __init__(self, input_channels, seq_len, init_sigma=None, min_sigma=1.0, max_sigma=50.0):
        super(FFTGuidedGaussianDecomp, self).__init__()

        self.input_channels = input_channels
        self.seq_len = seq_len
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # Learnable log(σ) for numerical stability
        # Will initialize from FFT on first forward pass if init_sigma is None
        if init_sigma is not None:
            init_log_sigma = torch.log(torch.tensor(init_sigma).clamp(min_sigma, max_sigma))
        else:
            # Default initialization: σ = seq_len / 6 (typical heuristic)
            init_log_sigma = torch.log(torch.tensor(seq_len / 6.0).clamp(min_sigma, max_sigma))

        self.log_sigma = nn.Parameter(init_log_sigma)
        self._fft_initialized = (init_sigma is not None)

        print(f"[FFTGuidedGaussianDecomp] Initialized with seq_len={seq_len}")
        print(f"[FFTGuidedGaussianDecomp] Initial σ = {torch.exp(self.log_sigma).item():.2f}")

    def estimate_period_from_fft(self, x):
        """
        Estimate dominant period from FFT analysis.

        Args:
            x: [Batch, Seq_Len, Channels]

        Returns:
            estimated_period: Scalar tensor
        """
        # Average across batch and channels for robust estimate
        x_mean = x.mean(dim=(0, 2))  # [Seq_Len]

        # FFT to find dominant frequency
        fft = torch.fft.rfft(x_mean)
        power_spectrum = torch.abs(fft) ** 2

        # Ignore DC component (index 0) and find peak
        peak_idx = torch.argmax(power_spectrum[1:]) + 1

        # Convert frequency index to period
        # period = seq_len / frequency_index
        estimated_period = self.seq_len / peak_idx.float()

        return estimated_period

    def initialize_from_fft(self, x):
        """
        Initialize σ from FFT analysis (called once on first forward).

        Heuristic: σ ≈ period / 4 (smooth over ~quarter of dominant cycle)
        """
        if self._fft_initialized:
            return

        with torch.no_grad():
            period = self.estimate_period_from_fft(x)
            # Initialize σ to smooth over ~quarter of the dominant period
            init_sigma = (period / 4.0).clamp(self.min_sigma, self.max_sigma)
            self.log_sigma.data = torch.log(init_sigma)

            print(f"[FFTGuidedGaussianDecomp] FFT Initialization:")
            print(f"  Estimated period: {period.item():.2f}")
            print(f"  Initial σ: {init_sigma.item():.2f}")

        self._fft_initialized = True

    def gaussian_kernel_1d(self, sigma, kernel_size):
        """
        Generate 1D Gaussian kernel.

        Args:
            sigma: Standard deviation
            kernel_size: Kernel size (should be odd)

        Returns:
            kernel: [kernel_size] normalized Gaussian kernel
        """
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create coordinate grid centered at 0
        x = torch.arange(kernel_size, dtype=sigma.dtype, device=sigma.device)
        x = x - kernel_size // 2

        # Gaussian formula: exp(-x^2 / (2*σ^2))
        kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))

        # Normalize to sum to 1
        kernel = kernel / kernel.sum()

        return kernel

    def gaussian_smooth(self, x, sigma):
        """
        Apply Gaussian smoothing to extract trend.

        Args:
            x: [Batch, Seq_Len, Channels]
            sigma: Scalar tensor (smoothing scale)

        Returns:
            trend: [Batch, Seq_Len, Channels]
        """
        B, S, C = x.shape

        # Dynamic kernel size: 6σ rule (covers 99.7% of Gaussian)
        kernel_size = int(6 * sigma.item()) + 1
        kernel_size = min(kernel_size, S)  # Cap at sequence length

        # Generate Gaussian kernel
        kernel = self.gaussian_kernel_1d(sigma, kernel_size)  # [K]

        # Reshape for conv1d: [out_channels, in_channels, kernel_size]
        # Apply same kernel to all channels independently
        kernel = kernel.view(1, 1, -1).repeat(C, 1, 1)  # [C, 1, K]

        # Reshape input for conv1d: [B, C, S]
        x_perm = x.permute(0, 2, 1)

        # Apply depthwise convolution (groups=C)
        # Padding: 'same' to preserve length
        padding = kernel_size // 2
        trend = F.conv1d(x_perm, kernel, padding=padding, groups=C)

        # Reshape back: [B, S, C]
        trend = trend.permute(0, 2, 1)

        return trend

    def get_sigma_status(self):
        """Get current σ value (for logging)"""
        with torch.no_grad():
            sigma = torch.exp(self.log_sigma).clamp(self.min_sigma, self.max_sigma)
        return {"sigma": sigma.item()}

    def forward(self, x):
        """
        Args:
            x: [Batch, Seq_Len, Channels]

        Returns:
            trend: [Batch, Seq_Len, Channels]
            seasonal: [Batch, Seq_Len, Channels]
            sigma: Scalar tensor (for logging)
        """
        # Initialize from FFT on first forward pass
        self.initialize_from_fft(x)

        # Get current σ (constrained to valid range)
        sigma = torch.exp(self.log_sigma).clamp(self.min_sigma, self.max_sigma)

        # Apply Gaussian smoothing to extract trend
        trend = self.gaussian_smooth(x, sigma)

        # Seasonal = residual
        seasonal = x - trend

        return trend, seasonal, sigma


class MultiScaleExpert(nn.Module):
    """
    Inception-style Multi-Scale CNN Expert.
    Multiple parallel conv branches with different kernel sizes, features concatenated.
    Much lighter than DNAS_CNNExpert (single model vs N parallel full CNNs).
    """
    def __init__(self, seq_len, pred_len, input_channels, hidden_dim=32, kernels=[3, 5, 7], dropout_p=0.3):
        super().__init__()
        self.pred_len = pred_len
        self.input_channels = input_channels
        branch_dim = hidden_dim // len(kernels)

        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(nn.Sequential(
                nn.Conv1d(input_channels, branch_dim, kernel_size=k, padding=(k-1)//2),
                nn.BatchNorm1d(branch_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_p),
            ))

        concat_dim = branch_dim * len(kernels)
        self.fc = nn.Linear(seq_len * concat_dim, pred_len * input_channels)

    def forward(self, x):
        # x: [B, S, C]
        B, S, C = x.shape
        x_perm = x.permute(0, 2, 1)  # [B, C, S]
        feats = [branch(x_perm) for branch in self.branches]
        feat = torch.cat(feats, dim=1)  # [B, concat_dim, S]
        feat = feat.reshape(B, -1)
        out = self.fc(feat)  # [B, pred_len * C]
        out = out.reshape(B, self.pred_len, C)  # [B, pred_len, C]
        return out


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
        self.pred_len = pred_len
        self.input_channels = input_channels
        self.fc = nn.Linear(self.flatten_dim, pred_len * input_channels)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, _, C = x.shape
        x_flat = x.reshape(B, -1)
        out_flat = self.fc(x_flat)  # [Batch, Pred_Len * Channels]
        out = out_flat.reshape(B, self.pred_len, C)  # [Batch, Pred_Len, Channels]

        # Handle dimension: squeeze to 2D for compatibility
        # (assumes univariate output, take first channel if multi-channel input)
        if out.dim() == 3:
            if out.size(-1) == 1:
                out = out.squeeze(-1)  # [B, P, 1] -> [B, P]
            else:
                out = out[..., 0]  # [B, P, C] -> [B, P], take first channel only

        return out, out, None, None, None
