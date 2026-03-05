"""
SOTA Deep Learning Baselines for TS-Booster Paper.

Standalone implementations of 4 recent top-venue models:
- DLinear (AAAI 2023): Moving avg decomposition + dual Linear
- PatchTST (ICLR 2023): Patch-based Transformer encoder
- iTransformer (ICLR 2024): Inverted attention across variates
- TimeMixer (ICLR 2024): Multi-scale MLP mixing + decomposition

All models: forward([B, seq_len, n_features]) -> (pred[B, pred_len], pred, None, None, None)
Compatible with train_v11() 5-tuple format.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Shared Components
# ──────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022).

    Used in PatchTST to handle distribution shift:
    - normalize: subtract mean, divide by std (per-instance, per-channel)
    - denormalize: reverse the operation on model output
    """

    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(n_features))
            self.affine_bias = nn.Parameter(torch.zeros(n_features))

    def normalize(self, x):
        # x: [B, T, C]
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x):
        # x: [B, T, C] or [B, pred_len, C]
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        x = x * self._std + self._mean
        return x


class MovingAvgDecomp(nn.Module):
    """Simple moving average decomposition (shared by DLinear & TimeMixer)."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        # x: [B, T, C]
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


# ──────────────────────────────────────────────────────────────
# DLinear (AAAI 2023)
# ──────────────────────────────────────────────────────────────

class DLinear(nn.Module):
    """
    DLinear: Decomposition + dual Linear layers.
    Ref: Are Transformers Effective for Time Series Forecasting? (Zeng et al., AAAI 2023)

    Channel-independent: each feature gets its own Linear weights.
    ~180 params for seq_len=30, pred_len=3, n_features=2.
    """

    def __init__(self, seq_len=30, pred_len=3, n_features=2, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.decomp = MovingAvgDecomp(kernel_size)
        # Channel-independent linear layers
        self.linear_seasonal = nn.ModuleList(
            [nn.Linear(seq_len, pred_len) for _ in range(n_features)]
        )
        self.linear_trend = nn.ModuleList(
            [nn.Linear(seq_len, pred_len) for _ in range(n_features)]
        )

    def forward(self, x):
        # x: [B, T, C]
        seasonal, trend = self.decomp(x)
        # Channel-independent projection
        seasonal_out = torch.stack(
            [self.linear_seasonal[i](seasonal[:, :, i]) for i in range(self.n_features)],
            dim=-1,
        )  # [B, pred_len, C]
        trend_out = torch.stack(
            [self.linear_trend[i](trend[:, :, i]) for i in range(self.n_features)],
            dim=-1,
        )  # [B, pred_len, C]
        out = seasonal_out + trend_out  # [B, pred_len, C]
        pred = out[:, :, 0]  # volatility channel only → [B, pred_len]
        return pred, pred, None, None, None


# ──────────────────────────────────────────────────────────────
# PatchTST (ICLR 2023)
# ──────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    PatchTST: Patching + channel-independent Transformer encoder.
    Ref: A Time Series is Worth 64 Words (Nie et al., ICLR 2023)

    Channel-independent: each feature processed separately through shared Transformer.
    ~200K params with d_model=128, n_heads=8, n_layers=3.
    """

    def __init__(
        self,
        seq_len=30,
        pred_len=3,
        n_features=2,
        patch_len=5,
        stride=3,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding: linear projection from patch_len -> d_model
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Learnable position encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Flatten head: n_patches * d_model -> pred_len
        self.head = nn.Linear(self.n_patches * d_model, pred_len)

        # RevIN for handling distribution shift
        self.revin = RevIN(n_features)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        # RevIN normalize
        x = self.revin.normalize(x)

        # Channel-independent: process each channel separately
        channel_preds = []
        for c in range(C):
            xc = x[:, :, c]  # [B, T]

            # Create patches: [B, n_patches, patch_len]
            patches = xc.unfold(dimension=1, size=self.patch_len, step=self.stride)

            # Patch embedding + positional encoding
            patches = self.patch_embed(patches) + self.pos_embed  # [B, n_patches, d_model]

            # Transformer encoder
            encoded = self.encoder(patches)  # [B, n_patches, d_model]

            # Flatten and predict
            flat = encoded.reshape(B, -1)  # [B, n_patches * d_model]
            pred = self.head(flat)  # [B, pred_len]
            channel_preds.append(pred)

        # Stack channels → [B, pred_len, C], denormalize, take channel 0
        out = torch.stack(channel_preds, dim=-1)  # [B, pred_len, C]
        out = self.revin.denormalize(out)
        pred = out[:, :, 0]  # [B, pred_len]
        return pred, pred, None, None, None


# ──────────────────────────────────────────────────────────────
# iTransformer (ICLR 2024)
# ──────────────────────────────────────────────────────────────

class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer — attention across variates, not time steps.
    Ref: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
         (Liu et al., ICLR 2024)

    Each variate's full time series is embedded as a token.
    With 2 variates, attention captures cross-variate dependencies.
    ~135K params with d_model=128, n_heads=4, n_layers=2.
    """

    def __init__(
        self,
        seq_len=30,
        pred_len=3,
        n_features=2,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_model = d_model

        # Variate embedding: project each variate's time series into d_model
        self.variate_embed = nn.Linear(seq_len, d_model)

        # Learnable variate position encoding
        self.pos_embed = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        # Transformer encoder (attention across variates)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Projection head: d_model -> pred_len (per variate)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [B, T, C] where C = n_features
        B, T, C = x.shape

        # Invert: treat each variate as a token with seq_len features
        # [B, T, C] -> [B, C, T]
        x_inv = x.permute(0, 2, 1)  # [B, C, T]

        # Embed each variate: [B, C, T] -> [B, C, d_model]
        tokens = self.variate_embed(x_inv) + self.pos_embed

        # Attention across variates
        encoded = self.encoder(tokens)  # [B, C, d_model]

        # Project to predictions: [B, C, pred_len]
        out = self.head(encoded)

        # Extract volatility channel (index 0)
        pred = out[:, 0, :]  # [B, pred_len]
        return pred, pred, None, None, None


# ──────────────────────────────────────────────────────────────
# TimeMixer (ICLR 2024)
# ──────────────────────────────────────────────────────────────

class TimeMixer(nn.Module):
    """
    TimeMixer: Multi-scale mixing with decomposition.
    Ref: TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting
         (Wang et al., ICLR 2024)

    Down-samples input to multiple scales, applies MLP mixing at each scale,
    then fuses predictions across scales.
    ~5K params with d_model=32, n_layers=2, down_sampling_layers=3.
    """

    def __init__(
        self,
        seq_len=30,
        pred_len=3,
        n_features=2,
        d_model=32,
        n_layers=2,
        down_sampling_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_model = d_model
        self.n_layers = n_layers
        self.down_sampling_layers = down_sampling_layers

        # Compute scale lengths: [30, 15, 8, 4] for down_sampling_layers=3
        self.scale_lengths = [seq_len]
        for _ in range(down_sampling_layers):
            self.scale_lengths.append(math.ceil(self.scale_lengths[-1] / 2))
        self.n_scales = len(self.scale_lengths)

        # Per-scale embedding
        self.scale_embeds = nn.ModuleList(
            [nn.Linear(n_features, d_model) for _ in range(self.n_scales)]
        )

        # Multi-scale mixing layers
        self.mixing_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mixing_layers.append(
                _TimeMixerBlock(self.scale_lengths, d_model, dropout)
            )

        # Per-scale prediction heads
        self.scale_heads = nn.ModuleList(
            [nn.Linear(sl * d_model, pred_len) for sl in self.scale_lengths]
        )

        # Learnable scale fusion weights
        self.scale_weights = nn.Parameter(torch.zeros(self.n_scales))

    def _downsample(self, x, target_len):
        """Downsample via adaptive average pooling."""
        # x: [B, T, D]
        out = F.adaptive_avg_pool1d(
            x.permute(0, 2, 1), target_len
        ).permute(0, 2, 1)
        return out

    def forward(self, x):
        # x: [B, T, C]
        B = x.shape[0]

        # Create multi-scale representations
        multi_scale = []
        for i, sl in enumerate(self.scale_lengths):
            if i == 0:
                xs = x
            else:
                xs = self._downsample(x, sl)
            xs = self.scale_embeds[i](xs)  # [B, sl, d_model]
            multi_scale.append(xs)

        # Apply mixing layers
        for mix_layer in self.mixing_layers:
            multi_scale = mix_layer(multi_scale)

        # Per-scale predictions + weighted fusion
        w = torch.softmax(self.scale_weights, dim=0)
        pred = torch.zeros(B, self.pred_len, device=x.device)
        for i, (ms, head) in enumerate(zip(multi_scale, self.scale_heads)):
            flat = ms.reshape(B, -1)  # [B, sl * d_model]
            pred = pred + w[i] * head(flat)

        return pred, pred, None, None, None


class _TimeMixerBlock(nn.Module):
    """Single mixing block: cross-scale MLP mixing + temporal MLP."""

    def __init__(self, scale_lengths, d_model, dropout=0.1):
        super().__init__()
        self.n_scales = len(scale_lengths)
        self.d_model = d_model

        # Temporal mixing (per-scale MLP on time dimension)
        self.temporal_mlps = nn.ModuleList()
        for sl in scale_lengths:
            self.temporal_mlps.append(
                nn.Sequential(
                    nn.Linear(sl, sl),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(sl, sl),
                )
            )

        # Cross-scale mixing: blend adjacent scales
        self.cross_scale_ups = nn.ModuleList()
        self.cross_scale_downs = nn.ModuleList()
        for i in range(self.n_scales - 1):
            # Coarser -> finer (upsample channel mixing)
            self.cross_scale_ups.append(nn.Linear(d_model, d_model))
            # Finer -> coarser (downsample channel mixing)
            self.cross_scale_downs.append(nn.Linear(d_model, d_model))

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.n_scales)]
        )

    def forward(self, multi_scale):
        out = []
        for i, (ms, temporal_mlp, norm) in enumerate(
            zip(multi_scale, self.temporal_mlps, self.norms)
        ):
            # Temporal mixing: [B, T_i, D] -> permute -> MLP on T -> permute back
            residual = ms
            ms_t = temporal_mlp(ms.permute(0, 2, 1)).permute(0, 2, 1)
            ms = norm(residual + ms_t)
            out.append(ms)

        # Cross-scale mixing
        mixed = [o.clone() for o in out]
        for i in range(self.n_scales - 1):
            # Coarse (i+1) -> Fine (i): upsample
            coarse = out[i + 1]
            fine_len = out[i].shape[1]
            upsampled = F.interpolate(
                coarse.permute(0, 2, 1), size=fine_len, mode='linear', align_corners=False
            ).permute(0, 2, 1)
            mixed[i] = mixed[i] + self.cross_scale_ups[i](upsampled)

            # Fine (i) -> Coarse (i+1): downsample
            fine = out[i]
            coarse_len = out[i + 1].shape[1]
            downsampled = F.adaptive_avg_pool1d(
                fine.permute(0, 2, 1), coarse_len
            ).permute(0, 2, 1)
            mixed[i + 1] = mixed[i + 1] + self.cross_scale_downs[i](downsampled)

        return mixed


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

SOTA_MODELS = {
    "DLinear": DLinear,
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
    "TimeMixer": TimeMixer,
}


def create_sota_model(name, seq_len=30, pred_len=3, n_features=2):
    """Create a SOTA baseline model by name."""
    if name not in SOTA_MODELS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(SOTA_MODELS.keys())}")
    return SOTA_MODELS[name](seq_len=seq_len, pred_len=pred_len, n_features=n_features)
