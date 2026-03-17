"""
SOTA Deep Learning Baselines for conference paper.

- DLinear (AAAI 2023): Moving avg decomposition + dual Linear
- PatchTST (ICLR 2023): Patch-based Transformer encoder + RevIN
- iTransformer (ICLR 2024): Inverted attention across variates

All models: forward([B, seq_len, n_features]) -> pred[B, pred_len]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Shared Components
# ──────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022)."""

    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(n_features))
            self.affine_bias = nn.Parameter(torch.zeros(n_features))

    def normalize(self, x):
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        x = x * self._std + self._mean
        return x


class MovingAvgDecomp(nn.Module):
    """Simple moving average decomposition."""

    def __init__(self, kernel_size):
        super().__init__()
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


# ──────────────────────────────────────────────────────────────
# DLinear (AAAI 2023)
# ──────────────────────────────────────────────────────────────

class DLinear(nn.Module):
    """Decomposition + dual Linear layers (channel-independent)."""

    def __init__(self, seq_len=30, pred_len=3, n_features=2, kernel_size=25):
        super().__init__()
        self.n_features = n_features
        self.decomp = MovingAvgDecomp(kernel_size)
        self.linear_seasonal = nn.ModuleList(
            [nn.Linear(seq_len, pred_len) for _ in range(n_features)]
        )
        self.linear_trend = nn.ModuleList(
            [nn.Linear(seq_len, pred_len) for _ in range(n_features)]
        )

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal_out = torch.stack(
            [self.linear_seasonal[i](seasonal[:, :, i]) for i in range(self.n_features)],
            dim=-1,
        )
        trend_out = torch.stack(
            [self.linear_trend[i](trend[:, :, i]) for i in range(self.n_features)],
            dim=-1,
        )
        out = seasonal_out + trend_out
        return out[:, :, 0]


# ──────────────────────────────────────────────────────────────
# PatchTST (ICLR 2023)
# ──────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """Patching + channel-independent Transformer encoder with RevIN."""

    def __init__(
        self, seq_len=30, pred_len=3, n_features=2,
        patch_len=5, stride=3, d_model=128, n_heads=8, n_layers=3,
        d_ff=256, dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride

        self.n_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)
        self.revin = RevIN(n_features)

    def forward(self, x):
        B, T, C = x.shape
        x = self.revin.normalize(x)

        channel_preds = []
        for c in range(C):
            xc = x[:, :, c]
            patches = xc.unfold(dimension=1, size=self.patch_len, step=self.stride)
            patches = self.patch_embed(patches) + self.pos_embed
            encoded = self.encoder(patches)
            flat = encoded.reshape(B, -1)
            pred = self.head(flat)
            channel_preds.append(pred)

        out = torch.stack(channel_preds, dim=-1)
        out = self.revin.denormalize(out)
        return out[:, :, 0]


# ──────────────────────────────────────────────────────────────
# iTransformer (ICLR 2024)
# ──────────────────────────────────────────────────────────────

class iTransformer(nn.Module):
    """Inverted Transformer — attention across variates, not time steps."""

    def __init__(
        self, seq_len=30, pred_len=3, n_features=2,
        d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
    ):
        super().__init__()
        self.variate_embed = nn.Linear(seq_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x_inv = x.permute(0, 2, 1)
        tokens = self.variate_embed(x_inv) + self.pos_embed
        encoded = self.encoder(tokens)
        out = self.head(encoded)
        return out[:, 0, :]


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

SOTA_MODELS = {
    "DLinear": DLinear,
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
}

SOTA_LR = {
    "DLinear": 1e-4,
    "PatchTST": 1e-4,
    "iTransformer": 5e-4,
}


def create_sota_model(name, seq_len=30, pred_len=3, n_features=2):
    """Create a SOTA baseline model by name."""
    if name not in SOTA_MODELS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(SOTA_MODELS.keys())}")
    return SOTA_MODELS[name](seq_len=seq_len, pred_len=pred_len, n_features=n_features)
