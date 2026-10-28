"""The TS-Booster paper's reimplementations of its deep comparators (Table 6).

Ported unchanged from the paper code (src/models/sota_baselines.py, configured as
in revision/phase10_lr_tuning). They are not TSLib's DLinear/PatchTST/iTransformer:
DLinear pads its moving average with zeros; PatchTST uses patch length 5, stride 3,
a learnable position embedding and RevIN; iTransformer has no normalization and a
learnable variate embedding. Channel 0 of the input is the forecast target.
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible instance normalization (Kim et al., ICLR 2022) with affine terms."""

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
        return x * self._std + self._mean


class MovingAvgDecomp(nn.Module):
    """x = seasonal + trend with a zero-padded moving average (AvgPool1d counts the padding)."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend


class DLinear(nn.Module):
    """Decomposition and per-channel linear maps (Zeng et al., AAAI 2023)."""

    def __init__(self, seq_len=30, pred_len=3, n_features=2, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.decomp = MovingAvgDecomp(kernel_size)
        self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_features)])
        self.linear_trend = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_features)])

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal_out = torch.stack([self.linear_seasonal[i](seasonal[:, :, i]) for i in range(self.n_features)],
                                   dim=-1)
        trend_out = torch.stack([self.linear_trend[i](trend[:, :, i]) for i in range(self.n_features)], dim=-1)
        return (seasonal_out + trend_out)[:, :, 0]


class PatchTST(nn.Module):
    """Channel-independent patch Transformer (Nie et al., ICLR 2023)."""

    def __init__(self, seq_len=30, pred_len=3, n_features=2, patch_len=5, stride=3, d_model=128, n_heads=8,
                 n_layers=3, d_ff=256, dropout=0.1, use_revin=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.use_revin = use_revin
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                                   dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)
        self.revin = RevIN(n_features) if use_revin else None

    def forward(self, x):
        B, T, C = x.shape
        if self.use_revin:
            x = self.revin.normalize(x)
        channel_preds = []
        for c in range(C):
            patches = x[:, :, c].unfold(dimension=1, size=self.patch_len, step=self.stride)
            encoded = self.encoder(self.patch_embed(patches) + self.pos_embed)
            channel_preds.append(self.head(encoded.reshape(B, -1)))
        out = torch.stack(channel_preds, dim=-1)
        if self.use_revin:
            out = self.revin.denormalize(out)
        return out[:, :, 0]


class iTransformer(nn.Module):
    """Attention across variates, each embedded from its whole window (Liu et al., ICLR 2024)."""

    def __init__(self, seq_len=30, pred_len=3, n_features=2, d_model=128, n_heads=4, n_layers=2, d_ff=256,
                 dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.d_model = d_model
        self.variate_embed = nn.Linear(seq_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                                   dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        tokens = self.variate_embed(x.permute(0, 2, 1)) + self.pos_embed
        return self.head(self.encoder(tokens))[:, 0, :]
