"""
Simplified layers for conference version of TS-Booster.

Simplifications vs thesis version:
- Single fixed moving average (no learnable multi-scale weights)
- Single expert round (no multi-round boosting)
- No SNR gate
"""

import torch
import torch.nn as nn


# === Decomposition ===


class MovingAvgDecomp(nn.Module):
    """Fixed single moving average decomposition."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        # x: (Batch, Time, Channel)
        trend = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal


# === Expert ===


class MultiScaleExpert(nn.Module):
    """
    Inception-style Multi-Scale CNN Expert.
    Multiple parallel conv branches with different kernel sizes.
    Outputs [B, pred_len] directly.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        input_channels,
        hidden_dim=32,
        kernels=[3, 5, 7],
        dropout_p=0.3,
    ):
        super().__init__()
        self.pred_len = pred_len
        branch_dim = hidden_dim // len(kernels)

        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_channels, branch_dim, kernel_size=k, padding=(k - 1) // 2
                    ),
                    nn.BatchNorm1d(branch_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout_p),
                )
            )

        concat_dim = branch_dim * len(kernels)
        self.fc = nn.Linear(seq_len * concat_dim, pred_len)

    def forward(self, x):
        # x: [B, S, C]
        B, S, C = x.shape
        x_perm = x.permute(0, 2, 1)  # [B, C, S]
        feats = [branch(x_perm) for branch in self.branches]
        feat = torch.cat(feats, dim=1)  # [B, concat_dim, S]
        feat = feat.reshape(B, -1)
        out = self.fc(feat)  # [B, pred_len]
        return out


# === Backbones ===


class LinearBackbone(nn.Module):
    """Simple Linear Backbone."""

    def __init__(self, seq_len, input_channels, pred_len, **kwargs):
        super().__init__()
        self.flatten_dim = seq_len * input_channels
        self.pred_len = pred_len
        self.input_channels = input_channels
        self.fc = nn.Linear(self.flatten_dim, pred_len * input_channels)

    def forward(self, x):
        B, _, C = x.shape
        x_flat = x.reshape(B, -1)
        out_flat = self.fc(x_flat)
        out = out_flat.reshape(B, self.pred_len, C)
        if out.size(-1) == 1:
            out = out.squeeze(-1)
        else:
            out = out[..., 0]
        return out


class LSTMBackbone(nn.Module):
    def __init__(self, seq_len, input_channels, pred_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_channels, hidden_dim,
            num_layers=num_layers, batch_first=True, dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUBackbone(nn.Module):
    def __init__(self, seq_len, input_channels, pred_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_channels, hidden_dim,
            num_layers=num_layers, batch_first=True, dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TransformerBackbone(nn.Module):
    def __init__(self, seq_len, input_channels, pred_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4,
            dim_feedforward=hidden_dim * 2, dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :])
