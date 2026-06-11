"""
Additional SOTA Deep Learning Baselines for TS-Booster Paper (Revision Phase 11).

Standalone implementations of 5 more recent/standard top-venue models, in the
same style and scale as src/models/sota_baselines.py (DLinear, PatchTST,
iTransformer, TimeMixer):

- Autoformer (NeurIPS 2021): Series decomposition + Auto-Correlation
- FEDformer  (ICML 2022): Frequency Enhanced Decomposed Transformer
- ETSformer  (arXiv 2022): Exponential smoothing + frequency attention
- LightTS    (arXiv 2022): Light sampling-oriented MLP (continuous + interval)
- TimesNet   (ICLR 2023): FFT period detection + Inception-style 2D conv

All models: forward([B, seq_len, n_features]) -> (pred[B, pred_len], pred, None, None, None)
Compatible with train_baseline() / train_v11() 5-tuple format.

Architectures are scaled down (small d_model / few layers) to match the
30 -> 3 short-window financial volatility task, following the same
"faithful reimplementation of key mechanisms, not official code" philosophy
used for the existing SOTA baselines.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sota_baselines import MovingAvgDecomp


# ──────────────────────────────────────────────────────────────
# Shared: Mixture-of-Experts decomposition (FEDformer)
# ──────────────────────────────────────────────────────────────

class MOEDecomp(nn.Module):
    """Mixture of multiple moving-average experts, mixed by a learned gate.

    Ref: FEDformer (Zhou et al., ICML 2022), Sec 3.2.
    `dim` is the feature dimension of the tensor being decomposed (works for
    both raw input features and hidden d_model representations).
    """

    def __init__(self, dim, kernel_sizes=(5, 9, 13)):
        super().__init__()
        self.decomps = nn.ModuleList([MovingAvgDecomp(k) for k in kernel_sizes])
        self.gate = nn.Linear(dim, len(kernel_sizes))

    def forward(self, x):
        # x: [B, T, dim]
        trends = torch.stack([d(x)[1] for d in self.decomps], dim=-1)  # [B,T,dim,K]
        w = torch.softmax(self.gate(x), dim=-1)  # [B,T,K]
        trend = (trends * w.unsqueeze(2)).sum(-1)  # [B,T,dim]
        seasonal = x - trend
        return seasonal, trend


# ──────────────────────────────────────────────────────────────
# Shared: decomposition-style encoder/decoder skeleton
# (used by Autoformer with AutoCorrelation, FEDformer with Fourier blocks)
# ──────────────────────────────────────────────────────────────

class _DecompEncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff, decomp1, decomp2, dropout=0.1):
        super().__init__()
        self.attn = attn
        self.decomp1 = decomp1
        self.decomp2 = decomp2
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(x, x, x))
        x, _ = self.decomp1(x)
        y = self.dropout(F.gelu(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res


class _DecompDecoderLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, d_model, d_ff, c_out,
                 decomp1, decomp2, decomp3, dropout=0.1):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.decomp1 = decomp1
        self.decomp2 = decomp2
        self.decomp3 = decomp3
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.trend_proj = nn.Linear(d_model, c_out)

    def forward(self, x, cross):
        x = x + self.dropout(self.self_attn(x, x, x))
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attn(x, cross, cross))
        x, trend2 = self.decomp2(x)
        y = self.dropout(F.gelu(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)
        trend = self.trend_proj(trend1 + trend2 + trend3)
        return x, trend


# ──────────────────────────────────────────────────────────────
# Autoformer (NeurIPS 2021)
# ──────────────────────────────────────────────────────────────

class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism: replaces self-attention with FFT-based
    period detection + time-delay aggregation.

    Ref: Autoformer (Wu et al., NeurIPS 2021).

    Simplification: the top-k delays are selected from the *batch-averaged*
    correlation (rather than per-sample), which keeps the aggregation a
    cheap vectorized `torch.roll` while preserving the "aggregate by
    dominant lags" mechanism.
    """

    def __init__(self, d_model, n_heads, factor=1, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.factor = factor
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def time_delay_agg(self, values, corr):
        # values, corr: [B, H, E, L]
        L = values.shape[-1]
        top_k = max(1, int(self.factor * math.log(max(L, 2))))
        top_k = min(top_k, L)
        mean_corr = corr.mean(dim=(0, 1, 2))  # [L]
        weights, delays = torch.topk(mean_corr, top_k)
        tmp_weights = torch.softmax(weights, dim=-1)
        agg = torch.zeros_like(values)
        for i in range(top_k):
            shift = int(delays[i].item())
            agg = agg + torch.roll(values, shifts=-shift, dims=-1) * tmp_weights[i]
        return agg

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        S = keys.shape[1]
        H = self.n_heads

        q = self.query_proj(queries).view(B, L, H, -1)
        k = self.key_proj(keys).view(B, S, H, -1)
        v = self.value_proj(values).view(B, S, H, -1)

        if L > S:
            pad = torch.zeros(B, L - S, H, v.shape[-1], device=v.device, dtype=v.dtype)
            v = torch.cat([v, pad], dim=1)
            k = torch.cat([k, pad], dim=1)
        else:
            v = v[:, :L, :, :]
            k = k[:, :L, :, :]

        q = q.permute(0, 2, 3, 1)  # [B,H,E,L]
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        q_fft = torch.fft.rfft(q, dim=-1)
        k_fft = torch.fft.rfft(k, dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=L, dim=-1)

        out = self.time_delay_agg(v, corr)  # [B,H,E,L]
        out = out.permute(0, 3, 1, 2).reshape(B, L, -1)  # [B,L,d_model]
        return self.out_proj(out)


class Autoformer(nn.Module):
    """Autoformer: Series decomposition + Auto-Correlation encoder-decoder.
    Ref: Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation
    for Long-Term Series Forecasting", NeurIPS 2021.
    """

    def __init__(self, seq_len=30, pred_len=3, n_features=2, d_model=32, n_heads=4,
                 e_layers=1, d_layers=1, d_ff=64, moving_avg=5, factor=1, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = seq_len // 2
        self.n_features = n_features

        self.init_decomp = MovingAvgDecomp(moving_avg)
        self.enc_embedding = nn.Linear(n_features, d_model)
        self.dec_embedding = nn.Linear(n_features, d_model)

        self.encoder = nn.ModuleList([
            _DecompEncoderLayer(
                AutoCorrelation(d_model, n_heads, factor, dropout),
                d_model, d_ff,
                MovingAvgDecomp(moving_avg), MovingAvgDecomp(moving_avg), dropout,
            ) for _ in range(e_layers)
        ])
        self.decoder = nn.ModuleList([
            _DecompDecoderLayer(
                AutoCorrelation(d_model, n_heads, factor, dropout),
                AutoCorrelation(d_model, n_heads, factor, dropout),
                d_model, d_ff, n_features,
                MovingAvgDecomp(moving_avg), MovingAvgDecomp(moving_avg), MovingAvgDecomp(moving_avg),
                dropout,
            ) for _ in range(d_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.seasonal_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        B = x.shape[0]
        seasonal_init, trend_init = self.init_decomp(x)

        mean = x.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(B, self.pred_len, self.n_features, device=x.device, dtype=x.dtype)
        trend_dec = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_dec = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x)
        for layer in self.encoder:
            enc_out = layer(enc_out)

        dec_out = self.dec_embedding(seasonal_dec)
        trend_acc = trend_dec
        for layer in self.decoder:
            dec_out, trend_part = layer(dec_out, enc_out)
            trend_acc = trend_acc + trend_part

        seasonal_out = self.seasonal_proj(self.dec_norm(dec_out))
        out = (seasonal_out + trend_acc)[:, -self.pred_len:, :]
        pred = out[:, :, 0]
        return pred, pred, None, None, None


# ──────────────────────────────────────────────────────────────
# FEDformer (ICML 2022)
# ──────────────────────────────────────────────────────────────

class FourierBlock(nn.Module):
    """Frequency Enhanced Block (FEB-f): self-attention replacement that
    mixes a fixed set of low-frequency Fourier modes with learnable
    complex weights.
    Ref: FEDformer (Zhou et al., ICML 2022), Sec 3.1.
    """

    def __init__(self, d_model, seq_len, modes=8):
        super().__init__()
        self.modes = max(1, min(modes, seq_len // 2 + 1))
        scale = 1 / (d_model * d_model)
        self.weights_real = nn.Parameter(scale * torch.rand(d_model, d_model, self.modes))
        self.weights_imag = nn.Parameter(scale * torch.rand(d_model, d_model, self.modes))

    def forward(self, q, k=None, v=None):
        # q: [B, L, D]  (k, v unused — self-only block)
        B, L, D = q.shape
        x_ft = torch.fft.rfft(q, dim=1)  # [B, L//2+1, D]
        m = min(self.modes, x_ft.shape[1])

        weight = torch.complex(self.weights_real[:, :, :m], self.weights_imag[:, :, :m])
        x_ft_m = x_ft[:, :m, :].to(torch.cfloat)
        out_modes = torch.einsum('bmi,iom->bmo', x_ft_m, weight)

        out_ft = torch.zeros(B, x_ft.shape[1], D, device=q.device, dtype=torch.cfloat)
        out_ft[:, :m, :] = out_modes
        return torch.fft.irfft(out_ft, n=L, dim=1)


class FourierCrossAttention(nn.Module):
    """Frequency Enhanced Attention (FEA-f): cross-attention computed in the
    frequency domain over a small set of low-frequency modes.
    Ref: FEDformer (Zhou et al., ICML 2022), Sec 3.1.
    """

    def __init__(self, d_model, seq_len_q, seq_len_kv, modes=8):
        super().__init__()
        self.modes = max(1, min(modes, seq_len_q // 2 + 1, seq_len_kv // 2 + 1))
        scale = 1 / (d_model * d_model)
        self.weights_real = nn.Parameter(scale * torch.rand(d_model, d_model, self.modes))
        self.weights_imag = nn.Parameter(scale * torch.rand(d_model, d_model, self.modes))

    def forward(self, q, k, v):
        B, Lq, D = q.shape
        xq_ft = torch.fft.rfft(q, dim=1)
        xk_ft = torch.fft.rfft(k, dim=1)
        xv_ft = torch.fft.rfft(v, dim=1)
        m = min(self.modes, xq_ft.shape[1], xk_ft.shape[1])

        xq_m, xk_m, xv_m = xq_ft[:, :m, :], xk_ft[:, :m, :], xv_ft[:, :m, :]
        attn = (torch.einsum('bme,bne->bmn', xq_m.real, xk_m.real)
                + torch.einsum('bme,bne->bmn', xq_m.imag, xk_m.imag))
        attn = torch.softmax(attn, dim=-1)
        out_real = torch.einsum('bmn,bne->bme', attn, xv_m.real)
        out_imag = torch.einsum('bmn,bne->bme', attn, xv_m.imag)
        out_m = torch.complex(out_real, out_imag)

        weight = torch.complex(self.weights_real[:, :, :m], self.weights_imag[:, :, :m])
        out_m = torch.einsum('bmi,iom->bmo', out_m, weight)

        out_ft = torch.zeros(B, xq_ft.shape[1], D, device=q.device, dtype=torch.cfloat)
        out_ft[:, :m, :] = out_m
        return torch.fft.irfft(out_ft, n=Lq, dim=1)


class FEDformer(nn.Module):
    """FEDformer: Frequency Enhanced Decomposed Transformer.
    Ref: Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer
    for Long-Term Series Forecasting", ICML 2022. (FEB-f / FEA-f variant)
    """

    def __init__(self, seq_len=30, pred_len=3, n_features=2, d_model=32, d_ff=64,
                 e_layers=1, d_layers=1, modes=8, kernel_sizes=(5, 9, 13), dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = seq_len // 2
        self.n_features = n_features

        self.init_decomp = MOEDecomp(n_features, kernel_sizes)
        self.enc_embedding = nn.Linear(n_features, d_model)
        self.dec_embedding = nn.Linear(n_features, d_model)

        enc_len = seq_len
        dec_len = self.label_len + pred_len

        self.encoder = nn.ModuleList([
            _DecompEncoderLayer(
                FourierBlock(d_model, enc_len, modes),
                d_model, d_ff,
                MOEDecomp(d_model, kernel_sizes), MOEDecomp(d_model, kernel_sizes), dropout,
            ) for _ in range(e_layers)
        ])
        self.decoder = nn.ModuleList([
            _DecompDecoderLayer(
                FourierBlock(d_model, dec_len, modes),
                FourierCrossAttention(d_model, dec_len, enc_len, modes),
                d_model, d_ff, n_features,
                MOEDecomp(d_model, kernel_sizes), MOEDecomp(d_model, kernel_sizes), MOEDecomp(d_model, kernel_sizes),
                dropout,
            ) for _ in range(d_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.seasonal_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        B = x.shape[0]
        seasonal_init, trend_init = self.init_decomp(x)

        mean = x.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(B, self.pred_len, self.n_features, device=x.device, dtype=x.dtype)
        trend_dec = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_dec = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_out = self.enc_embedding(x)
        for layer in self.encoder:
            enc_out = layer(enc_out)

        dec_out = self.dec_embedding(seasonal_dec)
        trend_acc = trend_dec
        for layer in self.decoder:
            dec_out, trend_part = layer(dec_out, enc_out)
            trend_acc = trend_acc + trend_part

        seasonal_out = self.seasonal_proj(self.dec_norm(dec_out))
        out = (seasonal_out + trend_acc)[:, -self.pred_len:, :]
        pred = out[:, :, 0]
        return pred, pred, None, None, None


# ──────────────────────────────────────────────────────────────
# ETSformer (arXiv 2022)
# ──────────────────────────────────────────────────────────────

class ETSformer(nn.Module):
    """ETSformer: classical exponential-smoothing decomposition (level +
    damped growth/trend + seasonality), with seasonality extrapolated via
    low-frequency Fourier modes.
    Ref: Woo et al., "ETSformer: Exponential Smoothing Transformers for
    Time-series Forecasting", arXiv 2022.

    Simplification: level/growth recursion uses per-channel learnable
    smoothing & damping coefficients (sigmoid-parameterized scalars) rather
    than multi-head ES-attention; seasonality uses the K lowest non-DC
    Fourier modes (shared across the batch) extrapolated via their
    amplitude/phase to the forecast horizon.
    """

    def __init__(self, seq_len=30, pred_len=3, n_features=2, K=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.K = K

        self.alpha = nn.Parameter(torch.zeros(n_features))   # level smoothing
        self.beta = nn.Parameter(torch.zeros(n_features))    # growth smoothing
        self.damping = nn.Parameter(torch.ones(n_features) * 1.5)  # trend damping

    def forward(self, x):
        B, T, C = x.shape
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        damping = torch.sigmoid(self.damping)

        level = x[:, 0, :]
        growth = x[:, 1, :] - x[:, 0, :]
        levels = [level]
        for t in range(1, T):
            obs = x[:, t, :]
            new_level = alpha * obs + (1 - alpha) * (level + damping * growth)
            new_growth = beta * (new_level - level) + (1 - beta) * damping * growth
            level, growth = new_level, new_growth
            levels.append(level)

        # Seasonality: residual after removing the level-only reconstruction
        levels = torch.stack(levels, dim=1)  # [B,T,C]
        season = x - levels

        season_ft = torch.fft.rfft(season, dim=1)  # [B, T//2+1, C]
        K = max(1, min(self.K, season_ft.shape[1] - 1))
        modes = season_ft[:, 1:1 + K, :]  # exclude DC

        freqs = torch.arange(1, K + 1, device=x.device, dtype=x.dtype) / T  # [K]
        future_t = torch.arange(T, T + self.pred_len, device=x.device, dtype=x.dtype)  # [pred_len]
        angle = 2 * math.pi * freqs[:, None] * future_t[None, :]  # [K, pred_len]

        amp = modes.abs() * 2.0 / T          # [B,K,C]
        phase = modes.angle()                # [B,K,C]
        cos_term = torch.cos(angle.unsqueeze(0).unsqueeze(-1) + phase.unsqueeze(2))  # [B,K,pred_len,C]
        season_future = (amp.unsqueeze(2) * cos_term).sum(dim=1)  # [B,pred_len,C]

        # Damped multi-step trend extrapolation
        h = torch.arange(1, self.pred_len + 1, device=x.device, dtype=x.dtype)  # [pred_len]
        damp_powers = damping[None, :] ** h[:, None]  # [pred_len, C]
        cum_damp = torch.cumsum(damp_powers, dim=0)   # [pred_len, C]
        growth_future = cum_damp.unsqueeze(0) * growth.unsqueeze(1)  # [B,pred_len,C]

        level_future = level.unsqueeze(1).expand(-1, self.pred_len, -1)

        out = level_future + growth_future + season_future  # [B,pred_len,C]
        pred = out[:, :, 0]
        return pred, pred, None, None, None


# ──────────────────────────────────────────────────────────────
# LightTS (arXiv 2022)
# ──────────────────────────────────────────────────────────────

class IEBlock(nn.Module):
    """Information Exchange Block: temporal MLP per channel + a channel
    mixing projection.
    Ref: LightTS (Zhang et al., "Less Is More: Fast Multivariate Time Series
    Forecasting with Light Sampling-oriented MLP Structures", arXiv 2022).
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_features):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.channel_proj = nn.Linear(n_features, n_features)

    def forward(self, x):
        # x: [B, n_features, in_dim]
        x = self.temporal(x)             # [B, n_features, out_dim]
        x = self.channel_proj(x.permute(0, 2, 1))  # [B, out_dim, n_features]
        return x.permute(0, 2, 1)        # [B, n_features, out_dim]


class LightTS(nn.Module):
    """LightTS: continuous-sampling and interval-sampling MLP branches,
    each processed by an IEBlock, fused for the final forecast.
    Ref: Zhang et al., arXiv 2022.
    """

    def __init__(self, seq_len=30, pred_len=3, n_features=2, chunk_size=6, hidden_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.chunk_size = chunk_size
        self.n_chunks = math.ceil(seq_len / chunk_size)
        self.pad_len = self.n_chunks * chunk_size - seq_len

        # continuous-sampling branch (contiguous chunks)
        self.cont_intra = IEBlock(chunk_size, hidden_dim, hidden_dim, n_features)
        self.cont_inter = IEBlock(self.n_chunks * hidden_dim, hidden_dim, pred_len, n_features)

        # interval-sampling branch (strided sub-sequences)
        self.inter_intra = IEBlock(self.n_chunks, hidden_dim, hidden_dim, n_features)
        self.inter_inter = IEBlock(chunk_size * hidden_dim, hidden_dim, pred_len, n_features)

        self.out_proj = nn.Linear(2 * pred_len, pred_len)

    def forward(self, x):
        B, T, C = x.shape
        if self.pad_len > 0:
            pad = x[:, -1:, :].repeat(1, self.pad_len, 1)
            x_p = torch.cat([x, pad], dim=1)
        else:
            x_p = x

        xc = x_p.permute(0, 2, 1)  # [B,C,Tp]
        cont = xc.reshape(B, C, self.n_chunks, self.chunk_size)  # contiguous chunks
        inter = cont.permute(0, 1, 3, 2)  # strided sub-sequences: [B,C,chunk_size,n_chunks]

        cont_feats = [self.cont_intra(cont[:, :, i, :]) for i in range(self.n_chunks)]
        cont_out = self.cont_inter(torch.cat(cont_feats, dim=-1))  # [B,C,pred_len]

        inter_feats = [self.inter_intra(inter[:, :, j, :]) for j in range(self.chunk_size)]
        inter_out = self.inter_inter(torch.cat(inter_feats, dim=-1))  # [B,C,pred_len]

        out = self.out_proj(torch.cat([cont_out, inter_out], dim=-1))  # [B,C,pred_len]
        out = out.permute(0, 2, 1)  # [B,pred_len,C]
        pred = out[:, :, 0]
        return pred, pred, None, None, None


# ──────────────────────────────────────────────────────────────
# TimesNet (ICLR 2023)
# ──────────────────────────────────────────────────────────────

class InceptionBlock2D(nn.Module):
    """Multi-kernel 2D Inception block used inside TimesBlock."""

    def __init__(self, in_ch, out_ch, kernel_sizes=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])

    def forward(self, x):
        return torch.stack([c(x) for c in self.convs], dim=-1).mean(-1)


class TimesBlock(nn.Module):
    """TimesBlock: detects the top-k dominant periods via FFT, reshapes the
    1D series into 2D tensors [n_periods, period_length] for each period,
    applies an Inception-style 2D conv, then aggregates back weighted by
    FFT amplitude.
    Ref: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General
    Time Series Analysis", ICLR 2023.
    """

    def __init__(self, seq_len, d_model, d_ff, k=2, kernel_sizes=(1, 3, 5)):
        super().__init__()
        self.seq_len = seq_len
        self.k = k
        self.conv = nn.Sequential(
            InceptionBlock2D(d_model, d_ff, kernel_sizes),
            nn.GELU(),
            InceptionBlock2D(d_ff, d_model, kernel_sizes),
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs().mean(dim=(0, 2))  # [T//2+1]
        amp = amp.clone()
        amp[0] = 0  # ignore DC component

        k = max(1, min(self.k, amp.shape[0] - 1))
        top_vals, top_idx = torch.topk(amp, k)
        period_weight = F.softmax(top_vals, dim=0)  # [k]

        outs = []
        for i in range(k):
            freq_idx = max(top_idx[i].item(), 1)
            period = max(1, min(T, T // freq_idx))

            n_rows = math.ceil(T / period)
            pad_len = n_rows * period - T
            if pad_len > 0:
                pad = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
                xp = torch.cat([x, pad], dim=1)
            else:
                xp = x

            xp = xp.reshape(B, n_rows, period, D).permute(0, 3, 1, 2)  # [B,D,n_rows,period]
            out2d = self.conv(xp)
            out = out2d.permute(0, 2, 3, 1).reshape(B, n_rows * period, D)[:, :T, :]
            outs.append(out)

        outs = torch.stack(outs, dim=-1)  # [B,T,D,k]
        weight = period_weight.reshape(1, 1, 1, -1)
        res = (outs * weight).sum(-1)
        return res + x


class TimesNet(nn.Module):
    """TimesNet: stack of TimesBlocks on an embedded representation,
    flattened and projected to the forecast horizon.
    Ref: Wu et al., ICLR 2023.
    """

    def __init__(self, seq_len=30, pred_len=3, n_features=2, d_model=16, d_ff=32,
                 e_layers=2, k=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([TimesBlock(seq_len, d_model, d_ff, k) for _ in range(e_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, x):
        B = x.shape[0]
        h = self.embed(x)  # [B,T,d_model]
        for block in self.blocks:
            h = self.norm(block(h))
        out = self.proj(h.reshape(B, -1))  # [B,pred_len]
        return out, out, None, None, None


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

SOTA_MODELS_V2 = {
    "Autoformer": Autoformer,
    "FEDformer": FEDformer,
    "ETSformer": ETSformer,
    "LightTS": LightTS,
    "TimesNet": TimesNet,
}


def create_sota_model_v2(name, seq_len=30, pred_len=3, n_features=2):
    """Create a new SOTA baseline model (Phase 11) by name."""
    if name not in SOTA_MODELS_V2:
        raise ValueError(f"Unknown model: {name}. Choose from {list(SOTA_MODELS_V2.keys())}")
    return SOTA_MODELS_V2[name](seq_len=seq_len, pred_len=pred_len, n_features=n_features)
