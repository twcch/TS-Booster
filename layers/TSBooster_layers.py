"""TS-Booster building blocks and the TSLib-facing forecaster shared by its variants.

The layer math, parameter names and construction order follow the journal code
(docs/TS-Booster_ori/src/models): zero-padded moving averages, Inception-style CNN
experts, the four reference hosts and FlexibleTSBooster's forward. Inside these
modules the forecast target is channel 0; ``SingleTargetForecaster`` moves TSLib's
target (the last column under ``features=MS``) there, so the same seed builds the
same weights as the original code.
"""

import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.model_contracts import single_target_index

BACKBONES = ("linear", "lstm", "gru", "transformer")
GATES = ("snr", "learned", "const", "none")
# trend/seasonal/both append decomposition parts to the expert input; raw ([x || x]) and
# zeros ([x || 0]) are width-matched controls that keep the decomposition for the gate only.
ENRICH_MODES = ("trend", "seasonal", "both", "raw", "zeros")
EXPERT_TYPES = ("cnn", "mlp")
# Hard-coded in the journal code; recorded in effective_config instead of exposed as flags.
RNN_DROPOUT = 0.2
TRANSFORMER_DROPOUT = 0.1
CONST_GATE_VALUE = 0.5


# === Decomposition ===


class FixedMovingAvg(nn.Module):
    """Centered moving average; AvgPool1d zero-pads and counts the padding."""

    def __init__(self, kernel_size):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        return self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)


class MultiScaleDecomp(nn.Module):
    """Softmax-weighted mixture of moving averages: x = trend + seasonal.

    learnable=False keeps the logits at zero as a buffer, i.e. a fixed equal mixture.
    """

    def __init__(self, kernels=(7, 13), learnable=True):
        super().__init__()
        self.kernels = list(kernels)
        self.avgs = nn.ModuleList([FixedMovingAvg(k) for k in self.kernels])
        if learnable:
            self.weights = nn.Parameter(torch.zeros(len(self.kernels)))
        else:
            self.register_buffer("weights", torch.zeros(len(self.kernels)))

    def mixture_weights(self):
        return torch.softmax(self.weights.detach(), dim=0)

    def forward(self, x):
        w = torch.softmax(self.weights, dim=0)
        trend = sum(w[i] * avg(x) for i, avg in enumerate(self.avgs))
        return trend, x - trend


# === Experts ===


class MultiScaleExpert(nn.Module):
    """Inception-style CNN expert: parallel conv branches, concatenated, then Linear."""

    def __init__(self, seq_len, pred_len, input_channels, hidden_dim=32, kernels=(3, 5, 7), dropout_p=0.3):
        super().__init__()
        branch_dim = hidden_dim // len(kernels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, branch_dim, kernel_size=k, padding=(k - 1) // 2),
                nn.BatchNorm1d(branch_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_p),
            )
            for k in kernels
        ])
        self.fc = nn.Linear(seq_len * branch_dim * len(kernels), pred_len)

    def forward(self, x):
        feat = torch.cat([branch(x.permute(0, 2, 1)) for branch in self.branches], dim=1)
        return self.fc(feat.reshape(x.shape[0], -1))


class MLPExpert(nn.Module):
    """MLP expert over the flattened window."""

    def __init__(self, seq_len, pred_len, input_channels, hidden_dim=32, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * input_channels, hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


def create_expert(expert_type, seq_len, pred_len, input_channels, hidden_dim=32, kernels=(3, 5, 7),
                  dropout_p=0.3):
    if expert_type == "cnn":
        return MultiScaleExpert(seq_len, pred_len, input_channels, hidden_dim, kernels, dropout_p)
    if expert_type == "mlp":
        return MLPExpert(seq_len, pred_len, input_channels, hidden_dim, dropout_p)
    raise ValueError(f"Unknown expert_type {expert_type!r}; choose from {EXPERT_TYPES}")


# === Hosts (backbones); each returns the channel-0 forecast [B, pred_len] ===


class LinearBackbone(nn.Module):
    """Linear map of the time-major flattened window (366 parameters at L=30, H=3, C=2)."""

    def __init__(self, seq_len, input_channels, pred_len):
        super().__init__()
        self.pred_len = pred_len
        self.fc = nn.Linear(seq_len * input_channels, pred_len * input_channels)

    def forward(self, x):
        out = self.fc(x.reshape(x.shape[0], -1))
        return out.reshape(x.shape[0], self.pred_len, x.shape[-1])[..., 0]


def _recurrent(cls, input_channels, hidden_dim, num_layers):
    # The journal code always passes dropout=0.2; with one layer PyTorch only warns,
    # but cuDNN still seeds a dropout state, so the value is kept for RNG parity.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="dropout option adds dropout")
        return cls(input_channels, hidden_dim, num_layers=num_layers, batch_first=True, dropout=RNN_DROPOUT)


class LSTMBackbone(nn.Module):
    def __init__(self, seq_len, input_channels, pred_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = _recurrent(nn.LSTM, input_channels, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUBackbone(nn.Module):
    def __init__(self, seq_len, input_channels, pred_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = _recurrent(nn.GRU, input_channels, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TransformerBackbone(nn.Module):
    """Encoder-only Transformer without positional encoding, read at the last step."""

    def __init__(self, seq_len, input_channels, pred_len, hidden_dim=64, num_layers=2, n_heads=4,
                 d_ff=128):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=d_ff,
                                                   dropout=TRANSFORMER_DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        return self.fc(self.transformer(self.input_proj(x))[:, -1, :])


class MLP7K(nn.Module):
    """Parameter-matched control: 2-hidden-layer MLP with 7,171 parameters at L=30, H=3, C=2."""

    def __init__(self, seq_len, input_channels, pred_len, h1=64, h2=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * input_channels, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, pred_len),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


def build_backbone(name, seq_len, input_channels, pred_len, *, d_model=64, e_layers=2, n_heads=4, d_ff=128):
    if name == "linear":
        return LinearBackbone(seq_len, input_channels, pred_len)
    if name == "lstm":
        return LSTMBackbone(seq_len, input_channels, pred_len, d_model, e_layers)
    if name == "gru":
        return GRUBackbone(seq_len, input_channels, pred_len, d_model, e_layers)
    if name == "transformer":
        return TransformerBackbone(seq_len, input_channels, pred_len, d_model, e_layers, n_heads, d_ff)
    raise ValueError(f"Unknown TS-Booster backbone {name!r}; choose from {BACKBONES}")


# === Booster (FlexibleTSBooster) ===


class TSBoosterCore(nn.Module):
    """y = y_base + g * sum_k expert_k(enrich(x)) with component toggles for ablations.

    use_decomp and use_experts select the paper variants: Vanilla (neither), +Decomp
    (two copies of the host on trend/seasonal), +Expert (experts on the raw window,
    no gate) and TS-Booster (both, gated). Gate parameters are registered before the
    host and the experts are built last, matching the original RNG consumption.
    """

    def __init__(self, backbone, seq_len, pred_len, input_channels, *, use_decomp=True, use_experts=True,
                 decomp_kernels=(7, 13), expert_kernels=(3, 5, 7), expert_hidden_dim=32, dropout=0.3,
                 n_experts=2, expert_type="cnn", gate="snr", enrich="trend", decomp_learnable=True):
        super().__init__()
        if gate not in GATES:
            raise ValueError(f"Unknown gate {gate!r}; choose from {GATES}")
        if enrich not in ENRICH_MODES:
            raise ValueError(f"Unknown enrichment {enrich!r}; choose from {ENRICH_MODES}")
        if use_experts and n_experts < 1:
            raise ValueError("TS-Booster needs at least one expert")
        if use_experts and expert_type == "cnn" and expert_hidden_dim < len(expert_kernels):
            raise ValueError("expert_hidden_dim must be at least the number of expert kernels")
        if use_decomp and any(k <= 0 or k % 2 == 0 for k in decomp_kernels):
            raise ValueError(f"decomposition kernels must be positive odd integers, got {list(decomp_kernels)}")
        if use_experts and expert_type == "cnn" and any(k <= 0 or k % 2 == 0 for k in expert_kernels):
            raise ValueError(f"expert kernels must be positive odd integers, got {list(expert_kernels)}")
        self.use_decomp, self.use_experts, self.enrich = use_decomp, use_experts, enrich
        # The gate only exists when the experts are enriched by the decomposition.
        self.gate = gate if use_decomp and use_experts else "none"
        if self.gate == "learned":
            self.expert_gate_param = nn.Parameter(torch.tensor(1.0))
        elif self.gate == "snr":
            self.snr_gate_w = nn.Parameter(torch.tensor(1.0))
            self.snr_gate_b = nn.Parameter(torch.tensor(0.0))
        if use_decomp and not use_experts:
            self.backbone_trend = copy.deepcopy(backbone)
            self.backbone_seasonal = copy.deepcopy(backbone)
        else:
            self.base_model = backbone
        self.decomp = MultiScaleDecomp(decomp_kernels, learnable=decomp_learnable) if use_decomp else None
        self.experts = None
        if use_experts:
            width = input_channels * (1 + (2 if enrich == "both" else 1) * use_decomp)
            self.experts = nn.ModuleList([
                create_expert(expert_type, seq_len, pred_len, width, expert_hidden_dim, expert_kernels, dropout)
                for _ in range(n_experts)
            ])

    def expert_gate(self, x, seasonal):
        if self.gate == "snr":
            seasonal_std = seasonal.std(dim=1).mean(dim=-1, keepdim=True)
            raw_std = x.std(dim=1).mean(dim=-1, keepdim=True)
            return torch.sigmoid(self.snr_gate_w * seasonal_std / (raw_std + 1e-8) + self.snr_gate_b)
        if self.gate == "learned":
            return torch.sigmoid(self.expert_gate_param)
        if self.gate == "const":
            return torch.tensor(CONST_GATE_VALUE, device=x.device)
        return None

    def forward(self, x):
        """x: [B, L, C] with the target in channel 0 -> (y_final [B, H], aux dict)."""
        if self.use_decomp and not self.use_experts:
            trend, seasonal = self.decomp(x)
            y_base = self.backbone_trend(trend) + self.backbone_seasonal(seasonal)
        else:
            y_base = self.base_model(x)
        y_expert, gate = None, None
        if self.use_experts:
            if self.use_decomp:
                trend, seasonal = self.decomp(x)
                parts = {"trend": [trend], "seasonal": [seasonal], "both": [trend, seasonal],
                         "raw": [x], "zeros": [torch.zeros_like(x)]}[self.enrich]
                x_expert = torch.cat([x, *parts], dim=-1)
            else:
                x_expert = x
            y_expert = torch.zeros_like(y_base)
            for expert in self.experts:
                y_expert = y_expert + expert(x_expert)
            if self.use_decomp:
                gate = self.expert_gate(x, seasonal)
                if gate is not None:
                    y_expert = gate * y_expert
        y_final = y_base if y_expert is None else y_base + y_expert
        return y_final, {"y_base": y_base, "y_expert": y_expert, "gate": gate}

    def diagnostics(self):
        """Learned scalars that summarize a trained booster (decomposition and gate)."""
        result = {"gate": self.gate}
        if self.decomp is not None:
            result["decomp_mixture"] = dict(zip(map(str, self.decomp.kernels),
                                                self.decomp.mixture_weights().tolist()))
        if self.gate == "snr":
            result["snr_gate_w"] = float(self.snr_gate_w.detach())
            result["snr_gate_b"] = float(self.snr_gate_b.detach())
        elif self.gate == "learned":
            result["learned_gate"] = float(torch.sigmoid(self.expert_gate_param.detach()))
        elif self.gate == "const":
            result["const_gate"] = CONST_GATE_VALUE
        return result


# === TSLib-facing forecasters ===


def _validate_single_target(configs):
    if getattr(configs, "task_name", "long_term_forecast") != "long_term_forecast":
        raise ValueError("TS-Booster models support long_term_forecast only")
    for key in ("seq_len", "pred_len", "enc_in"):
        if getattr(configs, key) <= 0:
            raise ValueError(f"{key} must be positive")
    features = getattr(configs, "features", "MS")
    if features not in {"MS", "S"} or configs.c_out != 1:
        raise ValueError("TS-Booster forecasts one target: use --features MS (or S) with --c_out 1")
    if features == "S" and configs.enc_in != 1:
        raise ValueError("--features S provides one input channel; set --enc_in 1")
    return single_target_index(configs)


class SingleTargetForecaster(nn.Module):
    """TSLib interface for hosts that forecast the target from all input channels.

    The input is reordered to [target, other channels in TSLib order]; for the
    two-channel volatility data this restores the original [Volatility, log_return].
    """

    has_auxiliary_loss = False

    def __init__(self, configs):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = configs.seq_len, configs.pred_len, configs.enc_in
        self.target_index = _validate_single_target(configs)
        order = [self.target_index] + [i for i in range(self.enc_in) if i != self.target_index]
        self.register_buffer("channel_order", torch.tensor(order, dtype=torch.long), persistent=False)
        self.effective_config = {
            "seq_len": self.seq_len, "pred_len": self.pred_len, "enc_in": self.enc_in, "c_out": 1,
            "target_channel_index": self.target_index, "internal_channel_order": order,
            "ignored_config_parameters": ["label_len", "dec_in", "d_layers", "embed", "freq", "factor",
                                          "activation", "moving_avg", "distil"],
        }

    def _check_input(self, x_enc):
        if x_enc.ndim != 3 or x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.enc_in:
            raise ValueError(f"Expected input [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}")
        return x_enc.index_select(-1, self.channel_order)


class TSBoosterForecaster(SingleTargetForecaster):
    """TS-Booster with TSLib hyperparameters.

    Reused flags: seq_len, pred_len, enc_in, dropout (expert dropout) and, for the
    host, d_model/e_layers (LSTM, GRU, Transformer) plus n_heads/d_ff (Transformer).
    TS-Booster flags (tsb_*) cover what TSLib has no equivalent for. The hosts'
    internal dropouts stay fixed as in the paper and are listed in effective_config.
    """

    supports_aux_outputs = True

    def __init__(self, configs, *, use_decomp=True, use_experts=True, gate="snr", variant="TS-Booster"):
        super().__init__(configs)
        backbone = getattr(configs, "tsb_backbone", "linear")
        host_args = {"d_model": configs.d_model, "e_layers": configs.e_layers,
                     "n_heads": configs.n_heads, "d_ff": configs.d_ff}
        if backbone in {"lstm", "gru", "transformer"} and min(configs.d_model, configs.e_layers) <= 0:
            raise ValueError("d_model and e_layers must be positive for the recurrent/attention hosts")
        if backbone == "transformer" and configs.d_model % configs.n_heads:
            raise ValueError("d_model must be divisible by n_heads for the Transformer host")
        decomp_kernels = list(getattr(configs, "tsb_decomp_kernels", [7, 13]))
        expert_kernels = list(getattr(configs, "tsb_expert_kernels", [3, 5, 7]))
        expert_hidden = getattr(configs, "tsb_expert_hidden", 32)
        n_experts = getattr(configs, "tsb_n_experts", 2)
        expert_type = getattr(configs, "tsb_expert_type", "cnn")
        enrich = getattr(configs, "tsb_enrich", "trend")
        decomp_fixed = bool(getattr(configs, "tsb_decomp_fixed", False))
        self.aux_weight = float(getattr(configs, "tsb_aux_weight", 0.3))
        self.base_weight = float(getattr(configs, "tsb_base_weight", 0.1))
        # Host first, experts last: the same seed then yields the original weights.
        host = build_backbone(backbone, self.seq_len, self.enc_in, self.pred_len, **host_args)
        self.booster = TSBoosterCore(
            host, self.seq_len, self.pred_len, self.enc_in, use_decomp=use_decomp, use_experts=use_experts,
            decomp_kernels=decomp_kernels, expert_kernels=expert_kernels, expert_hidden_dim=expert_hidden,
            dropout=configs.dropout, n_experts=n_experts, expert_type=expert_type, gate=gate, enrich=enrich,
            decomp_learnable=not decomp_fixed)
        self.has_auxiliary_loss = use_experts
        ignored = list(self.effective_config["ignored_config_parameters"])
        if backbone == "linear":
            ignored += ["d_model", "e_layers", "n_heads", "d_ff"]
        elif backbone in {"lstm", "gru"}:
            ignored += ["n_heads", "d_ff"]
        if not use_experts:
            ignored += ["dropout", "tsb_expert_kernels", "tsb_expert_hidden", "tsb_n_experts", "tsb_expert_type",
                        "tsb_aux_weight", "tsb_base_weight"]
        if not use_decomp:
            ignored += ["tsb_decomp_kernels", "tsb_enrich", "tsb_decomp_fixed"]
        elif not use_experts:
            ignored += ["tsb_enrich"]
        host_config = {"type": backbone}
        if backbone in {"lstm", "gru"}:
            host_config.update(hidden=configs.d_model, layers=configs.e_layers, dropout=RNN_DROPOUT)
        elif backbone == "transformer":
            host_config.update(d_model=configs.d_model, layers=configs.e_layers, n_heads=configs.n_heads,
                               d_ff=configs.d_ff, dropout=TRANSFORMER_DROPOUT, positional_encoding=False)
        self.effective_config.update({
            "architecture": "TS-Booster", "variant": variant, "host": host_config,
            "use_decomp": use_decomp, "use_experts": use_experts, "gate": self.booster.gate,
            "const_gate_value": CONST_GATE_VALUE if self.booster.gate == "const" else None,
            "decomp_kernels": decomp_kernels if use_decomp else None,
            "decomp": ("softmax mixture of zero-padded centered moving averages, "
                       + ("fixed equal weights" if decomp_fixed else "learnable weights")) if use_decomp else None,
            "enrich": enrich if use_decomp and use_experts else None,
            "expert_input": ({"trend": "[x || trend]", "seasonal": "[x || seasonal]",
                              "both": "[x || trend || seasonal]", "raw": "[x || x] (width control)",
                              "zeros": "[x || 0] (width control)"}[enrich]
                             if use_decomp and use_experts else "[x]" if use_experts else None),
            "n_experts": n_experts if use_experts else 0,
            "expert_type": expert_type if use_experts else None,
            "expert_kernels": expert_kernels if use_experts and expert_type == "cnn" else None,
            "expert_hidden_dim": expert_hidden if use_experts else None,
            "expert_branch_dim": (expert_hidden // len(expert_kernels)
                                  if use_experts and expert_type == "cnn" else None),
            "expert_dropout": configs.dropout if use_experts else None,
            "auxiliary_loss": ({"aux_weight": self.aux_weight, "base_weight": self.base_weight,
                                "form": "aux*MSE(y_expert, y - sg(y_base)) + base*MSE(y_base, y)"}
                               if use_experts else None),
            "ignored_config_parameters": sorted(set(ignored)),
        })

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, return_aux=False):
        y_final, aux = self.booster(self._check_input(x_enc))
        prediction = y_final.unsqueeze(-1)
        if not return_aux:
            return prediction
        gate = aux["gate"]
        if gate is not None:
            gate = gate.reshape(-1).expand(x_enc.shape[0]) if gate.numel() == 1 else gate.reshape(-1)
        return prediction, {"y_base": aux["y_base"], "y_expert": aux["y_expert"], "gate": gate}

    def auxiliary_loss(self, aux, target):
        """Residual-expert and host terms added to the training criterion (not to validation)."""
        if aux["y_expert"] is None:
            return None
        y = target[..., 0]
        return (self.aux_weight * F.mse_loss(aux["y_expert"], y - aux["y_base"].detach())
                + self.base_weight * F.mse_loss(aux["y_base"], y))

    def diagnostics(self):
        return self.booster.diagnostics()
