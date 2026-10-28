import torch
from torch import nn
from layers.foundation_adapter import FoundationRuntime, require_provider, require_zero_shot
from layers.channel_adapter import flatten_channels, restore_channels
from layers.foundation_adapter import normalize_context


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        require_zero_shot(configs)
        self.runtime = FoundationRuntime.from_config(configs)
        provider = require_provider("tirex")
        model_id = "NX-AI/TiRex"
        self.model = provider.load_model(model_id, device=str(self.runtime.device),
                                         backend="torch", hf_kwargs=self.runtime.hub_kwargs())
        self.model.to(dtype=self.runtime.dtype)
        self.task_name, self.seq_len, self.pred_len = configs.task_name, configs.seq_len, configs.pred_len
        self.effective_config = self.runtime.metadata(model_id)
        self.effective_config.update(backend="torch", external_normalization="mean_variance_eps_1e-5")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        context, mean, stdev = normalize_context(x_enc)
        batch, _, channels = context.shape
        _, points = self.model.forecast(
            flatten_channels(context).to(device=self.runtime.device, dtype=self.runtime.dtype),
            prediction_length=self.pred_len)
        points = torch.as_tensor(points, device=x_enc.device, dtype=x_enc.dtype)
        return restore_channels(points, batch, channels, self.pred_len) * stdev + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
