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
        provider = require_provider("transformers")
        model_id = "Maple728/TimeMoE-50M"
        self.model = provider.AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=self.runtime.dtype,
            **self.runtime.hub_kwargs()).to(self.runtime.device)
        self.task_name, self.seq_len, self.pred_len = configs.task_name, configs.seq_len, configs.pred_len
        self.effective_config = self.runtime.metadata(model_id)
        self.effective_config.update(external_normalization="mean_variance_eps_1e-5", trust_remote_code=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        context, mean, stdev = normalize_context(x_enc)
        batch, _, channels = context.shape
        result = self.model.generate(flatten_channels(context).to(device=self.runtime.device, dtype=self.runtime.dtype),
                                     max_new_tokens=self.pred_len)
        points = result[:, -self.pred_len:].to(x_enc)
        return restore_channels(points, batch, channels, self.pred_len) * stdev + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
