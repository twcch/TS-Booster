import torch
from torch import nn
from layers.foundation_adapter import FoundationRuntime, require_provider, require_zero_shot
from layers.channel_adapter import flatten_channels, restore_channels
from layers.foundation_adapter import normalize_context, timesfm_compatible_class


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        require_zero_shot(configs)
        self.runtime = FoundationRuntime.from_config(configs, fp32_only=True)
        provider = require_provider("timesfm")
        model_id = "google/timesfm-2.5-200m-pytorch"
        provider_class = timesfm_compatible_class(provider.TimesFM_2p5_200M_torch)
        self.model = provider_class.from_pretrained(model_id, **self.runtime.hub_kwargs())
        # TimesFM manages its internal module rather than an nn.Module wrapper.
        module = getattr(self.model, "model", None)
        if module is None or not hasattr(module, "to"):
            raise RuntimeError("Unsupported TimesFM API: missing internal model runtime adapter")
        module.to(device=self.runtime.device, dtype=self.runtime.dtype)
        module.device = self.runtime.device
        module.device_count = 1
        self.model.compile(provider.ForecastConfig(
            max_context=configs.seq_len, max_horizon=configs.pred_len,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        self.task_name, self.seq_len, self.pred_len = configs.task_name, configs.seq_len, configs.pred_len
        self.effective_config = self.runtime.metadata(model_id)
        self.effective_config.update(external_normalization="mean_variance_eps_1e-5", provider_normalize_inputs=True,
                                     point_aggregation="provider_point_forecast")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        context, mean, stdev = normalize_context(x_enc)
        batch, _, channels = context.shape
        points, _ = self.model.forecast(horizon=self.pred_len,
            inputs=flatten_channels(context).detach().float().cpu().numpy())
        points = torch.as_tensor(points, device=x_enc.device, dtype=x_enc.dtype)
        return restore_channels(points, batch, channels, self.pred_len) * stdev + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
