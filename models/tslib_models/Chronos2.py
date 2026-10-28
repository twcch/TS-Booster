import torch
from torch import nn
from layers.foundation_adapter import FoundationRuntime, require_provider, require_zero_shot
from layers.foundation_adapter import normalize_context


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        require_zero_shot(configs)
        self.runtime = FoundationRuntime.from_config(configs)
        provider = require_provider("chronos")
        model_id = "amazon/chronos-2"
        self.model = provider.BaseChronosPipeline.from_pretrained(
            model_id, device_map=str(self.runtime.device), torch_dtype=self.runtime.dtype,
            **self.runtime.hub_kwargs())
        self.task_name, self.seq_len, self.pred_len = configs.task_name, configs.seq_len, configs.pred_len
        self.effective_config = self.runtime.metadata(model_id)
        self.effective_config["external_normalization"] = "mean_variance_eps_1e-5"
        self.effective_config["point_aggregation"] = "provider_point_forecast"

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        context, mean, stdev = normalize_context(x_enc)
        # [B,C,L] keeps each input sample as its own multivariate provider group.
        _, points = self.model.predict_quantiles(
            context.permute(0, 2, 1).detach().float().cpu().numpy(),
            prediction_length=self.pred_len, quantile_levels=[0.1, 0.5, 0.9])
        if isinstance(points, (list, tuple)):
            points = torch.stack([torch.as_tensor(point) for point in points])
        output = torch.as_tensor(points, device=x_enc.device, dtype=x_enc.dtype).permute(0, 2, 1)
        if output.shape != (x_enc.shape[0], self.pred_len, x_enc.shape[-1]):
            raise ValueError(f"Unexpected Chronos2 output shape: {tuple(output.shape)}")
        return output * stdev + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
