import torch
from torch import nn
from layers.foundation_adapter import FoundationRuntime, require_provider, require_zero_shot


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        require_zero_shot(configs)
        self.runtime = FoundationRuntime.from_config(configs)
        provider = require_provider("chronos")
        model_id = "amazon/chronos-bolt-base"
        self.model = provider.BaseChronosPipeline.from_pretrained(
            model_id, device_map=str(self.runtime.device), torch_dtype=self.runtime.dtype,
            **self.runtime.hub_kwargs())
        self.task_name, self.seq_len, self.pred_len = configs.task_name, configs.seq_len, configs.pred_len
        self.effective_config = self.runtime.metadata(model_id)
        self.effective_config["point_aggregation"] = "mean_over_provider_quantiles"

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Preserve separate provider calls and their aggregation for historical runs.
        outputs = []
        for channel in range(x_enc.shape[-1]):
            result = self.model.predict(x_enc[..., channel], prediction_length=self.pred_len)
            outputs.append(result.mean(dim=1).to(x_enc))
        output = torch.stack(outputs, dim=-1)
        if output.shape != (x_enc.shape[0], self.pred_len, x_enc.shape[-1]):
            raise ValueError(f"Unexpected Chronos output shape: {tuple(output.shape)}")
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
