import torch
from torch import nn
from layers.foundation_adapter import FoundationRuntime, require_provider, require_zero_shot


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        require_zero_shot(configs)
        self.runtime = FoundationRuntime.from_config(configs, fp32_only=True)
        provider = require_provider("uni2ts.model.moirai2")
        model_id = "Salesforce/moirai-2.0-R-small"
        self.model = provider.Moirai2Forecast(
            module=provider.Moirai2Module.from_pretrained(model_id, **self.runtime.hub_kwargs()),
            prediction_length=configs.pred_len, context_length=configs.seq_len,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0
        ).to(self.runtime.device)
        self.task_name, self.seq_len, self.pred_len = configs.task_name, configs.seq_len, configs.pred_len
        self.effective_config = self.runtime.metadata(model_id)
        self.effective_config["point_aggregation"] = "mean_over_provider_quantiles"

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for channel in range(x_enc.shape[-1]):
            result = self.model.predict(x_enc[..., channel].detach().float().cpu().numpy())
            points = torch.as_tensor(result, device=x_enc.device, dtype=x_enc.dtype).mean(dim=1)
            # Provider versions may retain the univariate target dimension.
            if points.ndim == 3 and points.shape[-1] == 1:
                points = points.squeeze(-1)
            if points.shape != (x_enc.shape[0], self.pred_len):
                raise ValueError(f"Unexpected Moirai output shape: {tuple(points.shape)}")
            outputs.append(points)
        return torch.stack(outputs, dim=-1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
