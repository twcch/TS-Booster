import torch
from torch import nn
from layers.foundation_adapter import FoundationRuntime, require_provider, require_zero_shot


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        require_zero_shot(configs)
        self.runtime = FoundationRuntime.from_config(configs)
        provider = require_provider("transformers")
        model_id = "thuml/sundial-base-128m"
        self.model = provider.AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=self.runtime.dtype,
            **self.runtime.hub_kwargs()).to(self.runtime.device)
        self.task_name, self.seq_len, self.pred_len = configs.task_name, configs.seq_len, configs.pred_len
        self.effective_config = self.runtime.metadata(model_id)
        self.effective_config.update(num_samples=20, point_aggregation="sample_mean", trust_remote_code=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for channel in range(x_enc.shape[-1]):
            points = self.model.generate(x_enc[..., channel].to(device=self.runtime.device, dtype=self.runtime.dtype),
                                         max_new_tokens=self.pred_len, num_samples=20)
            outputs.append(points.mean(dim=1).to(x_enc))
        output = torch.stack(outputs, dim=-1)
        if output.shape != (x_enc.shape[0], self.pred_len, x_enc.shape[-1]):
            raise ValueError(f"Unexpected Sundial output shape: {tuple(output.shape)}")
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
