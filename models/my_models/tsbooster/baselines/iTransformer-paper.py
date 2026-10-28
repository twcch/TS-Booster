"""iTransformer as reimplemented in the TS-Booster paper (Table 6); TSLib's is iTransformer.

Each variate's window is one token (linear embedding plus a learnable variate
embedding); a Transformer (2 layers, d_model 128, 4 heads, d_ff 256, dropout 0.1)
attends across variates and a linear head forecasts the target token. No input
normalization. At seq_len=30, pred_len=3 and two channels it has 269,571
parameters. The settings are fixed as in the paper.
"""

from layers.TSBooster_baselines import iTransformer
from layers.TSBooster_layers import SingleTargetForecaster

SUPPORTED_TASKS = ("long_term_forecast",)


class Model(SingleTargetForecaster):
    def __init__(self, configs):
        super().__init__(configs)
        self.net = iTransformer(self.seq_len, self.pred_len, self.enc_in)
        ignored = self.effective_config["ignored_config_parameters"]
        self.effective_config.update({
            "architecture": "iTransformer (paper reimplementation)", "d_model": 128, "n_heads": 4,
            "e_layers": 2, "d_ff": 256, "dropout": 0.1, "activation": "gelu",
            "variate_embedding": "learnable", "normalization": None, "time_mark_tokens": False,
            "ignored_config_parameters": sorted(set(ignored) | {"d_model", "e_layers", "n_heads", "d_ff",
                                                                "dropout", "use_norm"}),
        })

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.net(self._check_input(x_enc)).unsqueeze(-1)
