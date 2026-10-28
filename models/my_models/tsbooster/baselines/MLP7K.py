"""Parameter-matched control for TS-Booster: an MLP with hidden widths 64 and 48.

At seq_len=30, pred_len=3 and two channels it has 7,171 parameters, close to
TS-Booster with the Linear host (7,156). The widths are fixed as in the paper.
"""

from layers.TSBooster_layers import MLP7K, SingleTargetForecaster

SUPPORTED_TASKS = ("long_term_forecast",)


class Model(SingleTargetForecaster):
    def __init__(self, configs):
        super().__init__(configs)
        self.mlp = MLP7K(self.seq_len, self.enc_in, self.pred_len)
        ignored = self.effective_config["ignored_config_parameters"]
        self.effective_config.update({
            "architecture": "MLP-7K", "hidden": [64, 48], "activation": "ReLU",
            "ignored_config_parameters": sorted(set(ignored) | {"d_model", "e_layers", "n_heads", "d_ff",
                                                                "dropout"}),
        })

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.mlp(self._check_input(x_enc)).unsqueeze(-1)
