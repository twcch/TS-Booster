"""DLinear as reimplemented in the TS-Booster paper (Table 6); TSLib's is DLinear.

A zero-padded moving average (kernel 25) splits each channel into trend and
remainder, and per-channel linear maps forecast both. At seq_len=30, pred_len=3
and two channels it has 372 parameters. The settings are fixed as in the paper.
"""

from layers.TSBooster_baselines import DLinear
from layers.TSBooster_layers import SingleTargetForecaster

SUPPORTED_TASKS = ("long_term_forecast",)


class Model(SingleTargetForecaster):
    def __init__(self, configs):
        super().__init__(configs)
        self.net = DLinear(self.seq_len, self.pred_len, self.enc_in)
        ignored = self.effective_config["ignored_config_parameters"]
        self.effective_config.update({
            "architecture": "DLinear (paper reimplementation)", "moving_avg_kernel": 25,
            "moving_avg_padding": "zeros", "individual": True,
            "ignored_config_parameters": sorted(set(ignored) | {"d_model", "e_layers", "n_heads", "d_ff",
                                                                "dropout", "individual"}),
        })

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.net(self._check_input(x_enc)).unsqueeze(-1)
