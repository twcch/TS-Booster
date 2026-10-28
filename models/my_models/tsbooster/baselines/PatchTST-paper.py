"""PatchTST as reimplemented in the TS-Booster paper (Table 6); TSLib's is PatchTST.

Each channel is cut into patches (length 5, stride 3), embedded with a learnable
position embedding and encoded by a shared Transformer (3 layers, d_model 128,
8 heads, d_ff 256, dropout 0.1); a flatten head forecasts it. RevIN normalizes the
input, as in the paper's learning-rate study. At seq_len=30, pred_len=3 and two
channels it has 402,823 parameters. The settings are fixed as in the paper.
"""

from layers.TSBooster_baselines import PatchTST
from layers.TSBooster_layers import SingleTargetForecaster

SUPPORTED_TASKS = ("long_term_forecast",)
PATCH_LEN, STRIDE = 5, 3


class Model(SingleTargetForecaster):
    def __init__(self, configs):
        super().__init__(configs)
        if self.seq_len < PATCH_LEN:
            raise ValueError(f"PatchTST-paper needs seq_len >= {PATCH_LEN}")
        self.net = PatchTST(self.seq_len, self.pred_len, self.enc_in, patch_len=PATCH_LEN, stride=STRIDE,
                            use_revin=True)
        ignored = self.effective_config["ignored_config_parameters"]
        self.effective_config.update({
            "architecture": "PatchTST (paper reimplementation)", "patch_len": PATCH_LEN, "stride": STRIDE,
            "n_patches": self.net.n_patches, "d_model": 128, "n_heads": 8, "e_layers": 3, "d_ff": 256,
            "dropout": 0.1, "activation": "gelu", "position_embedding": "learnable", "revin": True,
            "ignored_config_parameters": sorted(set(ignored) | {"d_model", "e_layers", "n_heads", "d_ff",
                                                                "dropout", "patch_len", "use_norm"}),
        })

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.net(self._check_input(x_enc)).unsqueeze(-1)
