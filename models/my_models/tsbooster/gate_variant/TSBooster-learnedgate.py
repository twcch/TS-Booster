"""TS-Booster with one learned scalar gate sigmoid(p), p initialized to 1 (gate ablation "Learned")."""

from layers.TSBooster_layers import TSBoosterForecaster

SUPPORTED_TASKS = ("long_term_forecast",)


class Model(TSBoosterForecaster):
    def __init__(self, configs):
        super().__init__(configs, use_decomp=True, use_experts=True, gate="learned", variant="Learned gate")
