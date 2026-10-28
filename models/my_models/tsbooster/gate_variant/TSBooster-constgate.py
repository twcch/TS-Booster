"""TS-Booster with a constant expert gate of 0.5 (gate ablation "Const")."""

from layers.TSBooster_layers import TSBoosterForecaster

SUPPORTED_TASKS = ("long_term_forecast",)


class Model(TSBoosterForecaster):
    def __init__(self, configs):
        super().__init__(configs, use_decomp=True, use_experts=True, gate="const", variant="Const gate")
