"""TS-Booster with the expert gate fixed at 1 (gate ablation "No gate")."""

from layers.TSBooster_layers import TSBoosterForecaster

SUPPORTED_TASKS = ("long_term_forecast",)


class Model(TSBoosterForecaster):
    def __init__(self, configs):
        super().__init__(configs, use_decomp=True, use_experts=True, gate="none", variant="No gate")
