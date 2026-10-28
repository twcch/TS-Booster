"""Vanilla host: the TS-Booster backbone alone (paper ablation "Vanilla")."""

from layers.TSBooster_layers import TSBoosterForecaster

SUPPORTED_TASKS = ("long_term_forecast",)


class Model(TSBoosterForecaster):
    def __init__(self, configs):
        super().__init__(configs, use_decomp=False, use_experts=False, gate="none", variant="Vanilla")
