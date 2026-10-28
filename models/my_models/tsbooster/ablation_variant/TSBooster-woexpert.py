"""Decomposition without experts (paper ablation "+Decomp").

Two identically initialized copies of the host forecast the trend and seasonal
parts, and their outputs are summed.
"""

from layers.TSBooster_layers import TSBoosterForecaster

SUPPORTED_TASKS = ("long_term_forecast",)


class Model(TSBoosterForecaster):
    def __init__(self, configs):
        super().__init__(configs, use_decomp=True, use_experts=False, gate="none", variant="+Decomp")
