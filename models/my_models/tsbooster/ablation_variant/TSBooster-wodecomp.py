"""Experts without decomposition (paper ablation "+Expert").

The boosting experts read the raw window only; without the decomposition there is
no gate.
"""

from layers.TSBooster_layers import TSBoosterForecaster

SUPPORTED_TASKS = ("long_term_forecast",)


class Model(TSBoosterForecaster):
    def __init__(self, configs):
        super().__init__(configs, use_decomp=False, use_experts=True, gate="none", variant="+Expert")
