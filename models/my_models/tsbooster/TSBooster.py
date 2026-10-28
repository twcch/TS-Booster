"""TS-Booster (paper model).

A host forecast is boosted by K CNN experts that read [x || trend(x)], where the
trend is a learnable mixture of moving averages; an SNR gate scales the experts:
y = y_base + sigmoid(w * std(seasonal) / std(x) + b) * sum_k expert_k.
Training adds the residual-expert and host terms of auxiliary_loss to --loss.
"""

from layers.TSBooster_layers import TSBoosterForecaster

SUPPORTED_TASKS = ("long_term_forecast",)


class Model(TSBoosterForecaster):
    def __init__(self, configs):
        super().__init__(configs, use_decomp=True, use_experts=True, gate="snr", variant="TS-Booster")
