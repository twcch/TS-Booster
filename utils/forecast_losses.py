"""Forecast training criteria selected with --loss.

HYBRID is TS-Booster's objective: (1 - w) * Huber + w * a soft-margin direction
term on steps whose true change from the last observed target exceeds a threshold.
Criteria with ``requires_last_value`` receive that observation as a third argument.
"""

import torch
from torch import nn

FORECAST_LOSSES = ("MSE", "MAE", "HUBER", "HYBRID")


class HybridDirectionalLoss(nn.Module):
    requires_last_value = True

    def __init__(self, delta=1.0, dir_weight=0.5, threshold=0.01):
        super().__init__()
        if delta <= 0 or not 0 <= dir_weight <= 1 or threshold < 0:
            raise ValueError("HYBRID needs delta > 0, 0 <= dir_weight <= 1 and threshold >= 0")
        self.huber = nn.HuberLoss(delta=delta, reduction="mean")
        self.dir_weight = dir_weight
        self.threshold = threshold

    def forward(self, pred, target, last_value):
        loss = self.huber(pred, target)
        true_delta = target - last_value
        mask = torch.abs(true_delta) > self.threshold
        if mask.any():
            direction = nn.functional.soft_margin_loss((pred - last_value)[mask], torch.sign(true_delta[mask]))
        else:
            direction = torch.zeros((), device=pred.device, dtype=pred.dtype)
        return (1 - self.dir_weight) * loss + self.dir_weight * direction


def build_forecast_criterion(args):
    name = str(getattr(args, "loss", "MSE")).upper()
    if name == "MSE":
        return nn.MSELoss()
    if name == "MAE":
        return nn.L1Loss()
    if name == "HUBER":
        return nn.HuberLoss(delta=getattr(args, "huber_delta", 1.0))
    if name == "HYBRID":
        return HybridDirectionalLoss(getattr(args, "huber_delta", 1.0), getattr(args, "dir_weight", 0.5),
                                     getattr(args, "dir_threshold", 0.01))
    raise ValueError(f"Unsupported forecast loss {name!r}; choose from {FORECAST_LOSSES}")


def apply_criterion(criterion, prediction, target, last_value):
    if getattr(criterion, "requires_last_value", False):
        return criterion(prediction, target, last_value)
    return criterion(prediction, target)
