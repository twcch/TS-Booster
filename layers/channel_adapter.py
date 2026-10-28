"""Shape adapters for providers that forecast independent univariate series."""

import torch


def flatten_channels(x: torch.Tensor) -> torch.Tensor:
    """[batch, time, channel] -> [batch * channel, time], batch then channel."""
    if x.ndim != 3 or min(x.shape) <= 0:
        raise ValueError(f"Expected nonempty [batch, time, channel], got {tuple(x.shape)}")
    batch, length, channels = x.shape
    return x.permute(0, 2, 1).reshape(batch * channels, length)


def restore_channels(output: torch.Tensor, batch: int, channels: int,
                     prediction_length: int | None = None) -> torch.Tensor:
    """[batch * channel, horizon] -> [batch, horizon, channel]."""
    if output.ndim != 2 or output.shape[0] != batch * channels:
        raise ValueError(f"Expected [{batch * channels}, horizon], got {tuple(output.shape)}")
    if prediction_length is not None and output.shape[1] != prediction_length:
        raise ValueError(f"Expected horizon {prediction_length}, got {output.shape[1]}")
    return output.reshape(batch, channels, output.shape[1]).permute(0, 2, 1)
