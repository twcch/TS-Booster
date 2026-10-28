"""Validation and output mapping shared by model families."""


FORECAST_TASKS = ("long_term_forecast", "short_term_forecast")


def validate_forecast_config(configs, *, fixed_heads=None):
    task = getattr(configs, "task_name", "long_term_forecast")
    if task not in FORECAST_TASKS:
        raise ValueError(f"This model supports {FORECAST_TASKS}, got {task!r}")
    for key in ("seq_len", "pred_len", "enc_in", "d_model"):
        if getattr(configs, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if fixed_heads and configs.d_model % fixed_heads:
        raise ValueError(f"This model uses fixed n_heads={fixed_heads}; d_model must be divisible by {fixed_heads}")


def validate_odd_kernel(kernel):
    if not isinstance(kernel, int) or isinstance(kernel, bool) or kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"moving_avg must be a positive odd integer, got {kernel!r}")
    return kernel


def output_channel_indices(configs):
    """Explicitly support all features or a single target, without broadcasting."""
    enc_in, c_out = configs.enc_in, configs.c_out
    if c_out == enc_in:
        return None
    if c_out != 1:
        raise ValueError(f"Output requires c_out=enc_in ({enc_in}) or c_out=1, got {c_out}")
    return (single_target_index(configs),)


def single_target_index(configs):
    """Resolve the one forecast target among enc_in inputs (MS puts it last)."""
    enc_in = configs.enc_in
    index = getattr(configs, "target_channel_index", -1)
    if not isinstance(index, int) or isinstance(index, bool) or not -enc_in <= index < enc_in:
        raise ValueError(f"target_channel_index must identify one of {enc_in} input features")
    return index % enc_in


def validate_decoder_channels(configs, *, extra_features=False):
    valid = configs.dec_in >= configs.enc_in if extra_features else configs.dec_in == configs.enc_in
    if not valid:
        relation = "at least" if extra_features else "equal to"
        raise ValueError(f"dec_in must be {relation} enc_in ({configs.enc_in}) for this decoder")
