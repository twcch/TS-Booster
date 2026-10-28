"""Forecast metrics. Percent errors are ratios; zero denominators are undefined.

The six-value API retains pooled R2 over all elements for historical comparison.
Use metric_report for explicitly separated channel and horizon results.
"""
import numpy as np

METRIC_NAMES = ('mae', 'mse', 'rmse', 'mape', 'mspe', 'r2')
METRIC_POLICY_VERSION = '2:zero-denominator-nan;pooled-r2'


def _arrays(pred, true):
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError("pred and true must have the same nonempty shape")
    return pred, true


def RSE(pred, true):
    pred, true = _arrays(pred, true)
    denom = np.sum((true - true.mean()) ** 2)
    return float(np.sqrt(np.sum((true - pred) ** 2) / denom)) if denom > 0 else np.nan


def CORR(pred, true):
    pred, true = _arrays(pred, true)
    centered_true, centered_pred = true - true.mean(axis=0), pred - pred.mean(axis=0)
    numerator = (centered_true * centered_pred).sum(axis=0)
    denominator = np.sqrt((centered_true ** 2).sum(axis=0) * (centered_pred ** 2).sum(axis=0))
    result = np.full(np.shape(numerator), np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result.mean(axis=-1) if result.ndim else float(result)


def MAE(pred, true):
    pred, true = _arrays(pred, true)
    return float(np.mean(np.abs(true - pred)))


def MSE(pred, true):
    pred, true = _arrays(pred, true)
    return float(np.mean((true - pred) ** 2))


def RMSE(pred, true):
    return float(np.sqrt(MSE(pred, true)))


def _percentage_errors(pred, true):
    pred, true = _arrays(pred, true)
    errors = np.full_like(true, np.nan)
    np.divide(true - pred, true, out=errors, where=true != 0)
    return errors


def MAPE(pred, true):
    return float(np.mean(np.abs(_percentage_errors(pred, true))))


def MSPE(pred, true):
    return float(np.mean(np.square(_percentage_errors(pred, true))))


def R2(pred, true):
    """Pooled R2; constant truth is undefined (NaN), without denominator clipping."""
    pred, true = _arrays(pred, true)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def metric(pred, true):
    pred, true = _arrays(pred, true)
    return MAE(pred, true), MSE(pred, true), RMSE(pred, true), MAPE(pred, true), MSPE(pred, true), R2(pred, true)


def _broadcast_anchor(anchor, pred):
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape == pred.shape:
        return anchor
    if pred.ndim == 3 and anchor.shape == (pred.shape[0], pred.shape[2]):
        return anchor[:, None, :]
    if pred.ndim == 2 and anchor.shape == (pred.shape[0],):
        return anchor[:, None]
    raise ValueError(f"anchor shape {anchor.shape} does not match forecasts {pred.shape}")


def MDA(pred, true, anchor):
    """Mean directional accuracy against the last observed value, pooled over all elements.

    ``anchor`` is the last input value of each sample/channel and applies to every
    horizon step; a step counts when sign(pred - anchor) == sign(true - anchor),
    exact zeros included, with no threshold.
    """
    pred, true = _arrays(pred, true)
    anchor = _broadcast_anchor(anchor, pred)
    return float(np.mean(np.sign(pred - anchor) == np.sign(true - anchor)))


def _scores(pred, true, anchor):
    values = dict(zip(METRIC_NAMES, metric(pred, true)))
    if anchor is not None:
        values['mda'] = MDA(pred, true, anchor)
    return values


def metric_report(pred, true, anchor=None):
    """JSON-ready metrics with explicit aggregation and nonfinite status.

    Input with shape [sample,horizon,channel] adds metrics per channel/horizon.
    No element with an undefined denominator is silently omitted. An ``anchor``
    ([sample, channel]) adds the directional accuracy (mda).
    """
    pred, true = _arrays(pred, true)
    overall = _scores(pred, true, anchor)
    report = {'policy_version': METRIC_POLICY_VERSION, 'aggregation': 'pooled',
              'overall': overall, 'nonfinite_metrics': [k for k, v in overall.items() if not np.isfinite(v)],
              'zero_target_count': int(np.count_nonzero(true == 0)),
              'nonfinite_input_count': int(np.count_nonzero(~np.isfinite(pred)) + np.count_nonzero(~np.isfinite(true)))}
    if pred.ndim == 3:
        anchor = None if anchor is None else _broadcast_anchor(anchor, pred)[:, 0, :]
        report['per_channel'] = [_scores(pred[:, :, c], true[:, :, c], None if anchor is None else anchor[:, c])
                                 for c in range(pred.shape[2])]
        report['per_horizon'] = [_scores(pred[:, h, :], true[:, h, :], anchor) for h in range(pred.shape[1])]
    return report
