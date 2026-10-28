"""Shared validation and preparation; split policies remain explicit at callers."""

import numpy as np
import pandas as pd

from utils.timefeatures import time_features, time_features_from_frequency_str


def validate_window(size):
    if len(size) != 3 or any(not isinstance(v, (int, np.integer)) for v in size):
        raise ValueError("size must contain integer seq_len, label_len, pred_len")
    seq_len, label_len, pred_len = size
    if seq_len <= 0 or pred_len <= 0 or not 0 <= label_len <= seq_len:
        raise ValueError("require seq_len > 0, pred_len > 0, 0 <= label_len <= seq_len")


def read_forecast_csv(path):
    """Read floats exactly as written; pandas' default parser can be one ulp off."""
    return pd.read_csv(path, float_precision="round_trip")


def validate_forecast_frame(frame, target, features):
    if features not in {'M', 'MS', 'S'}:
        raise ValueError("features must be M, MS, or S")
    if frame.columns.duplicated().any() or 'date' not in frame or target not in frame:
        raise ValueError(f"data requires unique columns, date, and target {target!r}")
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='raise')
    dates = frame['date']
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("date must be finite, strictly increasing, and unique")
    columns = [c for c in frame if c not in {'date', target}] + [target]
    if features == 'S':
        columns = [target]
    try:
        values = frame[columns].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("forecast features must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("forecast features must be finite; imputation must be explicit")
    return frame, columns, values


def calendar_features(dates, timeenc, freq, minute=False):
    dates = pd.DatetimeIndex(dates)
    if timeenc == 1:
        return time_features(dates, freq=freq).T
    if timeenc != 0:
        raise ValueError("timeenc must be 0 or 1")
    columns = [dates.month, dates.day, dates.dayofweek, dates.hour]
    if minute or len(time_features_from_frequency_str(freq)) >= 5:
        columns.append(dates.minute // 15)
    return np.column_stack(columns)


def split_anomaly_data(train_data, test_data, win_size, val_ratio=0.2):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    train_data = np.asarray(train_data, dtype=np.float64)
    test_data = np.asarray(test_data, dtype=np.float64)
    if train_data.ndim != 2 or test_data.ndim != 2 or train_data.shape[1] != test_data.shape[1]:
        raise ValueError("anomaly train/test must be 2D with matching feature counts")
    if not np.isfinite(train_data).all() or not np.isfinite(test_data).all():
        raise ValueError("anomaly features must be finite; imputation must be explicit")
    boundary = int(len(train_data) * (1 - val_ratio))
    if min(boundary, len(train_data) - boundary, len(test_data)) < win_size:
        raise ValueError("anomaly train/val/test each require at least win_size observations")
    return train_data[:boundary], train_data[boundary:], test_data
