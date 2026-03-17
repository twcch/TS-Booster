"""
Data loading pipeline — shared between thesis and conference versions.
Loads CSV → log_return + volatility → StandardScaler → sliding windows.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class VolatilityDataset(Dataset):
    def __init__(self, X, y, returns):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.r = torch.FloatTensor(returns)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "raw_input": self.X[idx],
            "target": self.y[idx],
            "return": self.r[idx],
        }


def prepare_data(df, lookback=30, horizon=3, batch_size=32):
    """
    Prepare train/val/test DataLoaders from a CSV with Date + Close columns.

    Features: [Volatility, log_return]
    Target: future volatility (7-day rolling, annualized)
    Split: 70% / 10% / 20% by time
    """
    df = df.copy()
    vol_window = 7

    # Log return
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return"] = df["log_return"].fillna(0.0)

    # Volatility (prediction target)
    df["Volatility"] = df["log_return"].rolling(vol_window).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["Volatility", "log_return"]
    features = df[feature_cols].values

    # Train/val/test split (70/10/20)
    n_samples = len(features)
    split_1 = int(n_samples * 0.70)
    split_2 = int(n_samples * 0.80)

    # StandardScaler fit on train only
    scaler = StandardScaler()
    scaler.fit(features[:split_1])
    features_scaled = scaler.transform(features)

    raw_returns = df["log_return"].values

    train_feat = features_scaled[:split_1]
    val_feat = features_scaled[split_1 - lookback : split_2]
    test_feat = features_scaled[split_2 - lookback :]

    train_ret = raw_returns[:split_1]
    val_ret = raw_returns[split_1 - lookback : split_2]
    test_ret = raw_returns[split_2 - lookback :]

    def create_sequences(data, raw_ret, lookback, horizon):
        X, y, y_ret = [], [], []
        if len(data) < lookback + horizon:
            return np.array([]), np.array([]), np.array([])
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i : i + lookback])
            y.append(data[i + lookback : i + lookback + horizon, 0])
            y_ret.append(raw_ret[i + lookback : i + lookback + horizon])
        return np.array(X), np.array(y), np.array(y_ret)

    X_train, y_train, r_train = create_sequences(train_feat, train_ret, lookback, horizon)
    X_val, y_val, r_val = create_sequences(val_feat, val_ret, lookback, horizon)
    X_test, y_test, r_test = create_sequences(test_feat, test_ret, lookback, horizon)

    train_loader = DataLoader(
        VolatilityDataset(X_train, y_train, r_train),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        VolatilityDataset(X_val, y_val, r_val),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )
    test_loader = DataLoader(
        VolatilityDataset(X_test, y_test, r_test),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )

    print(f"[Data] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return train_loader, val_loader, test_loader, scaler
