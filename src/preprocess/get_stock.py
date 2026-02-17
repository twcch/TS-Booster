"""Fetch Taiwan stock price data from Yahoo Finance via yfinance."""
import os

import pandas as pd
import yfinance as yf

TARGETS = ["0050", "2317", "2330", "2454", "2603", "2881"]


def fetch_stock(ticker_id):
    yahoo_ticker = f"{ticker_id}.TW"
    print(f"\n=== 抓取 {ticker_id} ({yahoo_ticker}) ===")

    try:
        ticker = yf.Ticker(yahoo_ticker)
        data = ticker.history(start="2000-01-01", auto_adjust=True)

        if data.empty:
            print(f"  無資料")
            return None

        df = pd.DataFrame({
            'Date': data.index.strftime('%Y-%m-%d'),
            'Close': data['Close'].values,
        }).reset_index(drop=True)

        filename = f"dataset/{ticker_id}_TW.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"  {len(df)} 筆 | {df['Date'].iloc[0]} ~ {df['Date'].iloc[-1]} | 存至 {filename}")

        return df

    except Exception as e:
        print(f"  錯誤: {e}")
        return None


if __name__ == "__main__":
    os.makedirs('dataset', exist_ok=True)
    results = {}
    for t in TARGETS:
        results[t] = fetch_stock(t)

    print("\n=== 結果彙整 ===")
    for t in TARGETS:
        status = f"{len(results[t])} 筆" if results[t] is not None else "失敗"
        print(f"  {t}_TW: {status}")
