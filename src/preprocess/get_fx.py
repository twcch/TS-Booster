"""Fetch all FX/TWD currency cross data from Investing.com via investpy."""
import datetime
import os

import investpy
import pandas as pd
import requests

TARGETS_TWD = ["USD", "EUR", "JPY", "CAD", "CNY", "KRW", "THB"]
TARGETS_INTL = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY")]


def _patch_headers():
    """Patch requests to avoid 403 from Investing.com."""
    original_post = requests.post
    original_get = requests.get

    def patched_request(method):
        def wrapper(*args, **kwargs):
            if 'headers' not in kwargs:
                kwargs['headers'] = {}
            kwargs['headers'].update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            })
            return method(*args, **kwargs)
        return wrapper

    requests.post = patched_request(original_post)
    requests.get = patched_request(original_get)
    return original_post, original_get


def fetch_fx(base, quote, filename):
    pair = f"{base}/{quote}"
    print(f"\n=== 抓取 {pair} ===")

    original_post, original_get = _patch_headers()

    try:
        data = investpy.get_currency_cross_historical_data(
            currency_cross=pair,
            from_date="01/01/2000",
            to_date=datetime.datetime.now().strftime("%d/%m/%Y"),
        )

        df = pd.DataFrame({
            'Date': data.index.strftime('%Y-%m-%d'),
            'Close': data['Close']
        }).reset_index(drop=True)

        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"  {len(df)} 筆 | {df['Date'].iloc[0]} ~ {df['Date'].iloc[-1]} | 存至 {filename}")

        return df

    except Exception as e:
        print(f"  錯誤: {e}")
        return None

    finally:
        requests.post = original_post
        requests.get = original_get


if __name__ == "__main__":
    os.makedirs('dataset', exist_ok=True)
    results = {}

    # TWD pairs
    for t in TARGETS_TWD:
        name = f"{t}_TWD"
        results[name] = fetch_fx(t, "TWD", f"dataset/{name}.csv")

    # International pairs
    for base, quote in TARGETS_INTL:
        name = f"{base}_{quote}"
        results[name] = fetch_fx(base, quote, f"dataset/{name}.csv")

    print("\n=== 結果彙整 ===")
    for name, df in results.items():
        status = f"{len(df)} 筆" if df is not None else "失敗"
        print(f"  {name}: {status}")
