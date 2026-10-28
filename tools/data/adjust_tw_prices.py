"""Dividend-, capital-reduction- and split-adjusted Taiwan stock closes from TWSE data.

Yahoo Finance's adjusted series for the paper's stocks move beyond the TWSE daily price
limit (unadjusted capital reductions and splits, bad prints), so this tool adjusts the
raw closes itself with the exchange's reference prices, fetched through the FinMind API:

    factor(event) = reference price / last close before the event
    adjusted close_t = close_t * product of the factors of the events after t

Events are ex-rights/ex-dividend results (TaiwanStockDividendResult), capital reductions
(TaiwanStockCapitalReductionReferencePrice) and splits (TaiwanStockSplitPrice). The
exchange's ex-rights records start in July 2003, so a series starts on --start (default
2003-07-01) or at listing. The result is checked against the daily price limit (7% before
2015-06-01, 10% after); any remaining violation stops the build.

    python tools/data/adjust_tw_prices.py --output dataset/raw
    python tools/data/build_volatility_dataset.py --input dataset/raw --output dataset/volatility
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

API = "https://api.finmindtrade.com/api/v4/data"
STOCKS = ["0050", "2317", "2330", "2454", "2603", "2881"]
START, END = "2003-07-01", "2026-02-16"
LIMIT_CHANGE = pd.Timestamp("2015-06-01")  # TWSE daily limit: 7% before, 10% from this date
TOLERANCE = 0.003  # tick rounding of the limit price


def fetch(dataset, code, start, end):
    query = urllib.parse.urlencode({"dataset": dataset, "data_id": code, "start_date": start, "end_date": end})
    with urllib.request.urlopen(f"{API}?{query}", timeout=60) as response:
        payload = json.loads(response.read())
    if payload.get("status") != 200:
        raise RuntimeError(f"{dataset} {code}: {payload.get('msg')}")
    return pd.DataFrame(payload["data"])


def events(dividends, reductions, splits):
    """[(date, factor, kind, inputs)] from the three event tables."""
    rows = []
    for row in dividends.itertuples():
        rows.append((row.date, row.reference_price / row.before_price, f"ex-{row.stock_or_cache_dividend}",
                     {"before_price": row.before_price, "reference_price": row.reference_price}))
    for row in reductions.itertuples():
        rows.append((row.date, row.PostReductionReferencePrice / row.ClosingPriceonTheLastTradingDay,
                     "capital reduction", {"last_close": row.ClosingPriceonTheLastTradingDay,
                                           "reference_price": row.PostReductionReferencePrice}))
    for row in splits.itertuples():
        rows.append((row.date, row.after_price / row.before_price, row.type,
                     {"before_price": row.before_price, "reference_price": row.after_price}))
    return sorted(rows)


def adjust(prices, event_rows):
    """Backward-adjusted closes: each factor scales every close before its event date."""
    close = prices.set_index("date")["close"].astype(float)
    factors = pd.Series(1.0, index=close.index)
    for date, factor, _, inputs in event_rows:
        position = close.index.searchsorted(pd.Timestamp(date))  # first trading day on or after the event
        if position == 0 or position == len(close):
            continue
        previous = close.iloc[position - 1]
        if abs(previous - inputs.get("before_price", inputs.get("last_close"))) > 1e-6 * previous:
            raise ValueError(f"{date}: the event's last close differs from the price series ({previous})")
        factors.iloc[position] *= factor
    multiplier = factors[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    return close * multiplier


def limit_violations(adjusted):
    returns = np.log(adjusted / adjusted.shift(1)).dropna()
    limit = np.where(returns.index < LIMIT_CHANGE, 0.07, 0.10)
    too_high = returns > np.log1p(limit) + TOLERANCE
    too_low = returns < np.log1p(-limit) - TOLERANCE
    return returns[too_high | too_low]


def build(code, output_dir, *, start=START, end=END, overwrite=False):
    path = Path(output_dir) / f"{code}_TW.csv"
    manifest_path = path.with_suffix(".csv.manifest.json")
    if not overwrite and (path.exists() or manifest_path.exists()):
        raise FileExistsError(f"{path} exists; pass --overwrite")
    tables = {}
    for dataset in ("TaiwanStockPrice", "TaiwanStockDividendResult", "TaiwanStockCapitalReductionReferencePrice",
                    "TaiwanStockSplitPrice"):
        tables[dataset] = fetch(dataset, code, "2000-01-01", end)
        time.sleep(1)
    prices = tables["TaiwanStockPrice"].assign(date=lambda frame: pd.to_datetime(frame["date"]))
    prices = prices[prices["close"] > 0].sort_values("date").reset_index(drop=True)
    event_rows = events(*(tables[name] if len(tables[name]) else pd.DataFrame(columns=columns) for name, columns in (
        ("TaiwanStockDividendResult", ["date", "before_price", "reference_price", "stock_or_cache_dividend"]),
        ("TaiwanStockCapitalReductionReferencePrice",
         ["date", "ClosingPriceonTheLastTradingDay", "PostReductionReferencePrice"]),
        ("TaiwanStockSplitPrice", ["date", "before_price", "after_price", "type"]))))
    adjusted = adjust(prices, event_rows)
    adjusted = adjusted[adjusted.index >= start]
    violations = limit_violations(adjusted)
    if len(violations):
        raise ValueError(f"{code}: moves beyond the daily price limit after adjustment: "
                         f"{[(f'{day:%Y-%m-%d}', round(value, 4)) for day, value in violations.items()]}")
    frame = pd.DataFrame({"Date": adjusted.index.strftime("%Y-%m-%d"), "Close": adjusted.to_numpy()})
    csv = frame.to_csv(index=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv)
    returns = np.log(adjusted / adjusted.shift(1)).dropna()
    manifest = {
        "source": "TWSE prices and reference prices via the FinMind API", "api": API, "stock": code,
        "fetched_at": datetime.now(timezone.utc).isoformat(), "requested": {"start": start, "end": end},
        "rows": len(frame), "first_date": frame["Date"].iloc[0], "last_date": frame["Date"].iloc[-1],
        "adjustment": "backward; factor = reference price / last close before each event",
        "events": [{"date": date, "kind": kind, "factor": factor, **inputs}
                   for date, factor, kind, inputs in event_rows if date > frame["Date"].iloc[0]],
        "max_abs_log_return": float(returns.abs().max()), "limit_violations": 0,
        "columns": list(frame), "sha256": hashlib.sha256(csv).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return frame, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--stocks", nargs="+", default=STOCKS)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for code in args.stocks:
        frame, manifest = build(code, args.output, start=args.start, end=args.end, overwrite=args.overwrite)
        print(f"{code}_TW  {manifest['rows']:5d} rows {manifest['first_date']} .. {manifest['last_date']}  "
              f"{len(manifest['events'])} events, max |log return| {manifest['max_abs_log_return']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
