"""Download the 16 TS-Booster close-price series (port of the journal crawlers).

FX crosses come from Investing.com through investpy, Taiwan stocks from Yahoo
Finance (yfinance, auto-adjusted), both from 2000-01-01, as in
docs/TS-Booster_ori/src/preprocess. Each CSV (date, Close) gets a provenance
manifest; existing files are kept unless --overwrite is given. Yahoo's Taiwan
series break the daily price limit, so the experiments replace them with
adjust_tw_prices.py and repair two FX series with repair_fx.py:

    python tools/data/fetch_tsbooster_data.py --output dataset/raw
    python tools/data/repair_fx.py --input dataset/raw --output dataset/raw --overwrite
    python tools/data/adjust_tw_prices.py --output dataset/raw --overwrite
    python tools/data/build_volatility_dataset.py --input dataset/raw --output dataset/volatility
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

FX_TWD = ["USD", "EUR", "JPY", "CAD", "CNY", "KRW", "THB"]
FX_INTL = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY")]
TW_STOCKS = ["0050", "2317", "2330", "2454", "2603", "2881"]
START = "2000-01-01"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def instruments():
    """(name, source, symbol) for every series used in the paper."""
    rows = [(f"{base}_TWD", "investpy", f"{base}/TWD") for base in FX_TWD]
    rows += [(f"{base}_{quote}", "investpy", f"{base}/{quote}") for base, quote in FX_INTL]
    rows += [(f"{code}_TW", "yfinance", f"{code}.TW") for code in TW_STOCKS]
    return rows


@contextmanager
def browser_headers():
    """Investing.com rejects the default requests user agent (HTTP 403)."""
    import requests
    original_get, original_post = requests.get, requests.post

    def patched(method):
        def wrapper(*args, **kwargs):
            kwargs["headers"] = {**kwargs.get("headers", {}), **BROWSER_HEADERS}
            return method(*args, **kwargs)
        return wrapper

    requests.get, requests.post = patched(original_get), patched(original_post)
    try:
        yield
    finally:
        requests.get, requests.post = original_get, original_post


def download_investpy(symbol, start, end):
    import investpy
    with browser_headers():
        data = investpy.get_currency_cross_historical_data(
            currency_cross=symbol, from_date=pd.Timestamp(start).strftime("%d/%m/%Y"),
            to_date=pd.Timestamp(end).strftime("%d/%m/%Y"))
    return pd.DataFrame({"date": data.index, "Close": data["Close"].to_numpy()}), getattr(investpy, "__version__", None)


def download_yfinance(symbol, start, end):
    import yfinance as yf
    data = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    return pd.DataFrame({"date": data.index.tz_localize(None) if data.index.tz else data.index,
                         "Close": data["Close"].to_numpy()}), yf.__version__


def fetch(name, source, symbol, output_dir, *, start=START, end=None, overwrite=False):
    output_dir = Path(output_dir)
    path = output_dir / f"{name}.csv"
    manifest_path = path.with_suffix(".csv.manifest.json")
    if not overwrite and (path.exists() or manifest_path.exists()):
        raise FileExistsError(f"{path} exists; pass --overwrite deliberately")
    end = end or datetime.now().strftime("%Y-%m-%d")
    downloader = {"investpy": download_investpy, "yfinance": download_yfinance}[source]
    frame, version = downloader(symbol, start, end)
    if frame.empty:
        raise ValueError(f"No data returned for {symbol!r}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame = frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    csv = frame.to_csv(index=False).encode("utf-8")
    metadata = {"name": name, "source": source, "source_version": version, "symbol": symbol,
                "auto_adjust": source == "yfinance", "requested_start": start, "requested_end": end,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(frame),
                "actual_start": frame["date"].iloc[0], "actual_end": frame["date"].iloc[-1],
                "columns": list(frame), "sha256": hashlib.sha256(csv).hexdigest()}
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv)
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--names", nargs="+", help="subset, e.g. USD_TWD 2330_TW (default: all 16)")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=None, help="exclusive end date, YYYY-MM-DD (default: today)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    selected = [row for row in instruments() if not args.names or row[0] in args.names]
    unknown = set(args.names or []) - {row[0] for row in selected}
    if unknown:
        parser.error(f"Unknown names: {sorted(unknown)}")
    failures = []
    for name, source, symbol in selected:
        try:
            path, metadata = fetch(name, source, symbol, args.output, start=args.start, end=args.end,
                                   overwrite=args.overwrite)
            print(f"{name:8s} {metadata['rows']:6d} rows {metadata['actual_start']} .. {metadata['actual_end']} -> {path}")
        except Exception as error:  # report every instrument, then fail
            failures.append(name)
            print(f"{name:8s} FAILED: {type(error).__name__}: {error}", file=sys.stderr)
    if failures:
        print(f"Failed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
