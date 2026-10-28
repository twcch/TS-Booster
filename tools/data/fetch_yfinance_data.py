"""Fetch an explicit OHLC dataset and provenance manifest without import side effects."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


def fetch(ticker, start, end, output, *, auto_adjust=False, overwrite=False):
    output = Path(output)
    manifest_path = output.with_suffix(output.suffix + '.manifest.json')
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(f"Output exists: {output}; pass --overwrite deliberately")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("start must be before end (end is exclusive)")
    import yfinance as yf
    frame = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust,
                        actions=False, progress=False)
    if frame.empty:
        raise ValueError(f"No data returned for {ticker!r}")
    if isinstance(frame.columns, pd.MultiIndex):
        if 'Ticker' not in frame.columns.names:
            raise ValueError("Unexpected yfinance column schema")
        frame.columns = frame.columns.droplevel('Ticker')
    frame.columns.name = None
    frame = frame.reset_index().rename(columns={'Date': 'date', 'Datetime': 'date'})
    missing = set(['date', 'Open', 'High', 'Low', 'Close']) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing yfinance columns: {sorted(missing)}")
    frame = frame[['date', 'Open', 'High', 'Low', 'Close']]
    # Use the same boundary validation as forecasting; no silent row repair.
    from data_provider.preparation import validate_forecast_frame
    frame, _, _ = validate_forecast_frame(frame, 'Close', 'MS')
    csv = frame.to_csv(index=False).encode('utf-8')
    dates = pd.DatetimeIndex(frame.date)
    metadata = {'source': 'yfinance', 'source_version': yf.__version__, 'ticker': ticker,
                'requested_start': start, 'requested_end_exclusive': end,
                'auto_adjust': bool(auto_adjust), 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'actual_start': dates.min().isoformat(), 'actual_end': dates.max().isoformat(),
                'rows': len(frame), 'columns': list(frame),
                'weekend_observations': int((dates.dayofweek >= 5).sum()),
                'calendar': 'daily_including_weekends' if (dates.dayofweek >= 5).any() else 'trading_observations',
                'sha256': hashlib.sha256(csv).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = 'wb' if overwrite else 'xb'
    with output.open(mode) as handle:
        handle.write(csv)
    with manifest_path.open('w' if overwrite else 'x', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return output, manifest_path


def main():
    parser = argparse.ArgumentParser(description='下載 OHLC 資料與來源 manifest；預設拒絕覆寫')
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--start', required=True, help='含起始日，YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='不含結束日，YYYY-MM-DD')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--auto-adjust', action=argparse.BooleanOptionalAction, default=False,
                        help='明確控制除權息價格調整；預設 False')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    paths = fetch(args.ticker, args.start, args.end, args.output,
                  auto_adjust=args.auto_adjust, overwrite=args.overwrite)
    print('\n'.join(str(p.resolve()) for p in paths))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
