"""Repair two flaws of the investpy FX series with FRED cross rates; the rest is unchanged.

  CNY/TWD  Investing.com has no quotes from 2003-12-12 to 2005-07-27, inside the yuan's
           dollar peg. The gap is filled with USD/TWD (the same investpy data) divided by
           CNY per USD (FRED DEXCHUS) on the USD/TWD trading days.
  THB/TWD  From February to April 2001 the quotes alternate between the right level and one
           about 5% higher, and 2007-05-28 is a one-day spike: there, quotes more than 1.5% off
           the USD/TWD / THB-per-USD (FRED DEXTHUS) cross are replaced by it. From 2007-11-05
           to 2008-02-29 the quotes barely move and sit 3-12% below the FRED and Yahoo crosses,
           which agree with each other; that whole stretch is replaced.

Replacements are scaled by the median quote/cross ratio of the 21 good days around them, so
they join the series without a step, and every change is listed in the manifest.

    python tools/data/repair_fx.py --input dataset/raw --output dataset/raw --overwrite
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
CNY_GAP = ("2003-12-12", "2005-07-27")
# (start, end, mode): "flagged" replaces quotes off the cross, "all" the whole stretch.
THB_EPISODES = [("2001-01-29", "2001-04-30", "flagged"), ("2007-05-28", "2007-05-28", "flagged"),
                ("2007-11-05", "2008-02-29", "all")]
THRESHOLD = 0.015
NEIGHBOURS = 21
RULES = {"CNY_TWD": f"no quotes {CNY_GAP[0]}..{CNY_GAP[1]}: filled with the cross on USD/TWD trading days",
         "THB_TWD": f"episodes {THB_EPISODES}: flagged = quotes more than {THRESHOLD:.1%} off the cross "
                    "(against the 45 days around the episode), all = every quote; replaced by the cross"}


def fred_series(series_id):
    with urllib.request.urlopen(FRED.format(series_id), timeout=60) as response:
        frame = pd.read_csv(io.BytesIO(response.read()))
    frame.columns = ["date", "value"]
    values = pd.to_numeric(frame["value"], errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.to_datetime(frame["date"])).dropna()


def read(path):
    frame = pd.read_csv(path, parse_dates=["Date"], float_precision="round_trip")
    return frame.set_index("Date")["Close"]


def cross_rate(usd_twd, per_usd):
    """TWD per unit of the currency on the USD/TWD dates (FRED's last value on US holidays)."""
    return usd_twd / per_usd.reindex(per_usd.index.union(usd_twd.index)).ffill().reindex(usd_twd.index)


def local_scale(quotes, cross, dates, exclude):
    """Median quote/cross ratio over the good days nearest to the dates."""
    ratio = np.log(quotes / cross.reindex(quotes.index)).dropna().drop(exclude, errors="ignore")
    position = ratio.index.searchsorted(dates.min())
    window = ratio.iloc[max(0, position - NEIGHBOURS // 2):position + NEIGHBOURS // 2 + 1]
    return float(np.exp(window.median()))


def fill_cny_gap(cny, cross):
    gap = cross[CNY_GAP[0]:CNY_GAP[1]]
    if len(cny[CNY_GAP[0]:CNY_GAP[1]]):
        raise ValueError("CNY/TWD already has quotes inside the gap")
    scale = local_scale(cny, cross, gap.index, [])
    filled = gap * scale
    return pd.concat([cny, filled]).sort_index(), [
        {"date": day.strftime("%Y-%m-%d"), "old": None, "new": float(value)} for day, value in filled.items()]


def repair_thb(thb, cross):
    deviation = np.log(thb / cross.reindex(thb.index))
    flagged = []
    for start, end, mode in THB_EPISODES:
        episode = deviation[start:end]
        if mode == "all":
            flagged += list(episode.index)
            continue
        # judged against the median deviation of the 45 days before and after the episode
        before = deviation[pd.Timestamp(start) - pd.Timedelta(days=45):pd.Timestamp(start) - pd.Timedelta(days=1)]
        after = deviation[pd.Timestamp(end) + pd.Timedelta(days=1):pd.Timestamp(end) + pd.Timedelta(days=45)]
        baseline = pd.concat([before, after]).median()
        flagged += [day for day, value in episode.items() if abs(value - baseline) > THRESHOLD]
    flagged = pd.DatetimeIndex(flagged)
    repaired, changes = thb.copy(), []
    for day in flagged:
        new = float(cross[day] * local_scale(thb, cross, pd.DatetimeIndex([day]), flagged))
        changes.append({"date": day.strftime("%Y-%m-%d"), "old": float(thb[day]), "new": new})
        repaired[day] = new
    return repaired, changes


def write(series, path, source, changes, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    frame = pd.DataFrame({"Date": series.index.strftime("%Y-%m-%d"), "Close": series.to_numpy()})
    csv = frame.to_csv(index=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv)
    manifest = {"source": source, "repaired_at": datetime.now(timezone.utc).isoformat(), "rows": len(frame),
                "changes": changes, "columns": list(frame), "sha256": hashlib.sha256(csv).hexdigest()}
    path.with_suffix(".csv.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("dataset/raw"), help="folder with the investpy CSVs")
    parser.add_argument("--output", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    usd_twd = read(args.input / "USD_TWD.csv")
    for name, series_id, repair in (("CNY_TWD", "DEXCHUS", fill_cny_gap), ("THB_TWD", "DEXTHUS", repair_thb)):
        path = args.input / f"{name}.csv"
        original = path.read_bytes()
        repaired, changes = repair(read(path), cross_rate(usd_twd, fred_series(series_id)))
        source = {"quotes": f"investpy ({name}.csv, sha256 {hashlib.sha256(original).hexdigest()})",
                  "cross": f"USD_TWD.csv / FRED {series_id}, scaled by the local median quote/cross ratio",
                  "rule": RULES[name]}
        write(repaired, args.output / f"{name}.csv", source, changes, args.overwrite)
        print(f"{name}: {len(changes)} quotes {'filled' if name == 'CNY_TWD' else 'replaced'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
