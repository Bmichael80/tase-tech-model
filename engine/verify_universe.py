"""Check that every Yahoo ticker in config/universe.csv is the intended TASE company.

Writes results/universe_check.csv: ticker, our name, Yahoo's name, currency, last close, bars.
Usage: python -m engine.verify_universe
"""
from pathlib import Path

import pandas as pd

from . import data as D

ROOT = Path(__file__).resolve().parents[1]


def main():
    import yfinance as yf
    uni = D.load_universe()
    frames = D.download([s.yahoo for s in uni], (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d"))
    rows = []
    for s in uni:
        info = {}
        try:
            info = yf.Ticker(s.yahoo).get_info() or {}
        except Exception as e:  # network or unknown ticker
            info = {"error": str(e)[:80]}
        df = frames.get(s.yahoo)
        rows.append({"tase_id": s.tase_id, "name": s.name, "ticker": s.yahoo,
                     "yahoo_name": info.get("longName") or info.get("shortName") or info.get("error", ""),
                     "currency": info.get("currency", ""), "bars_2y": 0 if df is None else len(df),
                     "last_close": None if df is None else round(float(df["close"].iloc[-1]), 2),
                     "flag": "CHECK" if df is None or len(df) < 200 or s.needs_check else ""})
    out = ROOT / "results" / "universe_check.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    bad = [r for r in rows if r["bars_2y"] < 200]
    print(f"checked {len(rows)} tickers, {len(bad)} without enough data")
    for r in bad:
        print("  missing:", r["ticker"], r["name"])


if __name__ == "__main__":
    main()
