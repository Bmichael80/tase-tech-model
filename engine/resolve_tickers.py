"""Resolve and verify the Yahoo ticker of every stock in config/universe.csv.

For each row: download 2 years of the configured ticker, read Yahoo's company
name, and search Yahoo by ISIN for the .TA listing. Writes
results/ticker_resolution.csv so a wrong ticker can be spotted and fixed.

Usage: python -m engine.resolve_tickers
"""
from pathlib import Path
import time

import pandas as pd

from . import data as D

ROOT = Path(__file__).resolve().parents[1]


def search_isin(isin: str) -> tuple[str, str]:
    import yfinance as yf
    if not isin:
        return "", ""
    try:
        quotes = yf.Search(isin, max_results=10).quotes or []
    except Exception as e:  # network / API change
        return "", f"search error: {str(e)[:60]}"
    ta = [q for q in quotes if str(q.get("symbol", "")).endswith(".TA")]
    q = (ta or quotes or [{}])[0]
    return q.get("symbol", ""), q.get("longname") or q.get("shortname") or ""


def main():
    import yfinance as yf
    uni_rows = pd.read_csv(ROOT / "config" / "universe.csv", dtype=str).fillna("")
    tickers = uni_rows["yahoo"].tolist()
    frames = D.download(tickers, (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d"))
    out = []
    for _, r in uni_rows.iterrows():
        df = frames.get(r["yahoo"])
        try:
            info = yf.Ticker(r["yahoo"]).get_info() or {}
        except Exception:
            info = {}
        s_sym, s_name = search_isin(r["isin"])
        bars = 0 if df is None else len(df)
        status = "OK" if bars >= 200 and (not s_sym or s_sym == r["yahoo"]) else "CHECK"
        out.append({"tase_id": r["tase_id"], "name": r["name"], "isin": r["isin"], "configured": r["yahoo"],
                    "bars_2y": bars, "yahoo_name": info.get("longName") or info.get("shortName") or "",
                    "isin_search_symbol": s_sym, "isin_search_name": s_name, "status": status})
        time.sleep(0.3)
    res = pd.DataFrame(out)
    (ROOT / "results").mkdir(exist_ok=True)
    res.to_csv(ROOT / "results" / "ticker_resolution.csv", index=False, encoding="utf-8-sig")
    bad = res[res["status"] != "OK"]
    print(f"{len(res)} tickers, {len(bad)} to check")
    print(bad.to_string())


if __name__ == "__main__":
    main()
