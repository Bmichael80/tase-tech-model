"""Market data loading and data-quality checks."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH_TICKER = "^TA125.TA"


@dataclass
class Security:
    tase_id: str
    name: str
    sector: str
    yahoo: str
    needs_check: bool = False


def load_universe(path: Path | None = None) -> list[Security]:
    path = path or ROOT / "config" / "universe.csv"
    with open(path, encoding="utf8") as f:
        return [Security(r["tase_id"], r["name"], r["sector"], r["yahoo"], r.get("needs_check") == "1")
                for r in csv.DictReader(f)]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["close"])


def download(tickers: list[str], start: str, retries: int = 3) -> dict[str, pd.DataFrame]:
    """Daily OHLCV, split- and dividend-adjusted (auto_adjust=True), from Yahoo Finance."""
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    pending = list(tickers)
    for attempt in range(retries):
        if not pending:
            break
        raw = yf.download(pending, start=start, auto_adjust=True, group_by="ticker",
                          threads=True, progress=False)
        failed = []
        for t in pending:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = _normalize(d)
                if len(d) == 0:
                    raise ValueError("empty")
                out[t] = d
            except Exception:
                failed.append(t)
        pending = failed
        if pending:
            time.sleep(5 * (attempt + 1))
    return out


@dataclass
class Quality:
    ok: bool = True
    issues: list[str] = field(default_factory=list)

    def fail(self, msg: str):
        self.ok = False
        self.issues.append(msg)

    def warn(self, msg: str):
        self.issues.append(msg)


def check_quality(df: pd.DataFrame | None, ref_last_date: pd.Timestamp, min_bars: int = 260) -> Quality:
    q = Quality()
    if df is None or df.empty:
        q.fail("no data")
        return q
    if len(df) < min_bars:
        q.fail(f"only {len(df)} daily bars (need {min_bars})")
    lag = (ref_last_date - df.index[-1]).days
    if lag > 4:
        q.fail(f"stale: last bar {df.index[-1].date()} vs market {ref_last_date.date()}")
    tail = df.tail(60)
    if tail[["open", "high", "low", "close"]].isna().any().any():
        q.fail("missing prices in last 60 sessions")
    if (tail["close"] <= 0).any():
        q.fail("non-positive prices")
    bad_hl = (tail["high"] < tail["low"]) | (tail["close"] > tail["high"] * 1.001) | (tail["close"] < tail["low"] * 0.999)
    if bad_hl.sum() > 2:
        q.fail(f"{int(bad_hl.sum())} bars with inconsistent high/low/close")
    zero_vol = int((df["volume"].tail(20) <= 0).sum())
    if zero_vol > 5:
        q.warn(f"{zero_vol} of last 20 sessions without volume")
    jumps = df["close"].pct_change().abs().tail(250)
    big = jumps[jumps > 0.35]
    for d, j in big.items():
        q.warn(f"price move of {j:.0%} on {d.date()} (check for unadjusted corporate action)")
    return q


def sector_composites(closes: dict[str, pd.Series], sectors: dict[str, str]) -> dict[str, pd.Series]:
    """Equal-weight sector index for each stock, built from the OTHER stocks in its sector.

    Needs at least two peers; otherwise None (the model then falls back to the market index).
    """
    rets = pd.DataFrame({t: s.pct_change() for t, s in closes.items()})
    out: dict[str, pd.Series] = {}
    for t, sec in sectors.items():
        peers = [p for p, s in sectors.items() if s == sec and p != t and p in rets]
        if len(peers) < 2:
            out[t] = None
            continue
        r = rets[peers].mean(axis=1, skipna=True).fillna(0)
        out[t] = (1 + r).cumprod() * 100
    return out


def rs_percentiles(closes: dict[str, pd.Series], bench: pd.Series, period: int = 63) -> dict[str, pd.Series]:
    """Cross-sectional percentile rank (0-1) of 3-month excess return vs the benchmark."""
    df = pd.DataFrame(closes)
    b = bench.reindex(df.index).ffill()
    rs = df.pct_change(period) .sub(b.pct_change(period), axis=0)
    pct = rs.rank(axis=1, pct=True)
    return {t: pct[t] for t in df.columns}
