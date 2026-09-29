"""Pine Script (TradingView) compatible indicator implementations.

Every function mirrors the behaviour of the matching `ta.*` built-in so that
scores computed here match what the TradingView indicator shows on the chart:
- RMA / RSI / ATR / DMI use Wilder smoothing seeded with an SMA.
- EMA is seeded with an SMA of the first `length` values.
- stdev is the population (biased) standard deviation, as in `ta.stdev`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _seeded_recursive(s: pd.Series, n: int, alpha: float) -> pd.Series:
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    start = None
    # find first window of n non-nan values
    count = 0
    for i, x in enumerate(v):
        if np.isnan(x):
            count = 0
            continue
        count += 1
        if count == n:
            start = i
            break
    if start is None:
        return pd.Series(out, index=s.index)
    out[start] = np.mean(v[start - n + 1:start + 1])
    prev = out[start]
    for i in range(start + 1, len(v)):
        x = v[i]
        if np.isnan(x):
            out[i] = prev
            continue
        prev = alpha * x + (1 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=s.index)


def ema(s: pd.Series, n: int) -> pd.Series:
    return _seeded_recursive(s, n, 2.0 / (n + 1))


def rma(s: pd.Series, n: int) -> pd.Series:
    return _seeded_recursive(s, n, 1.0 / n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    ch = close.diff()
    up = rma(ch.clip(lower=0), n)
    dn = rma((-ch).clip(lower=0), n)
    rs = up / dn
    out = 100 - 100 / (1 + rs)
    out = out.where(dn != 0, 100.0)
    out = out.where(up != 0, 0.0).where(~(up.isna() | dn.isna()))
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high, low, close, n: int = 14) -> pd.Series:
    return rma(true_range(high, low, close), n)


def macd(close: pd.Series, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def stdev(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).std(ddof=0)


def bb(close: pd.Series, n=20, mult=2.0):
    mid = sma(close, n)
    dev = stdev(close, n) * mult
    return mid, mid + dev, mid - dev


def dmi(high, low, close, di_len=14, adx_len=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_rma = rma(true_range(high, low, close), di_len)
    plus = 100 * rma(pd.Series(plus_dm, index=high.index), di_len) / tr_rma
    minus = 100 * rma(pd.Series(minus_dm, index=high.index), di_len) / tr_rma
    ssum = plus + minus
    dx = (plus - minus).abs() / ssum.where(ssum != 0, 1)
    adx = 100 * rma(dx, adx_len)
    return plus, minus, adx


def mfi(src: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    ch = src.diff()
    upper = (volume * np.where(ch <= 0, 0.0, src)).rolling(n, min_periods=n).sum()
    lower = (volume * np.where(ch >= 0, 0.0, src)).rolling(n, min_periods=n).sum()
    return 100.0 - 100.0 / (1.0 + upper / lower)


def correlation(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    return a.rolling(n, min_periods=n).corr(b)


def highest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).max()


def lowest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).min()


def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    """Value at bar t = low[t-right] when that bar is a pivot low, else NaN (like ta.pivotlow)."""
    v = low.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for t in range(left + right, len(v)):
        p = t - right
        c = v[p]
        lft = v[p - left:p]
        rgt = v[p + 1:t + 1]
        if np.isnan(c) or np.isnan(lft).any() or np.isnan(rgt).any():
            continue
        if c < lft.min() and c <= rgt.min():
            out[t] = c
    return pd.Series(out, index=low.index)


def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    v = high.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    for t in range(left + right, len(v)):
        p = t - right
        c = v[p]
        lft = v[p - left:p]
        rgt = v[p + 1:t + 1]
        if np.isnan(c) or np.isnan(lft).any() or np.isnan(rgt).any():
            continue
        if c > lft.max() and c >= rgt.max():
            out[t] = c
    return pd.Series(out, index=high.index)


def valuewhen(cond: pd.Series, src: pd.Series, occurrence: int = 0) -> pd.Series:
    """ta.valuewhen: value of src on the Nth most recent bar where cond was true."""
    c = cond.fillna(False).to_numpy(dtype=bool)
    s = src.to_numpy(dtype=float)
    out = np.full(len(s), np.nan)
    hist: list[float] = []
    for i in range(len(s)):
        if c[i]:
            hist.append(s[i])
        if len(hist) > occurrence:
            out[i] = hist[-1 - occurrence]
    return pd.Series(out, index=src.index)


def barssince(cond: pd.Series) -> pd.Series:
    c = cond.fillna(False).to_numpy(dtype=bool)
    out = np.full(len(c), np.nan)
    last = None
    for i in range(len(c)):
        if c[i]:
            last = i
        if last is not None:
            out[i] = i - last
    return pd.Series(out, index=cond.index)


def safe_div(a, b):
    if isinstance(b, (pd.Series, np.ndarray)):
        return np.where(b != 0, a / np.where(b == 0, 1, b), 0.0)
    return a / b if b != 0 else 0.0
