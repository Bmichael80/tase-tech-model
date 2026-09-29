"""V22 technical score: strength vs the TA-125 over the last 6 months, plus the user's model as entry flags.

Chosen by engine/rs_study.py (results/research/RS.md). Seven look-backs were tested (1, 3, 6, 12 months,
6-1, 12-1, IBD blend); 6 months had the best in-sample 3-month IC (2013-2019) and held in the 2020-2026
blind test (IC 0.093, t 2.35; top quintile +4.4% vs TA-125 over the next 3 months).

Score  = percentile (0-100), among liquid stocks, of: 6-month stock return minus 6-month TA-125 return.
Rating = BUY  if score >= 80 and the user's model is not at EXIT (sell score >= 34)
         HOLD if 50 <= score < 80, or score >= 80 with the user's model at EXIT
         SELL if score < 50
         (exit below 50 was the best of 20/35/50 in both periods; the EXIT veto helped slightly in both)
Flag   = the user's model (Analyst Dashboard V20), shown next to the rating, never changing it:
         STRONG BUY / BUY  = its entry signal during the last trading week
         STRONG SELL / SELL = its sell action today: EXIT (sell score >= 34) / REDUCE (22-33)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 126            # trading days = 6 months
BUY_AT = 80.0
SELL_BELOW = 50.0
V20_EXIT = 34.0
MIN_TURNOVER_ILS = 1_000_000
MIN_BARS = 260


def panel(frames: dict[str, pd.DataFrame], bench_close: pd.Series, index: pd.Index):
    C = pd.DataFrame({t: f["close"] for t, f in frames.items()}).reindex(index)
    V = pd.DataFrame({t: f["volume"] for t, f in frames.items()}).reindex(index)
    bars = C.notna().cumsum()
    C = C.ffill(limit=5)
    b = bench_close.reindex(index).ffill()
    stock_ret = C / C.shift(LOOKBACK) - 1
    bench_ret = b / b.shift(LOOKBACK) - 1
    excess = stock_ret.sub(bench_ret, axis=0)
    turnover = (C * V / 100.0).rolling(20, min_periods=10).median()      # agorot -> ILS
    eligible = (turnover >= MIN_TURNOVER_ILS) & (bars >= MIN_BARS) & excess.notna()
    score = excess.where(eligible).rank(axis=1, pct=True) * 100
    return score, stock_ret, bench_ret, excess, turnover, eligible


def rating(score: float | None, sell_score: float | None) -> tuple[str, str]:
    if score is None or np.isnan(score):
        return "N/A", ""
    if score >= BUY_AT:
        if sell_score is not None and not np.isnan(sell_score) and sell_score >= V20_EXIT:
            return "HOLD", "BUY blocked: your model shows EXIT"
        return "BUY", ""
    return ("HOLD", "") if score >= SELL_BELOW else ("SELL", "")


def flag(signal_week: str | None, sell_action: str | None) -> str | None:
    if sell_action == "EXIT":
        return "STRONG SELL"
    if sell_action == "REDUCE":
        return "SELL"
    if signal_week in ("STRONG BUY", "BUY"):
        return signal_week
    return None
