"""V21 technical rating: cross-sectional momentum / trend composite (TREND4).

Chosen by the factor research (results/research/FACTORS.md): the four factors
below passed the in-sample rule (2013-2019, 3-month IC t >= 2) and belong to
the momentum/trend family documented in the academic literature. In the
2020-2026 blind test the top quintile beat the TA-125 by +2.3% over 3 months
and +3.6% over 6 months, while quintiles 1-4 lagged it.

Score = percentile (0-100) of the equal-weight average of the four factor
percentiles, ranked among the liquid stocks of the universe on the same day.
Rating: BUY = top quintile (>= 80), SELL = bottom quintile (< 20), HOLD otherwise.
The rating is relative: it says how a stock's technical strength compares with
the other stocks in the universe, not whether the market will rise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ta

FACTORS = {
    "mom_6_1": "מומנטום 6 חודשים (ללא החודש האחרון)",
    "hi52": "קרבה לשיא 52 שבועות",
    "dist_ema200": "מרחק מממוצע 200",
    "ema50_200": "ממוצע 50 מול ממוצע 200",
}
BUY_AT = 80.0
SELL_BELOW = 20.0
MIN_TURNOVER_ILS = 1_000_000
MIN_BARS = 260


def factor_panel(frames: dict[str, pd.DataFrame], index: pd.Index) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    C = pd.DataFrame({t: f["close"] for t, f in frames.items()}).reindex(index)
    H = pd.DataFrame({t: f["high"] for t, f in frames.items()}).reindex(index)
    V = pd.DataFrame({t: f["volume"] for t, f in frames.items()}).reindex(index)
    bars = C.notna().cumsum()
    C = C.ffill(limit=5)
    ema200 = C.apply(lambda s: ta.ema(s.dropna(), 200).reindex(s.index))
    ema50 = C.apply(lambda s: ta.ema(s.dropna(), 50).reindex(s.index))
    F = {
        "mom_6_1": C.shift(21) / C.shift(126) - 1,
        "hi52": C / H.rolling(252, min_periods=200).max(),
        "dist_ema200": C / ema200 - 1,
        "ema50_200": ema50 / ema200 - 1,
    }
    turnover = (C * V / 100.0).rolling(20, min_periods=10).median()  # agorot -> ILS
    eligible = (turnover >= MIN_TURNOVER_ILS) & (bars >= MIN_BARS)
    return F, turnover, eligible


def score(F: dict[str, pd.DataFrame], eligible: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Composite percentile score (0-100) per stock and day, plus each factor's percentile."""
    pct = {n: F[n].where(eligible).rank(axis=1, pct=True) for n in FACTORS}
    comp = sum(pct.values()) / len(pct)
    return comp.rank(axis=1, pct=True) * 100, {n: p * 100 for n, p in pct.items()}


def rating(s: float | None) -> str:
    if s is None or np.isnan(s):
        return "N/A"
    return "BUY" if s >= BUY_AT else "SELL" if s < SELL_BELOW else "HOLD"
