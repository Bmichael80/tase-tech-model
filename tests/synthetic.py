"""Synthetic OHLCV generator used by the tests (no network needed)."""
import numpy as np
import pandas as pd


def make_ohlcv(n=1500, seed=0, drift=0.0004, vol=0.018, start_price=5000.0, end=None):
    rng = np.random.default_rng(seed)
    end = end or pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end=end, periods=n)
    r = rng.normal(drift, vol, n)
    # regime shifts so trends, pullbacks and reversals all occur
    r += np.sin(np.arange(n) / 60.0) * vol * 0.25
    close = start_price * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, vol * 0.35, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol * 0.5, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol * 0.5, n)))
    volume = rng.lognormal(12, 0.5, n) * (1 + 2 * (np.abs(r) > 2 * vol))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
