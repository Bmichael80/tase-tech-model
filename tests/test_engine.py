import numpy as np
import pandas as pd
import pytest

from engine import indicators as ta
from engine.model import ModelConfig, compute_daily, compute_weekly_signal, compute_whale, _divergence_raw
from tests.synthetic import make_ohlcv


@pytest.fixture(scope="module")
def px():
    return make_ohlcv(n=1200, seed=7)


@pytest.fixture(scope="module")
def bench():
    return make_ohlcv(n=1200, seed=99, vol=0.01, start_price=200000)


def test_ema_converges_to_reference(px):
    ref = px["close"].ewm(span=20, adjust=False).mean()
    assert abs(ta.ema(px["close"], 20).iloc[-1] - ref.iloc[-1]) < 1e-6 * ref.iloc[-1]


def test_rsi_matches_wilder_reference(px):
    ch = px["close"].diff()
    up = ch.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-ch).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    ref = 100 - 100 / (1 + up / dn)
    assert abs(ta.rsi(px["close"]).iloc[-1] - ref.iloc[-1]) < 1e-6


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 100, dtype=float))
    assert ta.rsi(up).iloc[-1] == 100.0
    r = ta.rsi(make_ohlcv(500, seed=3)["close"]).dropna()
    assert r.between(0, 100).all()


def test_atr_reference(px):
    tr = ta.true_range(px["high"], px["low"], px["close"])
    ref = tr.ewm(alpha=1 / 14, adjust=False).mean()
    assert abs(ta.atr(px["high"], px["low"], px["close"]).iloc[-1] - ref.iloc[-1]) < 1e-6 * ref.iloc[-1]


def test_pivot_low_detection():
    low = pd.Series([10, 9, 8, 7, 6, 5, 3, 4, 5, 6, 7], dtype=float)
    p = ta.pivot_low(low, 5, 2)
    assert p.iloc[8] == 3 and p.drop(8).isna().all()


def test_divergence_fix_detects_consecutive_pivots():
    # two pivot lows (bars 10 and 30): price lower low, oscillator higher low
    n = 40
    low = pd.Series(np.full(n, 100.0))
    low[10] = 90.0
    low[30] = 85.0
    osc = pd.Series(np.full(n, 50.0))
    osc[10] = 20.0
    osc[30] = 35.0
    piv = ta.pivot_low(low, 5, 2)
    fixed = _divergence_raw(low, osc, piv, True, True, 60)
    assert fixed[32] and fixed.sum() == 1
    # the V19 comparison (pivot vs the bar before it) cannot see this divergence
    old = _divergence_raw(low, osc, piv, True, False, 60)
    assert not old[32]


def test_scores_bounded_and_layers_add_up(px, bench):
    for cfg in (ModelConfig(), ModelConfig(version="V19")):
        d = compute_daily(px, bench, cfg)
        s = d["score"].dropna()
        assert s.between(0, 100).all()
        layers = d[[c for c in d.columns if c.startswith("layer_")]].sum(axis=1)
        assert np.allclose(layers.dropna(), d["score"].dropna())
        assert set(np.unique(d["signal"])) <= {0, 1, 2}
        # a BUY always requires the full setup and the threshold
        buys = d[d["signal"] > 0]
        assert (buys["setup_ready"]).all() and (buys["score"] >= cfg.buy_threshold).all()


def test_weekly_normalized(px):
    w = compute_weekly_signal(px, ModelConfig())
    assert w["weekly_score"].dropna().between(0, 100.0001).all()


def test_liquidity_gate_blocks_signals(px, bench):
    thin = px.copy()
    thin["volume"] = 1.0
    d = compute_daily(thin, bench, ModelConfig())
    assert (d["signal"] == 0).all() and not d["liquid"].iloc[-1]


def test_v19_flags():
    c = ModelConfig(version="V19")
    assert not (c.fix_divergence or c.liquidity_gate or c.normalize_weekly or c.adx_filter or c.rs_percentile)


def test_whale_runs(px):
    w = compute_whale(px, ModelConfig())
    assert len(w) == len(px) and w["mfi"].dropna().between(0, 100).all()
