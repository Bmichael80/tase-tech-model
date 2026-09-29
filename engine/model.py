"""Analyst Dashboard technical model — Python port of the Pine Script
"Analyst Dashboard V19.0 + Whale Fusion V2" with the V20 corrections.

`ModelConfig(version="V19")` reproduces the original TradingView logic as
written (including its divergence behaviour). `ModelConfig()` (V20) applies
the corrections documented in METHODOLOGY.md. Experimental options are off
by default and are only switched on after the backtest supports them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from . import indicators as ta

TICK = 1e-9


@dataclass
class ModelConfig:
    version: str = "V20"
    # --- Analyst Dashboard inputs (unchanged from V19) ---
    lookback_range: int = 5
    rsi_max: int = 45
    rsi_min: int = 30
    rsi_sell_limit: int = 70
    min_beta: float = 1.0
    rvol_threshold: float = 1.2
    adx_min: int = 20
    rs_period: int = 63
    max_dist_ema200: float = 15.0
    recent_days: int = 4
    pullback_lookback: int = 20
    min_pullback_pct: float = 4.0
    min_distance_from_high_pct: float = 1.5
    below_high_zero_before_pct: float = 5.0
    below_high_full_from_pct: float = 12.0
    below_high_full_to_pct: float = 18.0
    below_high_zero_after_pct: float = 25.0
    dist_ema200_zero_after_pct: float = 25.0
    max_atr_pct: float = 8.0
    max_stop_risk_pct: float = 7.0
    stop_buffer_atr_mult: float = 0.5
    min_trigger_body_pct: float = 40.0
    trigger_max_atr_mult: float = 1.8
    rs_rising_period: int = 20
    updown_vol_period: int = 20
    pocket_pivot_lookback: int = 10
    buy_threshold: float = 68.0
    strong_buy_threshold: float = 82.0
    # --- weights (sum = 100) ---
    w: dict = field(default_factory=lambda: dict(
        pullback=8.0, below_high=8.0, near_support=4.0, rsi_div=8.0, macd_div=5.0,
        engulf=7.0, hammer=6.0, harami=4.0, trigger=10.0, above_ema200=5.0,
        ema50_above_ema200=3.0, market_safe=2.0, sector_safe=2.0, dist_ema200=3.0,
        cmf=4.0, rs=2.0, rs_rising=3.0, rvol=3.0, inst_absorption=3.0, volume_dryup=2.0,
        vcp=2.0, updown_volume=3.0, pocket_pivot=3.0, atr_risk=5.0, stop_risk=5.0))
    # --- Whale Fusion inputs ---
    mfi_period: int = 14
    threshold_mult: float = 1.5
    alert_lookback: int = 10
    breakout_scan_window: int = 4
    bt_lookback: int = 5
    min_whale_sigma: float = 3.0
    extreme_whale_sigma: float = 5.0
    # --- V20 corrections ---
    fix_divergence: bool = True          # compare consecutive pivots, not the bar before the pivot
    div_max_gap: int = 60                # max bars between the two pivots of a divergence
    liquidity_gate: bool = True          # no BUY signal for illiquid stocks
    min_turnover_ils: float = 1_000_000  # median daily turnover (ILS) over 20 sessions
    normalize_weekly: bool = True        # weekly score rescaled to 0-100 (its max is 89)
    # --- experimental (evaluated in the backtest before adoption) ---
    adx_filter: bool = False             # trend context also requires ADX >= adx_min or +DI > -DI
    rs_percentile: bool = False          # RS layer from cross-sectional percentile rank

    def __post_init__(self):
        if self.version == "V19":
            self.fix_divergence = False
            self.liquidity_gate = False
            self.normalize_weekly = False
            self.adx_filter = False
            self.rs_percentile = False

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- scoring helpers
def score_below_recent_high(d, zb, ff, ft, za, w):
    d = np.asarray(d, dtype=float)
    out = np.zeros_like(d)
    m1 = (d > zb) & (d < ff)
    out[m1] = w * (d[m1] - zb) / (ff - zb)
    m2 = (d >= ff) & (d <= ft)
    out[m2] = w
    m3 = (d > ft) & (d < za)
    out[m3] = w * (za - d[m3]) / (za - ft)
    return np.nan_to_num(out)


def score_ema_distance(d, full, za, w):
    d = np.asarray(d, dtype=float)
    out = np.zeros_like(d)
    m1 = (d >= 0) & (d <= full)
    out[m1] = w
    m2 = (d > full) & (d < za)
    out[m2] = w * (za - d[m2]) / (za - full)
    return np.nan_to_num(out)


def score_under_threshold(v, full_to, zero_after, w):
    v = np.asarray(v, dtype=float)
    out = np.where(v <= full_to, w, np.where(v < zero_after, w * (zero_after - v) / (zero_after - full_to), 0.0))
    return np.where(np.isnan(v), 0.0, out)


def _b(x) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0).astype(bool)


def _divergence_raw(low_or_high: pd.Series, osc: pd.Series, pivot: pd.Series, bullish: bool,
                    fixed: bool, max_gap: int) -> np.ndarray:
    """Raw divergence event at the pivot confirmation bar (pivot bar = t-2)."""
    n = len(pivot)
    px = low_or_high.to_numpy(dtype=float)
    os_ = osc.to_numpy(dtype=float)
    piv = pivot.notna().to_numpy()
    out = np.zeros(n, dtype=bool)
    if not fixed:
        # V19 as written: barssince(pivot) == 0 on the pivot bar, so the comparison is
        # pivot bar (t-2) vs the bar right before it (t-3).
        for t in range(3, n):
            if not piv[t]:
                continue
            a, b = t - 2, t - 3
            if bullish:
                out[t] = px[a] < px[b] and os_[a] > os_[b]
            else:
                out[t] = px[a] > px[b] and os_[a] < os_[b]
        return out
    prev_p = None
    for t in range(n):
        if not piv[t]:
            continue
        p = t - 2
        if prev_p is not None and 3 <= p - prev_p <= max_gap and not np.isnan(os_[p]) and not np.isnan(os_[prev_p]):
            if bullish:
                out[t] = px[p] < px[prev_p] and os_[p] > os_[prev_p]
            else:
                out[t] = px[p] > px[prev_p] and os_[p] < os_[prev_p]
        prev_p = p
    return out


def _timer_active(raw: np.ndarray, start: int = 7) -> np.ndarray:
    timer = 0
    out = np.zeros(len(raw), dtype=bool)
    for i, r in enumerate(raw):
        if r:
            timer = start
        timer = max(0, timer - 1)
        out[i] = timer > 0
    return out


# ---------------------------------------------------------------- core bar computations
def _core(df: pd.DataFrame, cfg: ModelConfig) -> dict:
    """Indicators and layer scores shared by the daily and weekly models."""
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    W = cfg.w
    r = {}
    r["ema20"], r["ema50"], r["ema200"] = ta.ema(c, 20), ta.ema(c, 50), ta.ema(c, 200)
    r["rsi"] = ta.rsi(c, 14)
    bb_mid, bb_up, bb_lo = ta.bb(c, 20, 2.0)
    macd_line, sig_line, hist = ta.macd(c, 12, 26, 9)
    r["macd_line"], r["signal_line"], r["hist"] = macd_line, sig_line, hist
    plus_di, minus_di, adx = ta.dmi(h, l, c, 14, 14)
    r["plus_di"], r["minus_di"], r["adx"] = plus_di, minus_di, adx
    r["dist_ema200"] = pd.Series(ta.safe_div((c - r["ema200"]), r["ema200"]) * 100, index=c.index)
    atr = ta.atr(h, l, c, 14)
    r["atr"] = atr
    r["atr_pct"] = pd.Series(ta.safe_div(atr, c) * 100, index=c.index)

    low_p = ta.pivot_low(l, 5, 2)
    high_p = ta.pivot_high(h, 5, 2)
    r["low_p"], r["high_p"] = low_p, high_p
    last_low_p = ta.valuewhen(low_p.notna(), l.shift(2), 0)
    r["last_low_p"] = last_low_p
    r["near_support"] = _b(last_low_p.notna() & (c <= last_low_p * 1.025) & (c >= last_low_p * 0.98))

    tr = ta.true_range(h, l, c)
    maK, rK = ta.sma(c, 20), ta.sma(tr, 20)
    r["vcp"] = _b((bb_lo > maK - rK * 1.5) & (bb_up < maK + rK * 1.5))
    rng = (h - l).clip(lower=TICK)
    mf_mult = ((c - l) - (h - c)) / rng
    r["cmf"] = pd.Series(ta.safe_div(ta.sma(mf_mult * v, 20), ta.sma(v, 20)), index=c.index)
    avg_v = ta.sma(v, 10)
    r["rvol"] = v / avg_v.where(avg_v > 0, 1)
    r["inst_absorption"] = _b((r["rvol"] > 1.5) & (c > l + (h - l) * 0.66))
    sma20v = ta.sma(v, 20)
    r["volume_dryup"] = _b(v.shift(1) < sma20v.shift(1))
    close_pos = (c - l) / rng * 100
    body = (c - o).abs()
    body_pct = body / rng * 100
    trig = (3.0 * (close_pos >= 66) + 3.0 * (c > h.shift(1)) + 2.0 * ((c > o) & (body_pct >= cfg.min_trigger_body_pct))
            + 2.0 * (rng <= atr * cfg.trigger_max_atr_mult))
    r["trigger_score"] = trig.to_numpy(dtype=float)
    r["trigger_ready"] = r["trigger_score"] >= 6.0
    up_avg = ta.sma(v.where(c > o, 0.0), cfg.updown_vol_period)
    dn_avg = ta.sma(v.where(c < o, 0.0), cfg.updown_vol_period)
    r["updown_ratio"] = pd.Series(ta.safe_div(up_avg, dn_avg), index=c.index)
    r["updown_ok"] = r["updown_ratio"].to_numpy() > 1.0
    down_for_pocket = v.where(c < o, 0.0)
    highest_down = ta.highest(down_for_pocket, cfg.pocket_pivot_lookback).shift(1)
    r["pocket_pivot"] = _b((c > o) & (v > highest_down) & (c > c.shift(1)))
    r["dryup_expansion"] = _b((v.shift(1) < sma20v.shift(1) * 0.8) & (v > sma20v) & (c > o))

    r["bull_div_rsi"] = _timer_active(_divergence_raw(l, r["rsi"], low_p, True, cfg.fix_divergence, cfg.div_max_gap))
    r["bull_div_macd"] = _timer_active(_divergence_raw(l, hist, low_p, True, cfg.fix_divergence, cfg.div_max_gap))
    r["bear_div_rsi"] = _timer_active(_divergence_raw(h, r["rsi"], high_p, False, cfg.fix_divergence, cfg.div_max_gap))
    r["bear_div_macd"] = _timer_active(_divergence_raw(h, hist, high_p, False, cfg.fix_divergence, cfg.div_max_gap))

    lw = pd.concat([o, c], axis=1).min(axis=1) - l
    uw = h - pd.concat([o, c], axis=1).max(axis=1)
    r["hammer_raw"] = _b((lw.shift(1) > body.shift(1) * 2) & (uw.shift(1) < body.shift(1) * 0.5) & (c > h.shift(1)))
    r["engulf_raw"] = _b((c.shift(2) < o.shift(2)) & (c.shift(1) > o.shift(1)) & (c.shift(1) > o.shift(2)) & (c > h.shift(1)))
    r["harami_raw"] = _b((c.shift(2) < o.shift(2)) & (c.shift(1) > o.shift(1)) & (c.shift(1) < o.shift(2)) & (c > h.shift(1)))
    r["star_raw"] = _b((uw.shift(1) > body.shift(1) * 2) & (lw.shift(1) < body.shift(1) * 0.5) & (c < l.shift(1)))

    recent_high = ta.highest(h, cfg.pullback_lookback)
    recent_low = ta.lowest(l, cfg.pullback_lookback)
    r["pullback_depth_pct"] = ((recent_high - recent_low) / recent_high * 100).to_numpy()
    r["dist_from_high_pct"] = ((recent_high - c) / recent_high * 100).to_numpy()
    r["had_pullback"] = r["pullback_depth_pct"] >= cfg.min_pullback_pct
    r["below_high_score"] = score_below_recent_high(r["dist_from_high_pct"], cfg.below_high_zero_before_pct,
                                                    cfg.below_high_full_from_pct, cfg.below_high_full_to_pct,
                                                    cfg.below_high_zero_after_pct, W["below_high"])
    za = max(cfg.dist_ema200_zero_after_pct, cfg.max_dist_ema200 + 0.5)
    r["dist_ema200_score"] = score_ema_distance(r["dist_ema200"], cfg.max_dist_ema200, za, W["dist_ema200"])
    stop_ref = np.where(last_low_p.notna(), last_low_p - atr * cfg.stop_buffer_atr_mult, c - atr * 2)
    r["stop_reference"] = stop_ref
    r["stop_risk_pct"] = np.maximum(0.0, np.nan_to_num((c.to_numpy() - stop_ref) / c.to_numpy() * 100))
    r["atr_risk_score"] = score_under_threshold(r["atr_pct"], cfg.max_atr_pct, cfg.max_atr_pct * 1.6, W["atr_risk"])
    r["stop_risk_score"] = score_under_threshold(r["stop_risk_pct"], cfg.max_stop_risk_pct, cfg.max_stop_risk_pct * 1.6, W["stop_risk"])
    r["risk_score"] = r["atr_risk_score"] + r["stop_risk_score"]
    r["structure_intact"] = np.where(last_low_p.notna(), c >= last_low_p, True).astype(bool)
    r["above_ema200"] = _b(c > r["ema200"])
    r["ema50_above_ema200"] = _b(r["ema50"] > r["ema200"])
    return r


def _patterns_in_window(r: dict, window: int, W: dict):
    """Pine loop: for i in 0..window-1, engulf > hammer > harami, most recent bar first."""
    n = len(r["engulf_raw"])
    score = np.zeros(n)
    name = np.array(["None"] * n, dtype=object)
    bear = np.zeros(n, dtype=bool)
    for t in range(n):
        for i in range(window):
            j = t - i
            if j < 0:
                break
            if r["engulf_raw"][j]:
                score[t], name[t] = W["engulf"], "Engulfing"; break
            if r["hammer_raw"][j]:
                score[t], name[t] = W["hammer"], "Hammer"; break
            if r["harami_raw"][j]:
                score[t], name[t] = W["harami"], "Harami"; break
        bear[t] = r["star_raw"][max(0, t - window + 1):t + 1].any()
    return score, name, bear


def _align(series: pd.Series | None, index: pd.Index) -> pd.Series | None:
    if series is None:
        return None
    return series.reindex(index.union(series.index)).ffill().reindex(index)


def compute_daily(df: pd.DataFrame, bench: pd.DataFrame, cfg: ModelConfig,
                  sector_close: pd.Series | None = None,
                  rs_pct: pd.Series | None = None, price_divisor: float = 100.0) -> pd.DataFrame:
    """Full daily model. df/bench: DataFrames with open/high/low/close/volume on a DatetimeIndex."""
    W = cfg.w
    c, o, v = df["close"].astype(float), df["open"].astype(float), df["volume"].astype(float)
    r = _core(df, cfg)
    idx = df.index
    pat_score, pat_name, bear_pat = _patterns_in_window(r, cfg.lookback_range, W)

    bench_c = _align(bench["close"].astype(float), idx)
    bench_e200 = _align(ta.ema(bench["close"].astype(float), 200), idx)
    bench_c = bench_c.fillna(c)
    bench_e200 = bench_e200.fillna(r["ema200"])
    if sector_close is not None and sector_close.notna().sum() > 60:
        sect_c = _align(sector_close, idx)
        sect_e50 = _align(ta.ema(sector_close.dropna(), 50), idx)
    else:
        sect_c, sect_e50 = bench_c, ta.ema(bench_c, 50)
    sect_c = sect_c.fillna(bench_c)
    sect_e50 = sect_e50.fillna(ta.ema(bench_c, 50))

    ret = c.pct_change()
    bret = bench_c.pct_change()
    beta = ta.correlation(ret, bret, 20) * (ta.stdev(ret, 20) / ta.stdev(bret, 20))
    rs_value = (c / c.shift(cfg.rs_period) - 1) - (bench_c / bench_c.shift(cfg.rs_period) - 1)
    rs_rising = _b(rs_value > rs_value.shift(cfg.rs_rising_period))

    market_safe = _b(bench_c > bench_e200)
    sector_safe = _b(sect_c > sect_e50)

    trend = (W["above_ema200"] * r["above_ema200"] + W["ema50_above_ema200"] * r["ema50_above_ema200"]
             + W["market_safe"] * market_safe + W["sector_safe"] * sector_safe + r["dist_ema200_score"])
    pullback = W["pullback"] * r["had_pullback"] + r["below_high_score"] + W["near_support"] * r["near_support"]
    reversal = (W["rsi_div"] * r["bull_div_rsi"] + W["macd_div"] * r["bull_div_macd"] + pat_score
                + np.minimum(r["trigger_score"], W["trigger"]))
    rvol_ok = _b(r["rvol"] >= cfg.rvol_threshold)
    cmf_pos = _b(r["cmf"] > 0)
    volume = (W["cmf"] * cmf_pos + W["rvol"] * rvol_ok + W["inst_absorption"] * r["inst_absorption"]
              + W["volume_dryup"] * r["volume_dryup"] + W["vcp"] * r["vcp"] + W["updown_volume"] * r["updown_ok"]
              + W["pocket_pivot"] * r["pocket_pivot"])
    if cfg.rs_percentile and rs_pct is not None:
        pct = _align(rs_pct, idx).fillna(0.5).to_numpy()
        rel = (W["rs"] + W["rs_rising"]) * pct
    else:
        rel = W["rs"] * _b(rs_value > 0) + W["rs_rising"] * rs_rising
    risk = r["risk_score"]
    score = trend + pullback + reversal + volume + risk + rel

    bull_pattern = pat_score > 0
    rev_count = (r["near_support"].astype(int) + bull_pattern.astype(int) + r["bull_div_rsi"].astype(int)
                 + r["bull_div_macd"].astype(int) + r["trigger_ready"].astype(int) + r["pocket_pivot"].astype(int))
    reversal_ready = rev_count >= 2
    strong_rev = (rev_count >= 3) | ((rev_count >= 2) & (r["bull_div_rsi"] | r["bull_div_macd"]))

    trend_ready = r["above_ema200"] & r["ema50_above_ema200"] & (r["dist_ema200_score"] > 0)
    if cfg.adx_filter:
        trend_ready = trend_ready & (_b(r["adx"] >= cfg.adx_min) | _b(r["plus_di"] > r["minus_di"]))
    pullback_ready = r["had_pullback"] & (r["below_high_score"] > 0) & r["structure_intact"]
    volume_ready = cmf_pos & (rvol_ok | r["pocket_pivot"] | r["updown_ok"] | r["dryup_expansion"])
    risk_ready = risk >= 6.0
    market_regime_ready = market_safe | sector_safe
    setup_ready = market_regime_ready & trend_ready & pullback_ready & reversal_ready & volume_ready & risk_ready

    # liquidity
    turnover = (c * v) / price_divisor          # TASE prices on Yahoo are in agorot
    med_turnover = turnover.rolling(20, min_periods=10).median()
    liquid = np.nan_to_num(med_turnover.to_numpy(), nan=0.0) >= cfg.min_turnover_ils
    # sell score
    llp = r["last_low_p"]
    down_volume = _b((c < o) & (r["rvol"] > 1.1))
    macd_bear = _b((r["macd_line"] < r["signal_line"]) & (r["hist"] < r["hist"].shift(1)))
    below20, below50 = _b(c < r["ema20"]), _b(c < r["ema50"])
    breaks_pivot = _b(llp.notna() & (c < llp))
    sell = (8 * _b(r["rsi"] >= cfg.rsi_sell_limit) + 12 * r["bear_div_rsi"] + 10 * r["bear_div_macd"] + 8 * bear_pat
            + 7 * below20 + 8 * below50 + 8 * _b(r["cmf"] < 0) + 8 * macd_bear + 8 * breaks_pivot + 6 * down_volume)

    n = len(c)
    sig = np.zeros(n, dtype=int)
    conflict = np.zeros(n, dtype=bool)
    suppressed = np.zeros(n, dtype=bool)
    cooldown = 0
    sc = np.nan_to_num(score)
    for t in range(n):
        if cooldown > 0:
            cooldown -= 1
        gate = (not cfg.liquidity_gate) or liquid[t]
        strong_raw = gate and setup_ready[t] and sc[t] >= cfg.strong_buy_threshold and strong_rev[t] and cooldown == 0
        buy_raw = gate and setup_ready[t] and sc[t] >= cfg.buy_threshold and not strong_raw and cooldown == 0
        final_sell = sell[t] >= 34
        conflict[t] = buy_raw and final_sell
        suppressed[t] = strong_raw and final_sell
        is_buy = buy_raw and not conflict[t]
        if strong_raw or is_buy:
            cooldown = 4
        sig[t] = 2 if strong_raw else (1 if is_buy else 0)

    out = pd.DataFrame(index=idx)
    out["close"] = c
    out["score"] = score
    out["layer_trend"], out["layer_pullback"], out["layer_reversal"] = trend, pullback, reversal
    out["layer_volume"], out["layer_risk"], out["layer_rs"] = volume, risk, rel
    out["signal"] = sig
    out["conflict"] = conflict
    out["sell_suppressed"] = suppressed
    out["setup_ready"] = setup_ready
    out["reversal_count"] = rev_count
    out["sell_score"] = sell
    out["rsi"], out["adx"], out["atr_pct"] = r["rsi"], r["adx"], r["atr_pct"]
    out["dist_ema200"], out["cmf"], out["rvol"] = r["dist_ema200"], r["cmf"], r["rvol"]
    out["beta"], out["rs_value"] = beta, rs_value
    out["ema20"], out["ema50"], out["ema200"] = r["ema20"], r["ema50"], r["ema200"]
    out["dist_from_high_pct"], out["stop_risk_pct"] = r["dist_from_high_pct"], r["stop_risk_pct"]
    out["stop_reference"] = r["stop_reference"]
    out["pattern"] = pat_name
    out["bull_div_rsi"], out["bull_div_macd"] = r["bull_div_rsi"], r["bull_div_macd"]
    out["market_safe"], out["sector_safe"] = market_safe, sector_safe
    out["trend_ready"], out["pullback_ready"] = trend_ready, pullback_ready
    out["volume_ready"], out["risk_ready"], out["reversal_ready"] = volume_ready, risk_ready, reversal_ready
    out["pocket_pivot"], out["trigger_ready"], out["near_support"] = r["pocket_pivot"], r["trigger_ready"], r["near_support"]
    out["median_turnover"] = med_turnover
    out["liquid"] = liquid
    return out


def compute_weekly_signal(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Weekly bundle from the Pine script (no market/sector/RS/VCP terms)."""
    W = cfg.w
    wk = df.resample("W-SAT").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    r = _core(wk, cfg)
    c = wk["close"]
    pat = np.where(r["engulf_raw"], W["engulf"], np.where(r["hammer_raw"], W["hammer"], np.where(r["harami_raw"], W["harami"], 0.0)))
    trend = W["above_ema200"] * r["above_ema200"] + W["ema50_above_ema200"] * r["ema50_above_ema200"] + r["dist_ema200_score"]
    pullback = W["pullback"] * r["had_pullback"] + r["below_high_score"] + W["near_support"] * r["near_support"]
    reversal = W["rsi_div"] * r["bull_div_rsi"] + W["macd_div"] * r["bull_div_macd"] + pat + np.minimum(r["trigger_score"], W["trigger"])
    rvol_ok = _b(r["rvol"] >= cfg.rvol_threshold)
    cmf_pos = _b(r["cmf"] > 0)
    volume = (W["cmf"] * cmf_pos + W["rvol"] * rvol_ok + W["inst_absorption"] * r["inst_absorption"]
              + W["volume_dryup"] * r["volume_dryup"] + W["updown_volume"] * r["updown_ok"] + W["pocket_pivot"] * r["pocket_pivot"])
    score = trend + pullback + reversal + volume + r["risk_score"]
    max_score = (W["above_ema200"] + W["ema50_above_ema200"] + W["dist_ema200"] + W["pullback"] + W["below_high"]
                 + W["near_support"] + W["rsi_div"] + W["macd_div"] + W["engulf"] + W["trigger"] + W["cmf"] + W["rvol"]
                 + W["inst_absorption"] + W["volume_dryup"] + W["updown_volume"] + W["pocket_pivot"] + W["atr_risk"] + W["stop_risk"])
    if cfg.normalize_weekly:
        score = score * 100.0 / max_score
    rev_count = (r["near_support"].astype(int) + (pat > 0).astype(int) + r["bull_div_rsi"].astype(int)
                 + r["bull_div_macd"].astype(int) + r["trigger_ready"].astype(int) + r["pocket_pivot"].astype(int))
    strong_rev = (rev_count >= 3) | ((rev_count >= 2) & (r["bull_div_rsi"] | r["bull_div_macd"]))
    trend_ready = r["above_ema200"] & r["ema50_above_ema200"] & (r["dist_ema200_score"] > 0)
    pullback_ready = r["had_pullback"] & (r["below_high_score"] > 0) & r["structure_intact"]
    volume_ready = cmf_pos & (rvol_ok | r["pocket_pivot"] | r["updown_ok"] | r["dryup_expansion"])
    setup = trend_ready & pullback_ready & (rev_count >= 2) & volume_ready & (r["risk_score"] >= 6.0)
    sig = np.where(setup & (score >= cfg.strong_buy_threshold) & strong_rev, 2, np.where(setup & (score >= cfg.buy_threshold), 1, 0))
    return pd.DataFrame({"weekly_score": score, "weekly_signal": sig}, index=wk.index)


def compute_whale(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    h, l, c, v = (df[k].astype(float) for k in ("high", "low", "close", "volume"))
    m = ta.mfi(c, v, cfg.mfi_period)
    pch = c - c.shift(5)
    mch = m - m.shift(5)
    sd = ta.stdev(mch, 50)
    acc = _b((sd > 0) & (pch <= 0) & (mch > sd * cfg.threshold_mult))
    dist = _b((sd > 0) & (pch >= 0) & (mch < -sd * cfg.threshold_mult))
    since_acc = ta.barssince(pd.Series(acc, index=c.index)).to_numpy()
    hv = h.to_numpy()
    acc_high = np.full(len(c), np.nan)
    cur = np.nan
    for t in range(len(c)):
        if acc[t]:
            cur = max(hv[t], cur) if (t > 0 and acc[t - 1] and not np.isnan(cur)) else hv[t]
        elif not np.isnan(cur) and since_acc[t] > cfg.alert_lookback:
            cur = np.nan
        acc_high[t] = cur
    ah = pd.Series(acc_high, index=c.index)
    cross = (c > ah) & (c.shift(1) <= ah.shift(1))
    breakout = _b(cross & (pd.Series(since_acc, index=c.index) <= cfg.alert_lookback) & ~pd.Series(acc, index=c.index)
                  & (v > ta.sma(v, 20)))
    rng = (h - l)
    buy_v = ((c - l) / rng.where(rng != 0) * v).fillna(0)
    sell_v = ((h - c) / rng.where(rng != 0) * v).fillna(0)
    ab, sb = buy_v.rolling(cfg.bt_lookback).mean().shift(1), buy_v.rolling(cfg.bt_lookback).std(ddof=0).shift(1)
    asl, ssl = sell_v.rolling(cfg.bt_lookback).mean().shift(1), sell_v.rolling(cfg.bt_lookback).std(ddof=0).shift(1)
    return pd.DataFrame({
        "accumulating": acc, "distributing": dist, "breakout": breakout,
        "whale_buy": _b(buy_v > ab + sb * cfg.min_whale_sigma), "extreme_buy": _b(buy_v > ab + sb * cfg.extreme_whale_sigma),
        "whale_sell": _b(sell_v > asl + ssl * cfg.min_whale_sigma), "extreme_sell": _b(sell_v > asl + ssl * cfg.extreme_whale_sigma),
        "mfi": m,
    }, index=c.index)


def sell_action(sell_score: float, conflict: bool, suppressed: bool) -> str:
    if suppressed:
        return "SUPPRESSED"
    if conflict:
        return "CONFLICT"
    return "EXIT" if sell_score >= 34 else "REDUCE" if sell_score >= 22 else "TRIM" if sell_score >= 12 else "HOLD"
