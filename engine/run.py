"""Weekly scoring run: downloads data, scores every stock, writes results/latest.json.

Usage:  python -m engine.run [--offline DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import v22 as V22
from .model import ModelConfig, compute_daily, compute_weekly_signal, compute_whale, sell_action

ROOT = Path(__file__).resolve().parents[1]
SIGNAL_TEXT = {2: "STRONG BUY", 1: "BUY", 0: "WAIT"}
RECENT_SESSIONS = 5  # one trading week


def _num(x, nd=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def score_universe(frames: dict[str, pd.DataFrame], bench: pd.DataFrame, universe, cfg: ModelConfig) -> dict:
    ref_last = bench.index[-1]
    closes = {s.yahoo: frames[s.yahoo]["close"] for s in universe if s.yahoo in frames}
    sectors = {s.yahoo: s.sector for s in universe if s.yahoo in frames}
    composites = D.sector_composites(closes, sectors)
    rs_pct = D.rs_percentiles(closes, bench["close"], cfg.rs_period) if closes else {}

    good = {t: f for t, f in frames.items() if D.check_quality(f, ref_last).ok}
    sc22, sret22, bret22, ex22, turn22, elig22 = V22.panel(good, bench["close"], bench.index)
    d22 = bench.index[-1]

    records, errors = [], []
    for s in universe:
        df = frames.get(s.yahoo)
        q = D.check_quality(df, ref_last)
        rec = {"tase_id": s.tase_id, "name": s.name, "ticker": s.yahoo, "sector": s.sector,
               "data_ok": q.ok, "data_issues": q.issues}
        if not q.ok:
            errors.append({"ticker": s.yahoo, "name": s.name, "issues": q.issues})
            records.append(rec)
            continue
        d = compute_daily(df, bench, cfg, sector_close=composites.get(s.yahoo), rs_pct=rs_pct.get(s.yahoo))
        wk = compute_weekly_signal(df, cfg)
        wh = compute_whale(df, cfg)
        last = d.iloc[-1]
        recent = d["signal"].tail(RECENT_SESSIONS)
        wk_done = wk[wk.index <= d.index[-1] + pd.Timedelta(days=1)]
        wlast = wk_done.iloc[-1] if len(wk_done) else None
        conflict_recent = bool(d["conflict"].tail(RECENT_SESSIONS).any())
        sig_now = int(last["signal"])
        s22 = sc22.at[d22, s.yahoo] if s.yahoo in sc22.columns else np.nan
        rec["score"] = _num(s22, 0)
        sell_now = float(last["sell_score"])
        rec["rating"], rec["rating_note"] = V22.rating(s22, sell_now)
        if np.isnan(s22):
            rec["rating_note"] = ("illiquid: median turnover below ILS 1M"
                                  if s.yahoo in turn22.columns and turn22.at[d22, s.yahoo] < V22.MIN_TURNOVER_ILS
                                  else "insufficient history")
        rec["rs"] = {"months": 6,
                     "stock_pct": _num(sret22.at[d22, s.yahoo] * 100, 1) if s.yahoo in sret22.columns else None,
                     "index_pct": _num(bret22.at[d22] * 100, 1),
                     "excess_pct": _num(ex22.at[d22, s.yahoo] * 100, 1) if s.yahoo in ex22.columns else None}
        rec["timing"] = {
            "score": _num(last["score"], 1),
            "score_week_max": _num(d["score"].tail(RECENT_SESSIONS).max(), 1),
            "signal": "CONFLICT" if bool(last["conflict"]) else SIGNAL_TEXT[sig_now],
            "signal_week": SIGNAL_TEXT[int(recent.max())] if not (conflict_recent and recent.max() == 0) else "CONFLICT",
            "weekly_signal": SIGNAL_TEXT[int(wlast["weekly_signal"])] if wlast is not None else None,
            "weekly_score": _num(wlast["weekly_score"], 1) if wlast is not None else None,
            "sell_score": _num(last["sell_score"], 0),
            "sell_action": sell_action(float(last["sell_score"]), bool(last["conflict"]), bool(last["sell_suppressed"])),
            "layers": {k: _num(last[f"layer_{k}"], 1) for k in ("trend", "pullback", "reversal", "volume", "risk", "rs")},
            "checks": {k: bool(last[k]) for k in ("market_safe", "sector_safe", "trend_ready", "pullback_ready",
                                                  "reversal_ready", "volume_ready", "risk_ready", "setup_ready", "liquid")},
        }
        rec["flag"] = V22.flag(rec["timing"]["signal_week"], rec["timing"]["sell_action"])
        rec.update({
            "asof": str(d.index[-1].date()),
            "metrics": {
                "close": _num(last["close"] / 100, 2),  # agorot -> ILS
                "chg_1w_pct": _num((d["close"].iloc[-1] / d["close"].iloc[-6] - 1) * 100, 1) if len(d) > 6 else None,
                "chg_1m_pct": _num((d["close"].iloc[-1] / d["close"].iloc[-22] - 1) * 100, 1) if len(d) > 22 else None,
                "rsi": _num(last["rsi"], 1), "adx": _num(last["adx"], 1), "atr_pct": _num(last["atr_pct"], 2),
                "dist_ema200_pct": _num(last["dist_ema200"], 1), "dist_from_high_pct": _num(last["dist_from_high_pct"], 1),
                "cmf": _num(last["cmf"], 3), "rvol": _num(last["rvol"], 2), "beta": _num(last["beta"], 2),
                "rs_3m_pct": _num(last["rs_value"] * 100, 1),
                "rs_percentile": _num(rs_pct[s.yahoo].iloc[-1] * 100, 0) if s.yahoo in rs_pct else None,
                "stop_ref": _num(last["stop_reference"] / 100, 2), "stop_risk_pct": _num(last["stop_risk_pct"], 1),
                "median_turnover_ils": _num(last["median_turnover"], 0),
                "reversal_signals": int(last["reversal_count"]), "pattern": str(last["pattern"]),
            },
            "whale": {
                "breakout_week": bool(wh["breakout"].tail(RECENT_SESSIONS).any()),
                "accumulating": bool(wh["accumulating"].iloc[-1]),
                "distributing": bool(wh["distributing"].iloc[-1]),
                "whale_buy_week": bool(wh["whale_buy"].tail(RECENT_SESSIONS).any()),
                "whale_sell_week": bool(wh["whale_sell"].tail(RECENT_SESSIONS).any()),
            },
        })
        records.append(rec)
    return {"records": records, "errors": errors, "market_date": str(ref_last.date())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", help="directory with <ticker>.csv files instead of downloading")
    ap.add_argument("--years", type=int, default=6)
    args = ap.parse_args(argv)
    cfg = ModelConfig()
    universe = D.load_universe()
    tickers = [s.yahoo for s in universe] + [D.BENCH_TICKER]
    if args.offline:
        frames = {t: pd.read_csv(Path(args.offline) / f"{t}.csv", index_col=0, parse_dates=True)
                  for t in tickers if (Path(args.offline) / f"{t}.csv").exists()}
    else:
        start = (pd.Timestamp.today() - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
        frames = D.download(tickers, start)
    bench = frames.pop(D.BENCH_TICKER, None)
    if bench is None or bench.empty:
        raise SystemExit("benchmark TA-125 download failed; nothing written")
    res = score_universe(frames, bench, universe, cfg)
    ok = [r for r in res["records"] if r.get("data_ok")]
    out = {
        "model": "V22 strength vs TA-125 over 6 months + Analyst Dashboard V20 flags", "model_version": "V22",
        "rating_rule": {"BUY": f"score >= {V22.BUY_AT:g} and your model not at EXIT", "HOLD": f"{V22.SELL_BELOW:g} <= score < {V22.BUY_AT:g}",
                        "SELL": f"score < {V22.SELL_BELOW:g}"},
        "flag_rule": {"STRONG BUY/BUY": "your model's entry signal in the last trading week",
                      "STRONG SELL/SELL": "your model's sell action today: EXIT / REDUCE"},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market_date": res["market_date"], "data_source": "Yahoo Finance (adjusted daily OHLCV)",
        "benchmark": D.BENCH_TICKER, "timing_thresholds": {"buy": cfg.buy_threshold, "strong_buy": cfg.strong_buy_threshold},
        "count": len(res["records"]), "scored": len(ok), "errors": res["errors"],
        "records": res["records"],
    }
    rdir = ROOT / "results"
    (rdir / "history").mkdir(parents=True, exist_ok=True)
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    (rdir / "latest.json").write_text(txt, encoding="utf8")
    (rdir / "history" / f"{res['market_date']}.json").write_text(txt, encoding="utf8")
    rows = [{"tase_id": r["tase_id"], "name": r["name"], "ticker": r["ticker"], "score": r.get("score"),
             "rating": r.get("rating"), "flag": r.get("flag"), "timing_signal_week": (r.get("timing") or {}).get("signal_week"),
             "sell_action": (r.get("timing") or {}).get("sell_action"), "data_ok": r["data_ok"]} for r in res["records"]]
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(rdir / "latest.csv", index=False, encoding="utf-8-sig")
    print(f"scored {len(ok)}/{len(res['records'])} stocks, market date {res['market_date']}")
    for e in res["errors"]:
        print("  data problem:", e["ticker"], e["name"], "; ".join(e["issues"]))


if __name__ == "__main__":
    main()
