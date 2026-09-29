"""Historical validation of the technical model on TASE stocks.

For every stock and every week-end in the sample, record the model's score,
signal and sell action, then measure the forward return in excess of the
TA-125 over 1, 3 and 6 months. Results are reported per score bucket and per
signal, separately for an in-sample and an out-of-sample period, for each
model variant. A variant is only recommended if it is at least as good out
of sample.

Usage:  python -m engine.backtest [--start 2012-01-01] [--split 2020-01-01] [--offline DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .model import ModelConfig, compute_daily

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = {"1m": 21, "3m": 63, "6m": 126}
BUCKETS = [0, 45, 55, 68, 82, 101]
BUCKET_LABELS = ["<45", "45-55", "55-68", "68-82", "82+"]

VARIANTS = {
    "V19 (original)": dict(version="V19"),
    "V20": dict(),
    "V20 + ADX filter": dict(adx_filter=True),
    "V20 + RS percentile": dict(rs_percentile=True),
}


def build_panel(frames, bench, universe, cfg: ModelConfig) -> pd.DataFrame:
    closes = {s.yahoo: frames[s.yahoo]["close"] for s in universe if s.yahoo in frames}
    sectors = {s.yahoo: s.sector for s in universe if s.yahoo in frames}
    comps = D.sector_composites(closes, sectors)
    rs_pct = D.rs_percentiles(closes, bench["close"], cfg.rs_period)
    b = bench["close"]
    rows = []
    for s in universe:
        df = frames.get(s.yahoo)
        if df is None or len(df) < 300:
            continue
        d = compute_daily(df, bench, cfg, sector_close=comps.get(s.yahoo), rs_pct=rs_pct.get(s.yahoo))
        c = d["close"]
        bb = b.reindex(c.index).ffill()
        # entry at next session's close to avoid using the signal bar's own close
        for k, h in HORIZONS.items():
            fwd = c.shift(-(h + 1)) / c.shift(-1) - 1
            bfwd = bb.shift(-(h + 1)) / bb.shift(-1) - 1
            d[f"ex_{k}"] = (fwd - bfwd) * 100
        d = d.iloc[250:]  # indicator warm-up
        wk = d.groupby(pd.Grouper(freq="W-SAT"))
        snap = wk.tail(1).copy()
        snap["signal_week"] = \
            d["signal"].rolling(5, min_periods=1).max().loc[snap.index].values
        snap["ticker"] = s.yahoo
        rows.append(snap[["ticker", "score", "signal_week", "sell_score", "liquid"] + [f"ex_{k}" for k in HORIZONS]])
    return pd.concat(rows) if rows else pd.DataFrame()


def _stats(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    if n == 0:
        return {"n": 0}
    sd = x.std(ddof=1) if n > 1 else np.nan
    return {"n": int(n), "mean": round(float(x.mean()), 2), "median": round(float(x.median()), 2),
            "hit": round(float((x > 0).mean() * 100), 1),
            "t": round(float(x.mean() / (sd / np.sqrt(n))), 2) if n > 1 and sd > 0 else None}


def summarize(panel: pd.DataFrame, split: str) -> dict:
    out = {}
    for period, p in (("in_sample", panel[panel.index < split]), ("out_of_sample", panel[panel.index >= split])):
        res = {}
        p = p.copy()
        p["bucket"] = pd.cut(p["score"], BUCKETS, labels=BUCKET_LABELS, right=False)
        p["sig"] = p["signal_week"].map({0: "WAIT", 1: "BUY", 2: "STRONG BUY"})
        p["sell"] = pd.cut(p["sell_score"], [-1, 12, 22, 34, 1000], labels=["HOLD", "TRIM", "REDUCE", "EXIT"], right=False)
        for k in HORIZONS:
            col = f"ex_{k}"
            res[k] = {
                "by_bucket": {b: _stats(g[col]) for b, g in p.groupby("bucket", observed=False)},
                "by_signal": {sg: _stats(g[col]) for sg, g in p.groupby("sig")},
                "by_sell_action": {sa: _stats(g[col]) for sa, g in p.groupby("sell", observed=False)},
                "all": _stats(p[col]),
            }
            ics = p.groupby(level=0).apply(lambda g: g["score"].rank().corr(g[col].rank()) if g[col].notna().sum() > 8 else np.nan).dropna()
            res[k]["ic_mean"] = round(float(ics.mean()), 4) if len(ics) else None
            res[k]["ic_ir"] = round(float(ics.mean() / ics.std()), 3) if len(ics) > 2 and ics.std() > 0 else None
            res[k]["ic_weeks"] = int(len(ics))
        out[period] = res
    return out


def to_markdown(results: dict, meta: dict) -> str:
    L = ["# Backtest — Analyst Dashboard technical model (TASE)", "",
         f"Sample: {meta['start']} → {meta['end']} · split in/out of sample at {meta['split']} · "
         f"{meta['stocks']} stocks · benchmark {meta['benchmark']} · data: {meta['source']}", "",
         "Excess return = stock return minus TA-125 return over the horizon, entry at the next session's close, in %.",
         "hit = % of observations with positive excess return. t = t-statistic of the mean. IC = mean weekly Spearman rank correlation between score and forward excess return.",
         "", "**Caveats:** the universe is today's list of 62 stocks (survivorship bias), weekly observations overlap for 3m/6m horizons (t-stats overstated), no transaction costs.", ""]
    for name, res in results.items():
        L += [f"## {name}", ""]
        for period in ("in_sample", "out_of_sample"):
            L += [f"### {period.replace('_', ' ')}", ""]
            L += ["| horizon | IC | IC IR | WAIT mean / hit | BUY n · mean / hit | STRONG BUY n · mean / hit | EXIT mean / hit |", "|---|---|---|---|---|---|---|"]
            for k in HORIZONS:
                r = res[period][k]
                bs = r["by_signal"]
                def f(s):
                    return f"{s['n']} · {s['mean']} / {s['hit']}%" if s and s.get("n") else "—"
                ex = r["by_sell_action"].get("EXIT", {})
                L.append(f"| {k} | {r['ic_mean']} | {r['ic_ir']} | {bs.get('WAIT', {}).get('mean', '—')} / {bs.get('WAIT', {}).get('hit', '—')}% "
                         f"| {f(bs.get('BUY', {}))} | {f(bs.get('STRONG BUY', {}))} | {ex.get('mean', '—')} / {ex.get('hit', '—')}% |")
            L += ["", "Score buckets (3m excess return):", "", "| score | n | mean | median | hit |", "|---|---|---|---|---|"]
            for b, s in res[period]["3m"]["by_bucket"].items():
                if s.get("n"):
                    L.append(f"| {b} | {s['n']} | {s['mean']} | {s['median']} | {s['hit']}% |")
            L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--split", default="2020-01-01")
    ap.add_argument("--offline")
    args = ap.parse_args(argv)
    universe = D.load_universe()
    tickers = [s.yahoo for s in universe] + [D.BENCH_TICKER]
    if args.offline:
        frames = {t: pd.read_csv(Path(args.offline) / f"{t}.csv", index_col=0, parse_dates=True)
                  for t in tickers if (Path(args.offline) / f"{t}.csv").exists()}
    else:
        frames = D.download(tickers, args.start)
    bench = frames.pop(D.BENCH_TICKER)
    results, panels = {}, {}
    for name, kw in VARIANTS.items():
        panel = build_panel(frames, bench, universe, ModelConfig(**kw))
        panels[name] = panel
        results[name] = summarize(panel, args.split)
        print(f"{name}: {len(panel)} stock-weeks")
    first = next(iter(panels.values()))
    meta = {"start": str(first.index.min().date()), "end": str(first.index.max().date()), "split": args.split,
            "stocks": int(first["ticker"].nunique()), "benchmark": D.BENCH_TICKER,
            "source": "offline files" if args.offline else "Yahoo Finance"}
    out = ROOT / "results" / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest.json").write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf8")
    (out / "REPORT.md").write_text(to_markdown(results, meta), encoding="utf8")
    print("wrote", out / "REPORT.md")


if __name__ == "__main__":
    main()
