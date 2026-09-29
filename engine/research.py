"""Factor research for the V21 technical model.

Evaluates candidate technical factors on the TASE universe with a strict
in-sample / out-of-sample split. Factors are selected on the in-sample period
only; the out-of-sample period is a blind test.

For each factor and each week: cross-sectional rank IC (Spearman) against the
forward excess return (vs TA-125) over 1, 3 and 6 months, entry at the next
session's close. Only liquid stock-weeks (median 20-day turnover >= ILS 1M) are
used, because advisors can only act on tradable names.

Usage:  python -m engine.research [--start 2012-01-01] [--split 2020-01-01]
Writes results/research/FACTORS.md and factors.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as ta
from .model import ModelConfig, compute_daily

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = {"1m": 21, "3m": 63, "6m": 126}
MIN_TURNOVER = 1_000_000
MIN_NAMES = 15

# a-priori composite from the academic literature (chosen before looking at any TASE data)
PRIOR = {"mom_12_1": 1, "hi52": 1, "low_vol": 1}
# momentum / trend family that passed the in-sample rule (illiquidity is a liquidity premium, not a
# technical signal, and v20_trend is a coarse binary copy of dist_ema200 / ema50_200)
TREND4 = {"mom_6_1": 1, "hi52": 1, "dist_ema200": 1, "ema50_200": 1}


def _panel(frames: dict[str, pd.DataFrame], field: str, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({t: f[field] for t, f in frames.items()}).reindex(index)


def build_factors(frames, bench, universe) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    idx = bench.index
    C = _panel(frames, "close", idx)
    H = _panel(frames, "high", idx)
    L = _panel(frames, "low", idx)
    V = _panel(frames, "volume", idx)
    valid = C.notna()
    C = C.ffill(limit=5)
    R = C.pct_change(fill_method=None)
    b = bench["close"].reindex(idx).ffill()
    br = b.pct_change()
    turnover = (C * V / 100.0).rolling(20, min_periods=10).median()

    F: dict[str, pd.DataFrame] = {}
    F["mom_12_1"] = C.shift(21) / C.shift(252) - 1
    F["mom_6_1"] = C.shift(21) / C.shift(126) - 1
    F["mom_3m"] = C / C.shift(63) - 1
    F["rev_1m"] = -(C / C.shift(21) - 1)
    F["rev_1w"] = -(C / C.shift(5) - 1)
    F["hi52"] = C / H.rolling(252, min_periods=200).max()
    F["low_vol"] = -R.rolling(63, min_periods=50).std()
    F["low_max"] = -R.rolling(21, min_periods=15).max()
    cov = R.rolling(252, min_periods=200).cov(br)
    F["low_beta"] = -(cov.div(br.rolling(252, min_periods=200).var(), axis=0))
    ema200 = C.apply(lambda s: ta.ema(s.dropna(), 200).reindex(s.index))
    ema50 = C.apply(lambda s: ta.ema(s.dropna(), 50).reindex(s.index))
    F["dist_ema200"] = C / ema200 - 1
    F["ema50_200"] = ema50 / ema200 - 1
    F["rsi14"] = C.apply(lambda s: ta.rsi(s.dropna(), 14).reindex(s.index))
    rng = (H - L).where((H - L) > 0)
    mfm = ((C - L) - (H - C)) / rng
    F["cmf20"] = (mfm * V).rolling(20, min_periods=15).sum() / V.rolling(20, min_periods=15).sum()
    F["vol_trend"] = np.log(V.rolling(20, min_periods=15).mean() / V.rolling(120, min_periods=90).mean())
    F["illiquidity"] = (R.abs() / (C * V / 100.0).replace(0, np.nan)).rolling(21, min_periods=15).mean() * 1e6
    # the user's model (V20), score and layers
    cfg = ModelConfig()
    closes = {t: frames[t]["close"] for t in frames}
    sectors = {s.yahoo: s.sector for s in universe if s.yahoo in frames}
    comps = D.sector_composites(closes, sectors)
    parts = {k: {} for k in ("v20_score", "v20_trend", "v20_pullback", "v20_reversal", "v20_volume", "v20_risk", "v20_rs", "v20_sell")}
    for t, df in frames.items():
        if len(df) < 300:
            continue
        d = compute_daily(df, bench, cfg, sector_close=comps.get(t))
        parts["v20_score"][t] = d["score"]
        for k in ("trend", "pullback", "reversal", "volume", "risk", "rs"):
            parts[f"v20_{k}"][t] = d[f"layer_{k}"]
        parts["v20_sell"][t] = -d["sell_score"]
        parts.setdefault("v20_signal_week", {})[t] = d["signal"].rolling(5, min_periods=1).max()
    for k, v in parts.items():
        F[k] = pd.DataFrame(v).reindex(idx)

    fwd = {}
    for k, h in HORIZONS.items():
        fwd[k] = (C.shift(-(h + 1)) / C.shift(-1) - 1).sub(b.shift(-(h + 1)) / b.shift(-1) - 1, axis=0) * 100
    tradable = valid & (turnover >= MIN_TURNOVER)
    return F, fwd, tradable


def weekly_dates(idx: pd.Index) -> pd.Index:
    s = pd.Series(idx, index=idx)
    return pd.Index(s.groupby(pd.Grouper(freq="W-SAT")).last().dropna().values)


def rank_ic(f: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame, dates) -> pd.Series:
    out = {}
    for d in dates:
        a, r, m = f.loc[d], y.loc[d], mask.loc[d]
        ok = m & a.notna() & r.notna()
        if ok.sum() < MIN_NAMES:
            continue
        out[d] = a[ok].rank().corr(r[ok].rank())
    return pd.Series(out, dtype=float)


def quintile_spread(f, y, mask, dates) -> pd.Series:
    out = {}
    for d in dates:
        a, r, m = f.loc[d], y.loc[d], mask.loc[d]
        ok = m & a.notna() & r.notna()
        if ok.sum() < MIN_NAMES:
            continue
        q = a[ok].rank(pct=True)
        out[d] = r[ok][q > 0.8].mean() - r[ok][q <= 0.2].mean()
    return pd.Series(out, dtype=float)


def summarize_ic(ic: pd.Series, h_days: int) -> dict:
    if len(ic) < 10:
        return {"n": len(ic)}
    overlap = max(1.0, h_days / 5.0)   # weekly samples of an h-day return overlap
    n_eff = len(ic) / overlap
    m, s = ic.mean(), ic.std()
    return {"n": int(len(ic)), "ic": round(float(m), 4), "ir": round(float(m / s), 3) if s > 0 else None,
            "t": round(float(m / s * np.sqrt(n_eff)), 2) if s > 0 else None, "hit": round(float((ic > 0).mean() * 100), 1)}


def composite(F: dict, names_signs: dict, mask) -> pd.DataFrame:
    ranks = []
    for n, sgn in names_signs.items():
        r = (F[n] * sgn).where(mask).rank(axis=1, pct=True)
        ranks.append(r)
    return sum(ranks) / len(ranks)


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
    F, fwd, tradable = build_factors(frames, bench, universe)
    dates = weekly_dates(bench.index)
    dates = dates[dates >= bench.index[0] + pd.Timedelta(days=400)]
    split = pd.Timestamp(args.split)
    periods = {"IS": dates[dates < split], "OOS": dates[dates >= split],
               "OOS_2020_22": dates[(dates >= split) & (dates < "2023-01-01")], "OOS_2023_26": dates[dates >= "2023-01-01"]}

    def evaluate(f):
        res = {}
        for pname, pd_ in periods.items():
            res[pname] = {}
            for k, h in HORIZONS.items():
                ic = rank_ic(f, fwd[k], tradable, pd_)
                qs = quintile_spread(f, fwd[k], tradable, pd_)
                res[pname][k] = {**summarize_ic(ic, h), "q5_q1": round(float(qs.mean()), 2) if len(qs) else None}
        return res

    results = {n: evaluate(f) for n, f in F.items()}
    # selection on IS only: 3m IC with |t| >= 2, sign from IS
    selected = {n: int(np.sign(r["IS"]["3m"].get("ic", 0))) for n, r in results.items()
                if r["IS"]["3m"].get("t") is not None and abs(r["IS"]["3m"]["t"]) >= 2.0 and not n.startswith("v20_score")}
    comps = {"PRIOR (literature: mom 12-1, 52w high, low vol)": PRIOR}
    if selected:
        comps["SELECTED (IS |t|>=2, equal weight)"] = selected
    comps["TREND4 (momentum/trend family)"] = TREND4
    comp_frames = {name: composite(F, ns, tradable) for name, ns in comps.items()}
    comp_results = {name: evaluate(cf) for name, cf in comp_frames.items()}

    def quintiles(f, k, dts):
        rows = []
        for d in dts:
            a, r, m = f.loc[d], fwd[k].loc[d], tradable.loc[d]
            ok = m & a.notna() & r.notna()
            if ok.sum() < MIN_NAMES:
                continue
            q = np.ceil(a[ok].rank(pct=True) * 5).clip(1, 5)
            rows.append(r[ok].groupby(q).mean())
        t = pd.DataFrame(rows)
        return {int(c): round(float(t[c].mean()), 2) for c in t.columns}

    quint = {name: {p: {k: quintiles(cf, k, periods[p]) for k in ("1m", "3m", "6m")} for p in ("IS", "OOS")}
             for name, cf in comp_frames.items()}

    # does the V20 setup add value as an entry timer on top of a strong trend rating?
    sig = F["v20_signal_week"]
    trend = comp_frames["TREND4 (momentum/trend family)"].rank(axis=1, pct=True)
    timing = {}
    for p in ("IS", "OOS"):
        for k in ("1m", "3m"):
            cells = {"top40 & V20 BUY": [], "top40 & no signal": [], "bottom60 & V20 BUY": []}
            for d in periods[p]:
                r, m = fwd[k].loc[d], tradable.loc[d]
                hi = trend.loc[d] > 0.6
                s_ = sig.loc[d].reindex(r.index).fillna(0) > 0
                for name, mask_ in (("top40 & V20 BUY", hi & s_), ("top40 & no signal", hi & ~s_), ("bottom60 & V20 BUY", ~hi & s_)):
                    v = r[mask_ & m & r.notna()]
                    cells[name].extend(v.tolist())
            timing[f"{p} {k}"] = {n: {"n": len(v), "mean": round(float(np.mean(v)), 2) if v else None,
                                     "hit": round(float(np.mean(np.array(v) > 0) * 100), 1) if v else None}
                                  for n, v in cells.items()}

    out = ROOT / "results" / "research"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"start": str(dates[0].date()), "end": str(dates[-1].date()), "split": args.split,
            "stocks": len(frames), "min_turnover_ils": MIN_TURNOVER, "selected": selected}
    (out / "factors.json").write_text(json.dumps({"meta": meta, "factors": results, "composites": comp_results,
                                                  "composite_defs": comps, "quintiles": quint, "timing": timing}, indent=1, ensure_ascii=False), encoding="utf8")
    L = ["# Factor research — TASE technical model V21", "",
         f"Weeks {meta['start']} → {meta['end']} · in-sample before {args.split}, out-of-sample after · {meta['stocks']} stocks · "
         f"tradable filter: median turnover ≥ ₪{MIN_TURNOVER:,}", "",
         "IC = mean weekly rank correlation of the factor with forward excess return vs TA-125. t uses overlap-adjusted sample size. "
         "q5_q1 = mean excess return of the top 20% minus the bottom 20% (in %).", "",
         f"Selected on in-sample only (3m |t| ≥ 2): {selected}", ""]

    def table(title, res):
        rows = [f"## {title}", "", "| factor | IS 3m IC (t) | IS 3m q5-q1 | OOS 1m IC (t) | OOS 3m IC (t) | OOS 3m q5-q1 | OOS 6m IC (t) | 2020-22 3m IC | 2023-26 3m IC |",
                "|---|---|---|---|---|---|---|---|---|"]
        for n, r in res.items():
            def f(p, k):
                x = r[p][k]
                return f"{x.get('ic')} ({x.get('t')})" if x.get("ic") is not None else "—"
            rows.append(f"| {n} | {f('IS','3m')} | {r['IS']['3m'].get('q5_q1')} | {f('OOS','1m')} | {f('OOS','3m')} | "
                        f"{r['OOS']['3m'].get('q5_q1')} | {f('OOS','6m')} | {r['OOS_2020_22']['3m'].get('ic')} | {r['OOS_2023_26']['3m'].get('ic')} |")
        return rows + [""]

    ordered = dict(sorted(results.items(), key=lambda kv: -(kv[1]["OOS"]["3m"].get("ic") or -9)))
    L += table("Composites", comp_results)
    L += ["## Composite quintiles (mean excess return %, Q1 = weakest, Q5 = strongest)", "",
          "| composite | period | horizon | Q1 | Q2 | Q3 | Q4 | Q5 |", "|---|---|---|---|---|---|---|---|"]
    for name, qp in quint.items():
        for p in ("IS", "OOS"):
            for k in ("1m", "3m", "6m"):
                q = qp[p][k]
                L.append(f"| {name} | {p} | {k} | " + " | ".join(str(q.get(i, "—")) for i in range(1, 6)) + " |")
    L += ["", "## V20 setup as an entry timer on top of TREND4", "", "| sample | group | n | mean excess % | hit % |", "|---|---|---|---|---|"]
    for key, cells in timing.items():
        for n, v in cells.items():
            L.append(f"| {key} | {n} | {v['n']} | {v['mean']} | {v['hit']} |")
    L += [""] + table("Single factors (sorted by OOS 3m IC)", ordered)
    (out / "FACTORS.md").write_text("\n".join(L), encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
