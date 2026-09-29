"""Relative-strength score vs TA-125: which look-back works best, and which BUY/HOLD/SELL rules?

Part 1 - look-back. Candidate scores (all fixed in advance): the stock's return minus TA-125's return over
1, 3, 6, 12 months, the 6-1 and 12-1 variants (skip the last month) and an IBD-style blend
(2x3m + 6m + 9m + 12m). Each is ranked 0-100 among tradable names every day. The winner is chosen on
the in-sample years only (highest 3-month rank IC, 2013-2019); 2020 onward is the blind test.

Part 2 - signals on the chosen score, trade simulation (same engine and costs as engine/combo.py):
  exit threshold 20 / 35 / 50, and two ways of using the user's model (V20):
  (a) veto: no new BUY while the user's model shows EXIT (sell score >= 34)
  (b) early entry: also BUY at score >= 60 when the user's model gives a BUY signal that week.

Usage: python -m engine.rs_study [--start 2012-01-01] [--split 2020-01-01]
Writes results/research/RS.md and rs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .combo import COST, V20_EXIT, portfolio, stats, trade_stats
from .research import (HORIZONS, TREND4, build_factors, composite, quintile_spread, rank_ic, summarize_ic, weekly_dates)

ROOT = Path(__file__).resolve().parents[1]
LOOKBACKS = {"rs_1m": (21, 0), "rs_3m": (63, 0), "rs_6m": (126, 0), "rs_12m": (252, 0),
             "rs_6_1": (126, 21), "rs_12_1": (252, 21)}


def rs_scores(C: pd.DataFrame, b: pd.Series, tradable: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def ex(n, skip=0):
        s = (C.shift(skip) / C.shift(n)) .div(b.shift(skip) / b.shift(n), axis=0) - 1
        return s
    raw = {k: ex(n, s) for k, (n, s) in LOOKBACKS.items()}
    raw["rs_ibd"] = 2 * ex(63) + ex(126) + ex(189) + ex(252)
    return {k: v.where(tradable).rank(axis=1, pct=True) * 100 for k, v in raw.items()}, raw


def simulate(score, sig, sell, tradable, close, weekly, enter=80, exit_below=50, veto=False, early=False) -> list[dict]:
    trades, n = [], len(close.index)
    for t in close.columns:
        c, sc = close[t].to_numpy(), score[t].to_numpy()
        sg = pd.Series(sig[t].to_numpy()).rolling(5, min_periods=1).max().to_numpy()
        se, tr = sell[t].to_numpy(), tradable[t].to_numpy()
        pos, entry_i, pending = False, None, None
        for i in range(n - 1):
            if pending == "in":
                pos, entry_i = True, i
            elif pending == "out":
                trades.append({"ticker": t, "entry_i": entry_i, "exit_i": i})
                pos = False
            pending = None
            if np.isnan(c[i]) or not weekly[i]:
                continue
            if not pos:
                ok = tr[i] and (sc[i] >= enter or (early and sc[i] >= 60 and sg[i] > 0))
                if ok and veto and se[i] >= V20_EXIT:
                    ok = False
                if ok:
                    pending = "in"
            elif not (sc[i] >= exit_below):
                pending = "out"
        if pos:
            trades.append({"ticker": t, "entry_i": entry_i, "exit_i": n - 1})
    return trades


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
    X: dict = {}
    F, fwd, tradable = build_factors(frames, bench, universe, extras=X)
    C, b = X["close"], X["bench"]
    for k in ("signal", "sell_score"):
        X[k] = X[k].reindex(columns=C.columns)
    S, raw = rs_scores(C, b, tradable)
    S["v21_reference"] = composite(F, TREND4, tradable).rank(axis=1, pct=True) * 100   # today's V21, for comparison only

    dates = weekly_dates(C.index)
    start = C.index[0] + pd.Timedelta(days=400)
    dates = dates[dates >= start]
    split = pd.Timestamp(args.split)
    wp = {"IS": dates[dates < split], "OOS": dates[dates >= split],
          "2020-22": dates[(dates >= split) & (dates < "2023-01-01")], "2023-26": dates[dates >= "2023-01-01"]}

    # ---- part 1: look-back
    ic = {}
    for k, s in S.items():
        ic[k] = {}
        for p, ds in wp.items():
            ic[k][p] = {}
            for h, hd in HORIZONS.items():
                r = rank_ic(s, fwd[h], tradable, ds)
                q = quintile_spread(s, fwd[h], tradable, ds)
                ic[k][p][h] = {**summarize_ic(r, hd), "q5_q1": round(float(q.mean()), 2) if len(q) else None}
    best = max([k for k in S if k != "v21_reference"], key=lambda k: ic[k]["IS"]["3m"].get("ic") or -9)

    def quint(s, h, ds):
        rows = []
        for d in ds:
            a, r, m = s.loc[d], fwd[h].loc[d], tradable.loc[d]
            ok = m & a.notna() & r.notna()
            if ok.sum() < 15:
                continue
            rows.append(r[ok].groupby(np.ceil(a[ok].rank(pct=True) * 5).clip(1, 5)).mean())
        t = pd.DataFrame(rows)
        return {int(c): round(float(t[c].mean()), 2) for c in t.columns}
    quintiles = {p: {h: quint(S[best], h, wp[p]) for h in HORIZONS} for p in ("IS", "OOS")}

    # ---- part 2: signals on the chosen score
    score = S[best].copy()
    score.loc[C.index < start] = np.nan
    sig = X["signal"].copy()
    sig.loc[C.index < start] = 0
    wk = set(weekly_dates(C.index))
    weekly = np.array([d in wk for d in C.index])
    bret = b.pct_change().fillna(0.0)
    periods = {"IS 2013-2019": (start, split), "OOS 2020-2026": (split, C.index[-1] + pd.Timedelta(days=1)),
               "2020-2022": (split, pd.Timestamp("2023-01-01")), "2023-2026": (pd.Timestamp("2023-01-01"), C.index[-1] + pd.Timedelta(days=1))}
    variants = {
        "exit<20": dict(exit_below=20), "exit<35": dict(exit_below=35), "exit<50": dict(exit_below=50),
        "exit<50 + veto (no BUY while your model says EXIT)": dict(exit_below=50, veto=True),
        "exit<50 + early BUY (score>=60 + your BUY signal)": dict(exit_below=50, early=True),
    }
    sim = {}
    for name, kw in variants.items():
        trades = simulate(score, sig, X["sell_score"], tradable, C, weekly, **kw)
        ret, npos = portfolio(trades, C, b)
        sim[name] = {p: {**stats(ret[(ret.index >= lo) & (ret.index < hi)], bret[(ret.index >= lo) & (ret.index < hi)],
                                  npos[(ret.index >= lo) & (ret.index < hi)]),
                         "trades": trade_stats(trades, C, b, lo, hi)} for p, (lo, hi) in periods.items()}
    bst = {p: stats(bret[(bret.index >= lo) & (bret.index < hi)], bret[(bret.index >= lo) & (bret.index < hi)],
                    pd.Series(1, index=bret.index)[(bret.index >= lo) & (bret.index < hi)]) for p, (lo, hi) in periods.items()}

    out = ROOT / "results" / "research"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rs.json").write_text(json.dumps({"best": best, "ic": ic, "quintiles": quintiles, "signals": sim, "benchmark": bst},
                                            indent=1, ensure_ascii=False), encoding="utf8")
    L = ["# Relative strength vs TA-125 — look-back and signal study", "",
         f"{len(frames)} stocks · weekly · tradable names only · cost {COST*100:.2f}%/side · chosen on IS 3m IC: **{best}**", "",
         "## Part 1 — which look-back predicts best (rank IC, t in brackets)", "",
         "| score | IS 1m | IS 3m | IS 6m | OOS 1m | OOS 3m | OOS 6m | OOS 3m q5-q1 % | 2020-22 3m | 2023-26 3m |", "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: f"{x.get('ic')} ({x.get('t')})"
    for k in S:
        r = ic[k]
        L.append(f"| {k} | {f(r['IS']['1m'])} | {f(r['IS']['3m'])} | {f(r['IS']['6m'])} | {f(r['OOS']['1m'])} | {f(r['OOS']['3m'])} | "
                 f"{f(r['OOS']['6m'])} | {r['OOS']['3m'].get('q5_q1')} | {r['2020-22']['3m'].get('ic')} | {r['2023-26']['3m'].get('ic')} |")
    L += ["", f"## Quintiles of {best} (mean excess return vs TA-125 %, Q5 = strongest)", "", "| period | horizon | Q1 | Q2 | Q3 | Q4 | Q5 |", "|---|---|---|---|---|---|---|"]
    for p, hq in quintiles.items():
        for h, q in hq.items():
            L.append(f"| {p} | {h} | " + " | ".join(str(q.get(i, "–")) for i in range(1, 6)) + " |")
    L += ["", f"## Part 2 — signals on {best} (BUY at score >= 80)", ""]
    for p in periods:
        L += [f"### {p}", "", "| rule | CAGR % | excess % | max DD % | Sharpe | avg pos | trades | win % | avg trade % | days |", "|---|---|---|---|---|---|---|---|---|---|",
              f"| TA-125 | {bst[p].get('cagr')} | 0 | {bst[p].get('maxdd')} | {bst[p].get('sharpe')} | – | – | – | – | – |"]
        for name, r in sim.items():
            s, t = r[p], r[p]["trades"]
            L.append(f"| {name} | {s.get('cagr')} | {s.get('excess')} | {s.get('maxdd')} | {s.get('sharpe')} | {s.get('avg_pos')} | {t.get('n')} | {t.get('win')} | {t.get('avg')} | {t.get('days')} |")
        L.append("")
    (out / "RS.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
