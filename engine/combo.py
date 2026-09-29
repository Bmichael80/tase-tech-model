"""Which combination of V21 (ranking) and the user's model (V20 timing, stop, exit) adds the most value?

Trade simulation on the TASE universe. All rules are fixed in advance (below) and evaluated on the
in-sample years (2013-2019) and the blind out-of-sample years (2020 onward). A rule only counts as an
improvement if it beats plain V21 in BOTH periods.

Execution: every signal is computed on a day's close and acted on at the NEXT session's close.
Weekly rules are checked on the last trading day of each week; stops and the V20 EXIT are checked daily.
Only tradable names (median 20-day turnover >= ILS 1M) can be entered. Cost: 0.15% per side.
Portfolio: equal weight across open positions, fully invested when at least one position is open, cash (0%)
otherwise. Benchmark: TA-125.

Usage: python -m engine.combo [--start 2012-01-01] [--split 2020-01-01]
Writes results/research/COMBO.md and combo.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .research import TREND4, build_factors, composite, weekly_dates

ROOT = Path(__file__).resolve().parents[1]
COST = 0.0015
ENTER, EXIT, WIDE = 80, 50, 60
V20_EXIT = 34          # the user's model: sell score >= 34 = EXIT
V20_MAX_HOLD = 63      # the user's model alone has no time exit; 3 months is used so trades close

STRATEGIES = {
    "A": {"label": "V21 בלבד", "enter": ENTER},
    "B": {"label": "V21 + כניסה רק עם אות שלך", "enter": ENTER, "need_signal": True},
    "B2": {"label": "V21 (60+) + כניסה רק עם אות שלך", "enter": WIDE, "need_signal": True},
    "C": {"label": "V21 + סטופ שלך", "enter": ENTER, "stop": True},
    "D": {"label": "V21 + יציאה ב-EXIT שלך", "enter": ENTER, "v20_exit": True},
    "E": {"label": "V21 + סטופ + EXIT שלך", "enter": ENTER, "stop": True, "v20_exit": True},
    "F": {"label": "שילוב מלא: V21 (60+) + אות + סטופ + EXIT", "enter": WIDE, "need_signal": True, "stop": True, "v20_exit": True},
    "G": {"label": "ציון משולב 70% V21 / 30% שלך", "enter": ENTER, "blend": True},
    "H": {"label": "המודל שלך לבד (אות, סטופ, EXIT)", "v20_only": True},
}


def simulate(rule, v21, blend, sig, sell, stop_ref, tradable, close, weekly) -> list[dict]:
    """Returns trades: {ticker, entry_i, exit_i}. Indices refer to the daily index."""
    score = blend if rule.get("blend") else v21
    trades = []
    n = len(close.index)
    for t in close.columns:
        c = close[t].to_numpy()
        sc, sg = score[t].to_numpy(), sig[t].to_numpy()
        se, st, tr = sell[t].to_numpy(), stop_ref[t].to_numpy(), tradable[t].to_numpy()
        sg_week = pd.Series(sg).rolling(5, min_periods=1).max().to_numpy()
        pos, entry_i, stop, pending = False, None, np.nan, None
        for i in range(n - 1):
            if pending is not None:              # act at this close on yesterday's decision
                if pending == "in":
                    pos, entry_i, stop = True, i, st[i - 1] if rule.get("stop") or rule.get("v20_only") else np.nan
                else:
                    trades.append({"ticker": t, "entry_i": entry_i, "exit_i": i})
                    pos = False
                pending = None
            if np.isnan(c[i]):
                continue
            if not pos:
                if rule.get("v20_only"):
                    if sg[i] > 0 and tr[i]:
                        pending = "in"
                elif weekly[i] and tr[i] and sc[i] >= rule["enter"] and (not rule.get("need_signal") or sg_week[i] > 0):
                    pending = "in"
                continue
            # in a position: daily exits first
            if (rule.get("stop") or rule.get("v20_only")) and not np.isnan(st[i]):
                stop = st[i] if np.isnan(stop) else max(stop, st[i])       # the user's stop, trailed upward only
            if (rule.get("stop") or rule.get("v20_only")) and not np.isnan(stop) and c[i] < stop:
                pending = "out"
            elif (rule.get("v20_exit") or rule.get("v20_only")) and se[i] >= V20_EXIT:
                pending = "out"
            elif rule.get("v20_only") and i - entry_i >= V20_MAX_HOLD:
                pending = "out"
            elif not rule.get("v20_only") and weekly[i] and not (sc[i] >= EXIT):
                pending = "out"
        if pos:
            trades.append({"ticker": t, "entry_i": entry_i, "exit_i": n - 1})
    return trades


def portfolio(trades, close, bench_close) -> tuple[pd.Series, pd.Series]:
    idx = close.index
    R = close.pct_change(fill_method=None).fillna(0.0).to_numpy()
    cols = {t: j for j, t in enumerate(close.columns)}
    held = np.zeros_like(R, dtype=bool)
    events = np.zeros(len(idx))
    for tr in trades:
        j = cols[tr["ticker"]]
        held[tr["entry_i"] + 1: tr["exit_i"] + 1, j] = True   # earns returns after the entry close, through the exit close
        events[tr["entry_i"]] += 1
        events[tr["exit_i"]] += 1
    n_held = held.sum(axis=1)
    gross = np.where(n_held > 0, (R * held).sum(axis=1) / np.maximum(n_held, 1), 0.0)
    n_after = np.maximum(n_held, 1)
    net = gross - COST * events / n_after
    return pd.Series(net, index=idx), pd.Series(n_held, index=idx)


def stats(ret: pd.Series, bret: pd.Series, npos: pd.Series) -> dict:
    if len(ret) < 50:
        return {}
    yrs = len(ret) / 252
    eq = (1 + ret).cumprod()
    beq = (1 + bret).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    bcagr = beq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    vol = ret.std() * np.sqrt(252)
    ex = ret - bret
    te = ex.std() * np.sqrt(252)
    return {"cagr": round(cagr * 100, 1), "excess": round((cagr - bcagr) * 100, 1), "maxdd": round(dd * 100, 1),
            "vol": round(vol * 100, 1), "sharpe": round(ret.mean() / ret.std() * np.sqrt(252), 2) if ret.std() > 0 else None,
            "info": round(ex.mean() * 252 / te, 2) if te > 0 else None,
            "invested": round(float((npos > 0).mean() * 100), 0), "avg_pos": round(float(npos[npos > 0].mean()), 1) if (npos > 0).any() else 0}


def trade_stats(trades, close, bench_close, lo, hi) -> dict:
    idx = close.index
    rows = []
    for tr in trades:
        e, x = tr["entry_i"], tr["exit_i"]
        if not (lo <= idx[e] < hi):
            continue
        c = close[tr["ticker"]]
        r = c.iloc[x] / c.iloc[e] - 1 - 2 * COST
        b = bench_close.iloc[x] / bench_close.iloc[e] - 1
        rows.append((r, r - b, x - e))
    if not rows:
        return {"n": 0}
    a = np.array(rows)
    losses = a[:, 0][a[:, 0] < 0]
    return {"n": len(a), "win": round(float((a[:, 0] > 0).mean() * 100), 1), "beat": round(float((a[:, 1] > 0).mean() * 100), 1),
            "avg": round(float(a[:, 0].mean() * 100), 2), "avg_excess": round(float(a[:, 1].mean() * 100), 2),
            "worst": round(float(a[:, 0].min() * 100), 1), "avg_loss": round(float(losses.mean() * 100), 2) if len(losses) else 0.0,
            "days": round(float(a[:, 2].mean()), 0)}


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
    F, _, tradable = build_factors(frames, bench, universe, extras=X)
    close, bclose = X["close"], X["bench"]
    for k in ("signal", "sell_score", "stop_reference"):     # names with too little history have no V20 series
        X[k] = X[k].reindex(columns=close.columns)
    # V21 exactly as in production: TREND4 composite re-ranked 0-100 among tradable names
    v21 = composite(F, TREND4, tradable).rank(axis=1, pct=True) * 100
    v20pct = F["v20_score"].where(tradable).rank(axis=1, pct=True) * 100
    blend = (0.7 * v21 + 0.3 * v20pct).rank(axis=1, pct=True) * 100
    wk = set(weekly_dates(close.index))
    weekly = np.array([d in wk for d in close.index])
    start = close.index[0] + pd.Timedelta(days=400)
    split = pd.Timestamp(args.split)
    periods = {"IS 2013-2019": (start, split), "OOS 2020-2026": (split, close.index[-1] + pd.Timedelta(days=1)),
               "2020-2022": (split, pd.Timestamp("2023-01-01")), "2023-2026": (pd.Timestamp("2023-01-01"), close.index[-1] + pd.Timedelta(days=1))}
    bret = bclose.pct_change().fillna(0.0)

    res, yearly = {}, {}
    # before `start` no rule may trade: mask scores so warm-up rows cannot trigger entries
    early = close.index < start
    v21.loc[early] = np.nan
    blend.loc[early] = np.nan
    sig = X["signal"].copy()
    sig.loc[early] = 0
    for key, rule in STRATEGIES.items():
        trades = simulate(rule, v21, blend, sig, X["sell_score"], X["stop_reference"], tradable, close, weekly)
        ret, npos = portfolio(trades, close, bclose)
        res[key] = {"label": rule["label"], "periods": {}}
        for pname, (lo, hi) in periods.items():
            m = (ret.index >= lo) & (ret.index < hi)
            res[key]["periods"][pname] = {**stats(ret[m], bret[m], npos[m]), "trades": trade_stats(trades, close, bclose, lo, hi)}
        oos = ret[ret.index >= split]
        yearly[key] = {int(y): round(float(((1 + g).prod() - (1 + bret[g.index]).prod()) * 100), 1) for y, g in oos.groupby(oos.index.year)}
    bstats = {p: stats(bret[(bret.index >= lo) & (bret.index < hi)], bret[(bret.index >= lo) & (bret.index < hi)],
                       pd.Series(1, index=bret.index)[(bret.index >= lo) & (bret.index < hi)]) for p, (lo, hi) in periods.items()}

    out = ROOT / "results" / "research"
    out.mkdir(parents=True, exist_ok=True)
    (out / "combo.json").write_text(json.dumps({"strategies": res, "benchmark": bstats, "yearly_excess_oos": yearly,
                                                "cost_per_side": COST, "stocks": len(frames)}, indent=1, ensure_ascii=False), encoding="utf8")
    L = ["# Combination study — V21 × the user's model (V20)", "",
         f"{len(frames)} stocks · entry/exit at next close · cost {COST*100:.2f}% per side · equal weight · benchmark TA-125", ""]
    for pname in periods:
        L += [f"## {pname}", "", "| # | strategy | CAGR % | excess vs TA-125 % | max DD % | Sharpe | info ratio | invested % | avg pos | trades | win % | beat index % | avg trade % | avg loss % | worst % | days |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        b = bstats[pname]
        L.append(f"| – | TA-125 | {b.get('cagr')} | 0 | {b.get('maxdd')} | {b.get('sharpe')} | – | 100 | – | – | – | – | – | – | – | – |")
        for key, r in res.items():
            s, t = r["periods"][pname], r["periods"][pname]["trades"]
            L.append(f"| {key} | {r['label']} | {s.get('cagr')} | {s.get('excess')} | {s.get('maxdd')} | {s.get('sharpe')} | {s.get('info')} | "
                     f"{s.get('invested')} | {s.get('avg_pos')} | {t.get('n')} | {t.get('win')} | {t.get('beat')} | {t.get('avg')} | {t.get('avg_loss')} | {t.get('worst')} | {t.get('days')} |")
        L.append("")
    years = sorted({y for v in yearly.values() for y in v})
    L += ["## Out-of-sample excess return vs TA-125 by year (%)", "", "| # | " + " | ".join(map(str, years)) + " |", "|---|" + "---|" * len(years)]
    for key, v in yearly.items():
        L.append(f"| {key} | " + " | ".join(str(v.get(y, "–")) for y in years) + " |")
    (out / "COMBO.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
