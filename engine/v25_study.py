"""V25 study: does technical timing work over SHORT horizons (5 and 10 trading days)?

Follow-up to V24 (timing added nothing at 1, 3 and 6 months). Same nine timing votes as engine/v24_study.py
(moving averages, MACD, RSI, MFI, volume, Bollinger, support / resistance, the user's model). Fixed in advance:

  horizons     5 and 10 trading days (21 = 1 month for reference); entry at the next session's close,
               return in excess of the TA-125
  part 1       rank IC of each vote and of TIMING (their average) over all liquid stocks
  part 2       inside the strong group (V22 score >= 80): excess return of stocks with bullish timing
               (TIMING >= +0.5, or > 0) vs the whole strong group - "which strong stock to buy this week"
  part 3       inside the weak group (V22 < 50): bearish timing (<= -0.5, or < 0) vs the whole weak group
  verdict      timing is useful for an advisor only if, in BOTH IS (2013-2019) and OOS (2020-2026), the edge at
               5 or 10 days is at least 0.3 pp (one round-trip cost of 0.15% per side) with 30+ events per period

Usage: python -m engine.v25_study [--start 2012-01-01] [--split 2020-01-01] [--offline DIR]
Writes results/research/V25.md and v25.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .combo import V20_EXIT
from .research import build_factors, rank_ic, summarize_ic, weekly_dates
from .v23_study import group_edge, to_score
from .v24_study import votes_for

ROOT = Path(__file__).resolve().parents[1]
H = {"5d": 5, "10d": 10, "21d": 21}
MIN_EDGE, MIN_N = 0.3, 30


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
    C, b = X["close"], X["bench"]
    idx, cols = C.index, C.columns
    for k in ("signal", "sell_score"):
        X[k] = X[k].reindex(columns=cols)

    ex6 = (C / C.shift(126)).div(b / b.shift(126), axis=0) - 1
    m = tradable & ex6.notna()
    score = to_score(ex6, m)
    fwd = {k: ((C.shift(-(h + 1)) / C.shift(-1) - 1).sub(b.shift(-(h + 1)) / b.shift(-1) - 1, axis=0) * 100) for k, h in H.items()}

    V: dict[str, dict] = {}
    for t, df in frames.items():
        if t not in cols or len(df) < 260:
            continue
        for k, s in votes_for(df).items():
            V.setdefault(k, {})[t] = s.reindex(idx)
    votes = {k: pd.DataFrame(v).reindex(columns=cols) for k, v in V.items()}
    v20sig = X["signal"].fillna(0).rolling(5, min_periods=1).max()
    votes["your_model"] = pd.DataFrame(np.select([v20sig.to_numpy() >= 1, X["sell_score"].to_numpy() >= V20_EXIT], [1.0, -1.0], 0.0),
                                       index=idx, columns=cols).where(X["sell_score"].notna())
    timing = pd.DataFrame(np.nanmean(np.stack([votes[k].to_numpy() for k in votes]), axis=0), index=idx, columns=cols).where(m)
    votes["TIMING (all votes)"] = timing

    dates = weekly_dates(idx)
    dates = dates[dates >= idx[0] + pd.Timedelta(days=400)]
    split = pd.Timestamp(args.split)
    wp = {"IS": dates[dates < split], "OOS": dates[dates >= split],
          "2020-22": dates[(dates >= split) & (dates < "2023-01-01")], "2023-26": dates[dates >= "2023-01-01"]}

    ic = {k: {p: {h: summarize_ic(rank_ic(v.where(m), fwd[h], tradable, ds), hd) for h, hd in H.items()} for p, ds in wp.items()}
          for k, v in votes.items()}

    strong, weak = m & (score >= 80), m & (score < 50)
    tests = {
        "strong: timing >= +0.5": (strong & (timing >= 0.5), strong, +1),
        "strong: timing > 0": (strong & (timing > 0), strong, +1),
        "strong: timing < 0 (pullback)": (strong & (timing < 0), strong, +1),
        "strong: your model BUY signal": (strong & (v20sig >= 1), strong, +1),
        "weak: timing <= -0.5": (weak & (timing <= -0.5), weak, -1),
        "weak: timing < 0": (weak & (timing < 0), weak, -1),
        "weak: your model EXIT": (weak & (X["sell_score"] >= V20_EXIT), weak, -1),
    }
    res = {name: {p: {h: group_edge(g, base, fwd[h], wp[p], hd) for h, hd in H.items()} for p in wp} for name, (g, base, _) in tests.items()}

    useful = []
    for name, (_, _, sign) in tests.items():
        for h in ("5d", "10d"):
            r = [res[name][p][h] for p in ("IS", "OOS")]
            if all(x.get("n", 0) >= MIN_N and x.get("edge") is not None and sign * x["edge"] >= MIN_EDGE for x in r):
                useful.append(f"{name} @ {h}")

    out = ROOT / "results" / "research"
    out.mkdir(parents=True, exist_ok=True)
    (out / "v25.json").write_text(json.dumps({"useful": useful, "ic": ic, "groups": res}, indent=1, ensure_ascii=False, default=str), encoding="utf8")
    f = lambda x: f"{x.get('ic')} ({x.get('t')})"
    L = ["# V25 study — short-horizon technical timing (5 and 10 trading days)", "",
         f"{len(frames)} stocks · weekly observations · liquid names only · IS 2013-2019, OOS 2020-2026 blind", "",
         "## Result", "",
         ("Timing that passed (edge ≥ 0.3 pp in the right direction, 30+ events, in BOTH periods):\n" + "\n".join(f"* {u}" for u in useful))
         if useful else "**No timing rule passed** (edge ≥ 0.3 pp in the right direction, 30+ events, in both IS and OOS, at 5 or 10 days).", "",
         "## Part 1 — rank IC of each vote vs forward excess return", "",
         "| vote | IS 5d | IS 10d | IS 21d | OOS 5d | OOS 10d | OOS 21d | 2020-22 10d | 2023-26 10d |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in ic.items():
        L.append(f"| {k} | {f(r['IS']['5d'])} | {f(r['IS']['10d'])} | {f(r['IS']['21d'])} | {f(r['OOS']['5d'])} | {f(r['OOS']['10d'])} | "
                 f"{f(r['OOS']['21d'])} | {r['2020-22']['10d'].get('ic')} | {r['2023-26']['10d'].get('ic')} |")
    L += ["", "## Parts 2-3 — timing inside the strong / weak groups (excess return vs the whole group, pp; t in brackets)", "",
          "| test | IS n (10d) | IS 5d | IS 10d | IS 21d | OOS n (10d) | OOS 5d | OOS 10d | OOS 21d | 2020-22 10d | 2023-26 10d |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    e = lambda x: f"{x.get('edge')} ({x.get('t')})"
    for name, r in res.items():
        L.append(f"| {name} | {r['IS']['10d'].get('n')} | {e(r['IS']['5d'])} | {e(r['IS']['10d'])} | {e(r['IS']['21d'])} | {r['OOS']['10d'].get('n')} | "
                 f"{e(r['OOS']['5d'])} | {e(r['OOS']['10d'])} | {e(r['OOS']['21d'])} | {r['2020-22']['10d'].get('edge')} | {r['2023-26']['10d'].get('edge')} |")
    (out / "V25.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
