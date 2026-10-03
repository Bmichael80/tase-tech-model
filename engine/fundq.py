"""Quantitative fundamental score (0-100) from the companies' financial statements (Yahoo Finance).

Five components, each the average percentile of its metrics INSIDE THE PEER GROUP (similar industries):

  profitability 25%  gross profit / assets, operating margin, ROE            (banks / insurers: ROE, ROA)
  value         25%  earnings yield, book / price, EBITDA / EV, FCF yield     (banks / insurers: E/P, B/P)
  strength      20%  net debt / EBITDA (lower = better), interest cover, equity / assets
  quality       15%  cash flow from operations / assets, accruals (lower = better)   (not used for banks / insurers)
  growth        15%  revenue growth, change in ROA, change in operating margin (last fiscal year vs the one before,
                     and the latest quarter vs the same quarter a year earlier)   (banks / insurers: revenue, ROA change)

Why these (academic evidence): profitability - Novy-Marx 2013, Fama-French 2015 (RMW), Ball et al. 2016 (cash-based);
value - Fama-French 1992, Lakonishok-Shleifer-Vishny 1994, Loughran-Wellman 2011 (EBITDA/EV); strength - Piotroski
2000, Campbell-Hilscher-Szilagyi 2008 (distress); accruals - Sloan 1996; growth measured as IMPROVEMENT in profitability
(Piotroski, Asness-Frazzini-Pedersen "Quality minus junk") with a lower weight, because raw past sales growth on its own
has not predicted returns (Lakonishok et al. 1994).

Flows are trailing twelve months (sum of the last 4 quarters) when Yahoo has them, else the last fiscal year; balance
sheet items are the latest available; growth compares the last two fiscal years. A company needs at least 3 components
and 5 metrics to get a score. Market value is taken from Yahoo and checked against the universe file.
Usage: python -m engine.fundq            writes results/fundq/latest.json and results/fundq/history/<date>.json
"""
from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSION = "Q1"

COMPONENT_WEIGHTS = {"profitability": 0.25, "value": 0.25, "strength": 0.20, "quality": 0.15, "growth": 0.15}
COMPONENT_HE = {"profitability": "רווחיות", "value": "תמחור", "strength": "איתנות פיננסית", "quality": "איכות רווח ותזרים",
                "growth": "צמיחה ושיפור"}

PEER_GROUPS = {
    "בנקאות": "financials", "ביטוח": "financials",
    "שירותים פיננסים": "finance_holding", "השקעות ואחזקה": "finance_holding",
    'נדל"ן מניב': "real_estate", 'נדל"ן בנייה': "real_estate",
    "אנרגיה": "energy_infra", "גז ונפט": "energy_infra", "תשתיות": "energy_infra",
    "טכנולוגיה": "tech", "סמיקונדקטור": "tech", "אלקטרוניקה": "tech", "תעשייה ביטחונית": "tech",
    "תקשורת": "tech", "מדיה ותקשורת": "tech",
}
DEFAULT_GROUP = "consumer_industry"
GROUP_HE = {"financials": "בנקים וביטוח", "finance_holding": "פיננסים ואחזקות", "real_estate": "נדל\"ן",
            "energy_infra": "אנרגיה ותשתיות", "tech": "טכנולוגיה ותקשורת", "consumer_industry": "צריכה ותעשייה"}

# key: (component, higher_is_better, Hebrew label, models where it is NOT used)
METRICS = {
    "gp_assets":      ("profitability", True,  "רווח גולמי / נכסים",        {"financial", "real_estate"}),
    "op_margin":      ("profitability", True,  "מרווח תפעולי",              {"financial"}),
    "roe":            ("profitability", True,  "תשואה על ההון (ROE)",       set()),
    "roa":            ("profitability", True,  "תשואה על הנכסים (ROA)",     {"standard", "real_estate"}),
    "earn_yield":     ("value",         True,  "תשואת רווח (E/P)",          set()),
    "book_yield":     ("value",         True,  "הון / שווי שוק (B/P)",       set()),
    "ebitda_ev":      ("value",         True,  "EBITDA / שווי פירמה",       {"financial"}),
    "fcf_yield":      ("value",         True,  "תשואת תזרים חופשי",          {"financial"}),
    "netdebt_ebitda": ("strength",      False, "חוב נטו / EBITDA",          {"financial"}),
    "interest_cover": ("strength",      True,  "כיסוי ריבית",               {"financial"}),
    "equity_ratio":   ("strength",      True,  "הון עצמי / נכסים",          set()),
    "cfo_assets":     ("quality",       True,  "תזרים מפעילות / נכסים",      {"financial"}),
    "accruals":       ("quality",       False, "צבירות (רווח לא במזומן)",    {"financial"}),
    "rev_growth":     ("growth",        True,  "צמיחה בהכנסות",             set()),
    "d_roa":          ("growth",        True,  "שינוי ב-ROA",               set()),
    "d_margin":       ("growth",        True,  "שינוי במרווח התפעולי",      {"financial"}),
    "d_margin_q":     ("growth",        True,  "שינוי במרווח – רבעון אחרון מול רבעון מקביל", {"financial"}),
}
MIN_COMPONENTS, MIN_METRICS, MIN_PEERS = 3, 5, 5
MAX_AGE_DAYS = 550   # latest fiscal year older than this -> no score


def model_of(group: str) -> str:
    return "financial" if group == "financials" else "real_estate" if group == "real_estate" else "standard"


# ---------- statement helpers (frames: rows = items, columns = period end dates) ----------
def _series(df, *keys):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for k in keys:
        if k in df.index:
            s = pd.to_numeric(df.loc[k], errors="coerce")
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s[s.index.notna()].dropna().sort_index(ascending=False)
            if len(s):
                return s
    return None


def _ok(x):
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def latest(df, *keys):
    s = _series(df, *keys)
    return (float(s.iloc[0]), s.index[0]) if s is not None else (None, None)


def annual(df, i, *keys):
    s = _series(df, *keys)
    return float(s.iloc[i]) if s is not None and len(s) > i else None


def ttm(q, a, *keys):
    """Sum of the last 4 quarters if they are consecutive (about a year apart), else the last fiscal year."""
    s = _series(q, *keys)
    if s is not None and len(s) >= 4:
        span = (s.index[0] - s.index[3]).days
        if 250 <= span <= 300:
            return float(s.iloc[:4].sum()), True
    v = annual(a, 0, *keys)
    return v, False


def same_quarter_change(q, num_keys, den_keys):
    """Ratio num/den in the latest quarter minus the same ratio in the same quarter a year earlier (seasonality-safe)."""
    n, d = _series(q, *num_keys), _series(q, *den_keys)
    if n is None or d is None:
        return None
    df = pd.concat([n.rename("n"), d.rename("d")], axis=1).dropna().sort_index(ascending=False)
    if len(df) < 2:
        return None
    t0 = df.index[0]
    prior = df[(df.index <= t0 - pd.Timedelta(days=320)) & (df.index >= t0 - pd.Timedelta(days=410))]
    if prior.empty or df["d"].iloc[0] <= 0 or prior["d"].iloc[0] <= 0:
        return None
    return float(df["n"].iloc[0] / df["d"].iloc[0] - prior["n"].iloc[0] / prior["d"].iloc[0])


def _div(a, b, positive_den=True):
    if not (_ok(a) and _ok(b)) or b == 0 or (positive_den and b < 0):
        return None
    return a / b


# ---------- metrics for one company ----------
def compute_metrics(raw: dict) -> dict:
    """raw: income/balance/cash (yearly) and q_income/q_balance/q_cash (quarterly) frames, plus
    mcap_fin (market value in the reporting currency, or None)."""
    ai, ab, ac = raw.get("income"), raw.get("balance"), raw.get("cash")
    qi, qb, qc = raw.get("q_income"), raw.get("q_balance"), raw.get("q_cash")
    used_ttm = []

    def flow(*keys, cash=False):
        v, t = ttm(qc if cash else qi, ac if cash else ai, *keys)
        used_ttm.append(t)
        return v

    def bal(*keys):
        v, d = latest(qb, *keys)
        if v is None:
            v, d = latest(ab, *keys)
        return v

    ni = flow("NetIncomeCommonStockholders", "NetIncome")
    rev = flow("TotalRevenue", "OperatingRevenue")
    gp = flow("GrossProfit")
    oi = flow("OperatingIncome", "EBIT")
    ebit = flow("EBIT", "OperatingIncome")
    ebitda = flow("EBITDA", "NormalizedEBITDA")
    intexp = flow("InterestExpense", "InterestExpenseNonOperating")
    cfo = flow("OperatingCashFlow", cash=True)
    fcf = flow("FreeCashFlow", cash=True)
    if fcf is None and cfo is not None:
        capex = flow("CapitalExpenditure", cash=True)
        fcf = cfo + capex if capex is not None else None

    ta = bal("TotalAssets")
    eq = bal("StockholdersEquity", "CommonStockEquity")
    debt = bal("TotalDebt")
    cash = bal("CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalents")
    eq_pos = eq if _ok(eq) and eq > 0 else None
    mcap = raw.get("mcap_fin")

    m = {}
    m["gp_assets"] = _div(gp, ta)
    m["op_margin"] = _div(oi, rev)
    m["roe"] = _div(ni, eq_pos)
    m["roa"] = _div(ni, ta)
    m["earn_yield"] = _div(ni, mcap)
    m["book_yield"] = _div(eq_pos, mcap)
    ev = mcap + (debt or 0) - (cash or 0) if _ok(mcap) and (_ok(debt) or _ok(cash)) else None
    m["ebitda_ev"] = _div(ebitda, ev)
    m["fcf_yield"] = _div(fcf, mcap)
    if _ok(debt) or _ok(cash):
        nd = (debt or 0) - (cash or 0)
        m["netdebt_ebitda"] = -1.0 if nd <= 0 else (nd / ebitda if _ok(ebitda) and ebitda > 0 else 99.0)
    else:
        m["netdebt_ebitda"] = None
    if _ok(intexp) and abs(intexp) > 0:
        m["interest_cover"] = _div(ebit, abs(intexp), positive_den=False)
        if m["interest_cover"] is not None:
            m["interest_cover"] = max(min(m["interest_cover"], 100.0), -100.0)
    elif _ok(debt) and _ok(ta) and debt <= 0.02 * ta and _ok(ebit):
        m["interest_cover"] = 100.0      # practically no debt
    else:
        m["interest_cover"] = None
    m["equity_ratio"] = _div(eq, ta)
    m["cfo_assets"] = _div(cfo, ta)
    m["accruals"] = _div(ni - cfo, ta) if _ok(ni) and _ok(cfo) else None

    r0, r1 = annual(ai, 0, "TotalRevenue", "OperatingRevenue"), annual(ai, 1, "TotalRevenue", "OperatingRevenue")
    n0, n1 = annual(ai, 0, "NetIncomeCommonStockholders", "NetIncome"), annual(ai, 1, "NetIncomeCommonStockholders", "NetIncome")
    t0, t1 = annual(ab, 0, "TotalAssets"), annual(ab, 1, "TotalAssets")
    o0, o1 = annual(ai, 0, "OperatingIncome", "EBIT"), annual(ai, 1, "OperatingIncome", "EBIT")
    g = _div(r0, r1)
    m["rev_growth"] = None if g is None else max(min(g - 1, 5.0), -1.0)
    a0, a1 = _div(n0, t0), _div(n1, t1)
    m["d_roa"] = a0 - a1 if a0 is not None and a1 is not None else None
    mg0, mg1 = _div(o0, r0), _div(o1, r1)
    m["d_margin"] = mg0 - mg1 if mg0 is not None and mg1 is not None else None
    m["d_margin_q"] = same_quarter_change(qi, ("OperatingIncome", "EBIT"), ("TotalRevenue", "OperatingRevenue"))

    _, last_fy = latest(ai, "TotalRevenue", "NetIncome", "NetIncomeCommonStockholders")
    _, last_q = latest(qi, "TotalRevenue", "NetIncome", "NetIncomeCommonStockholders")
    meta = {"fiscal_year_end": last_fy.date().isoformat() if last_fy is not None else None,
            "last_quarter": last_q.date().isoformat() if last_q is not None else None,
            "ttm": bool(used_ttm) and any(used_ttm)}
    return {"metrics": {k: (round(v, 4) if _ok(v) else None) for k, v in m.items()}, "meta": meta}


# ---------- scoring across companies ----------
def _pct(values: pd.Series, higher: bool) -> pd.Series:
    """Percentile 0..100 (ties averaged); 100 = best in the group."""
    v = values.dropna()
    if len(v) < 2:
        return pd.Series(50.0, index=v.index)
    r = (v if higher else -v).rank(method="average")
    return (r - 1) / (len(v) - 1) * 100


def score_all(rows: list[dict], asof: datetime | None = None) -> dict:
    """rows: {tase_id, group, metrics, meta}. Returns {tase_id: result}."""
    asof = asof or datetime.now(timezone.utc)
    df = pd.DataFrame([{**{"tase_id": r["tase_id"], "group": r["group"], "model": model_of(r["group"])}, **r["metrics"]}
                       for r in rows]).set_index("tase_id")
    stale = set()
    for r in rows:
        fy = r["meta"].get("fiscal_year_end")
        if fy is None or (asof.date() - datetime.fromisoformat(fy).date()).days > MAX_AGE_DAYS:
            stale.add(r["tase_id"])
    pct = pd.DataFrame(index=df.index, columns=list(METRICS), dtype=float)
    peer_used = {}
    for key, (_, higher, _, skip) in METRICS.items():
        usable = df[~df["model"].isin(skip) & ~df.index.isin(stale)]
        for grp, sub in usable.groupby("group"):
            vals = sub[key].dropna()
            if len(vals) >= MIN_PEERS:
                pct.loc[vals.index, key] = _pct(vals, higher)
                peer_used.update({(t, key): grp for t in vals.index})
            else:   # too few peers: rank against every company that uses the same model
                same = usable[usable["model"] == sub["model"].iloc[0]][key].dropna()
                if len(same) >= MIN_PEERS:
                    p = _pct(same, higher)
                    ids = vals.index
                    pct.loc[ids, key] = p.loc[ids]
                    peer_used.update({(t, key): "all_" + sub["model"].iloc[0] for t in ids})
    out = {}
    for r in rows:
        tid = r["tase_id"]
        comps, n_metrics = {}, 0
        for c in COMPONENT_WEIGHTS:
            ps = [pct.at[tid, k] for k, (cc, _, _, _) in METRICS.items() if cc == c and pd.notna(pct.at[tid, k])]
            if ps:
                comps[c] = round(sum(ps) / len(ps), 1)
                n_metrics += len(ps)
        score = None
        reason = None
        if tid in stale:
            reason = "no recent financial statements"
        elif len(comps) < MIN_COMPONENTS or n_metrics < MIN_METRICS:
            reason = f"not enough data ({len(comps)} components, {n_metrics} metrics)"
        else:
            w = sum(COMPONENT_WEIGHTS[c] for c in comps)
            score = int(round(sum(COMPONENT_WEIGHTS[c] * v for c, v in comps.items()) / w))
        out[tid] = {
            "score": score, "reason": reason, "group": r["group"], "group_he": GROUP_HE.get(r["group"], r["group"]),
            "model": model_of(r["group"]), "components": comps,
            "metrics": {k: {"value": r["metrics"].get(k),
                            "pct": (round(float(pct.at[tid, k]), 1) if pd.notna(pct.at[tid, k]) else None)}
                        for k in METRICS if r["metrics"].get(k) is not None},
            **r["meta"],
        }
    return out


# ---------- data download (GitHub Actions only) ----------
def _frame(fn, **kw):
    try:
        f = fn(pretty=False, **kw)
        return f if isinstance(f, pd.DataFrame) and not f.empty else None
    except Exception:
        return None


def _fx(yf, cur: str, cache: dict):
    if cur in (None, "ILS", "ILA"):
        return 1.0
    if cur not in cache:
        try:
            h = yf.Ticker(f"{cur}ILS=X").history(period="10d")["Close"].dropna()
            cache[cur] = float(h.iloc[-1]) if len(h) else None
        except Exception:
            cache[cur] = None
    return cache[cur]


def fetch_one(yf, ticker: str, mcap_ref_ils: float | None, fx_cache: dict) -> dict:
    t = yf.Ticker(ticker)
    raw = {"income": _frame(t.get_income_stmt, freq="yearly"), "balance": _frame(t.get_balance_sheet, freq="yearly"),
           "cash": _frame(t.get_cash_flow, freq="yearly"), "q_income": _frame(t.get_income_stmt, freq="quarterly"),
           "q_balance": _frame(t.get_balance_sheet, freq="quarterly"), "q_cash": _frame(t.get_cash_flow, freq="quarterly")}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}
    fin_cur = info.get("financialCurrency") or "ILS"
    price = None
    try:
        h = t.history(period="10d")["Close"].dropna()
        price = float(h.iloc[-1]) if len(h) else None
    except Exception:
        pass
    px_cur = info.get("currency") or "ILA"
    price_ils = None if price is None else price / 100 if px_cur == "ILA" else price
    shares = info.get("sharesOutstanding")
    if not shares:
        shares, _ = latest(raw["q_balance"], "OrdinarySharesNumber", "ShareIssued")
        if shares is None:
            shares, _ = latest(raw["balance"], "OrdinarySharesNumber", "ShareIssued")
    cands = []
    if shares and price_ils:
        cands.append(("shares x price", shares * price_ils))
    if info.get("marketCap"):
        cands += [("yahoo market cap", float(info["marketCap"])), ("yahoo market cap / 100", float(info["marketCap"]) / 100)]
    mcap_ils, src, warn = None, None, []
    if mcap_ref_ils:
        best = min(cands, key=lambda c: abs(math.log(c[1] / mcap_ref_ils)), default=None) if cands else None
        if best and abs(math.log(best[1] / mcap_ref_ils)) <= math.log(3):
            src, mcap_ils = best
        elif cands:
            warn.append("market value does not match the universe file; valuation metrics skipped")
    elif cands:
        src, mcap_ils = cands[0]
    rate = _fx(yf, fin_cur, fx_cache)
    raw["mcap_fin"] = mcap_ils / rate if mcap_ils and rate else None
    res = compute_metrics(raw)
    res["meta"].update({"fin_currency": fin_cur, "mcap_ils_m": round(mcap_ils / 1e6, 1) if mcap_ils else None,
                        "mcap_source": src, "warnings": warn,
                        "statements": {k: raw[k] is not None for k in ("income", "balance", "cash", "q_income", "q_balance", "q_cash")}})
    return res


def main(argv=None):
    import yfinance as yf
    with open(ROOT / "config" / "universe.csv", encoding="utf8") as f:
        uni = list(csv.DictReader(f))
    rows, fx_cache, errors = [], {}, []
    for i, u in enumerate(uni):
        ref = float(u["mcap_m"]) * 1e6 if u.get("mcap_m") not in (None, "") else None
        res = None
        for attempt in range(3):
            try:
                res = fetch_one(yf, u["yahoo"], ref, fx_cache)
                break
            except Exception as e:
                err = str(e)[:200]
                time.sleep(3 * (attempt + 1))
        if res is None:
            errors.append({"ticker": u["yahoo"], "name": u["name"], "error": err})
            res = {"metrics": {k: None for k in METRICS}, "meta": {"fiscal_year_end": None, "warnings": ["download failed"]}}
        rows.append({"tase_id": u["tase_id"], "name": u["name"], "ticker": u["yahoo"], "sector": u["sector"],
                     "group": PEER_GROUPS.get(u["sector"], DEFAULT_GROUP), **res})
        time.sleep(0.4)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(uni)} downloaded")
    now = datetime.now(timezone.utc)
    scored = score_all(rows, now)
    for r in rows:
        scored[r["tase_id"]].update({"name": r["name"], "ticker": r["ticker"], "sector": r["sector"]})
    out = {"model": "Quantitative fundamental score " + VERSION, "version": VERSION,
           "generated_utc": now.isoformat(timespec="seconds"), "data_source": "Yahoo Finance financial statements",
           "component_weights": COMPONENT_WEIGHTS, "component_he": COMPONENT_HE,
           "metrics_he": {k: v[2] for k, v in METRICS.items()},
           "metric_direction": {k: ("higher" if v[1] else "lower") for k, v in METRICS.items()},
           "count": len(rows), "scored": sum(1 for v in scored.values() if v["score"] is not None),
           "errors": errors, "fx": fx_cache, "companies": scored}
    d = ROOT / "results" / "fundq"
    (d / "history").mkdir(parents=True, exist_ok=True)
    txt = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    (d / "latest.json").write_text(txt, encoding="utf8")
    (d / "history" / f"{now.date().isoformat()}.json").write_text(txt, encoding="utf8")
    print(f"quantitative fundamental score: {out['scored']}/{out['count']} companies scored")
    for e in errors:
        print("  download failed:", e["ticker"], e["name"], e["error"])


if __name__ == "__main__":
    main()
