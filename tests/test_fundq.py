from datetime import datetime, timezone

import pandas as pd

from engine import fundq as Q


def _stmt(items: dict, dates):
    return pd.DataFrame(items, index=pd.to_datetime(dates)).T


def _company(scale: float, debt: float):
    ydates = ["2025-12-31", "2024-12-31"]
    qdates = ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]
    inc = _stmt({"TotalRevenue": [1000 * scale, 900], "GrossProfit": [400 * scale, 350], "OperatingIncome": [150 * scale, 120],
                 "EBIT": [150 * scale, 120], "EBITDA": [200 * scale, 170], "NetIncome": [100 * scale, 80],
                 "InterestExpense": [10, 10]}, ydates)
    q_inc = _stmt({"TotalRevenue": [260 * scale] * 4, "NetIncome": [26 * scale] * 4, "GrossProfit": [100 * scale] * 4,
                   "OperatingIncome": [40 * scale] * 4, "EBIT": [40 * scale] * 4, "EBITDA": [52 * scale] * 4,
                   "InterestExpense": [2.5] * 4}, qdates)
    bal = _stmt({"TotalAssets": [2000, 1900], "StockholdersEquity": [800, 760], "TotalDebt": [debt, debt],
                 "CashAndCashEquivalents": [100, 100]}, ydates)
    cash = _stmt({"OperatingCashFlow": [120 * scale, 100], "FreeCashFlow": [80 * scale, 60]}, ydates)
    return {"income": inc, "balance": bal, "cash": cash, "q_income": q_inc, "q_balance": None, "q_cash": None, "mcap_fin": 1500}


def test_metrics_use_ttm_and_handle_net_cash():
    r = Q.compute_metrics(_company(1.0, 50))
    assert r["meta"]["ttm"] is True
    assert abs(r["metrics"]["op_margin"] - 160 / 1040) < 1e-3          # TTM from the 4 quarters
    assert r["metrics"]["netdebt_ebitda"] == -1.0                      # cash > debt
    assert abs(r["metrics"]["rev_growth"] - (1000 / 900 - 1)) < 1e-3   # fiscal years


def test_better_company_scores_higher_and_bounds():
    rows = []
    for i in range(8):
        m = Q.compute_metrics(_company(0.6 + 0.1 * i, 300 - 30 * i))
        rows.append({"tase_id": str(i), "group": "tech", **m})
    out = Q.score_all(rows, datetime(2026, 10, 3, tzinfo=timezone.utc))
    scores = [out[str(i)]["score"] for i in range(8)]
    assert all(s is not None and 0 <= s <= 100 for s in scores)
    assert scores[-1] > scores[0]


def test_financials_skip_cash_flow_metrics_and_stale_data():
    rows = []
    for i in range(6):
        m = Q.compute_metrics(_company(0.8 + 0.1 * i, 100))
        rows.append({"tase_id": f"b{i}", "group": "financials", **m})
    rows[0]["meta"]["fiscal_year_end"] = "2023-12-31"
    out = Q.score_all(rows, datetime(2026, 10, 3, tzinfo=timezone.utc))
    assert out["b0"]["score"] is None
    assert "quality" not in out["b1"]["components"]
    assert out["b1"]["score"] is not None


def test_same_quarter_margin_change():
    q = _stmt({"TotalRevenue": [100, 100, 100, 100, 100], "OperatingIncome": [20, 15, 15, 15, 10]},
              ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30"])
    assert abs(Q.same_quarter_change(q, ("OperatingIncome",), ("TotalRevenue",)) - 0.10) < 1e-9
    assert Q.same_quarter_change(q.iloc[:, :4], ("OperatingIncome",), ("TotalRevenue",)) is None
