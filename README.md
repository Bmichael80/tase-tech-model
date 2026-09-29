# tase-tech-model

The technical scoring engine behind the "Israel Stock Analysis" dashboard.

- **V21 rating (`engine/v21.py`)**: a cross-sectional momentum/trend composite, selected by out-of-sample factor research on TASE (`engine/research.py`, report in `results/research/FACTORS.md`).
- **V20 entry timing (`engine/model.py`)**: a Python port of the TradingView indicator *Analyst Dashboard V19.0 + Whale Fusion V2*, with corrections.

See [METHODOLOGY.md](METHODOLOGY.md) for definitions, evidence and limitations.

## What runs automatically

| Workflow | When | What it does |
|---|---|---|
| **Weekly technical scores** | Every Saturday, 05:15 Israel time | Downloads daily data for the 62 stocks, scores them, and commits `results/latest.json` and `latest.csv` |
| **Validate tickers and run backtest** | On the 2nd of every month, or on demand | Checks every ticker, backtests all model variants, and writes `results/backtest/REPORT.md` |

Both workflows can also be started by hand: go to **Actions**, pick the workflow, and click **Run workflow**.

## Files

| Path | Contents |
|---|---|
| `config/universe.csv` | The stocks: TASE security number, name, sector, Yahoo ticker. Add or remove rows here. |
| `engine/model.py` | The model. `ModelConfig()` is V20; `ModelConfig(version="V19")` is the original. |
| `engine/indicators.py` | TradingView-compatible indicators |
| `engine/run.py` | The weekly run |
| `engine/backtest.py` | Historical validation |
| `tests/` | Automated tests. They run before every weekly run. |

## Run locally

```
pip install -r requirements.txt
python -m pytest -q
python -m engine.run
```
