# Global Market Risk Intelligence Platform

Institutional-style market risk analytics in Python: live index data, log-return analytics, core risk metrics, and a Streamlit analyst shell. Built in **phases** so each layer stays reviewable and deployable.

## Phase 1 (current)

- Multi-index ingestion via **yfinance** with **SQLite audit trail** + on-disk Parquet cache
- **Log returns** and **simple returns** (Sharpe/Sortino use excess **simple** daily returns — industry-consistent)
- **CAGR**, annualised volatility, **Sharpe**, **Sortino**, **Calmar**, **max drawdown**, drawdown length
- Streamlit UI: index selection, date range, risk-free rate, metrics table, normalised price chart

## Repository layout

```
risk-intelligence-platform/
├── app.py                 # Streamlit entrypoint
├── requirements.txt
├── risk_intel/           # analytics + data layers (no UI logic inside metrics)
│   ├── config.py
│   ├── data/
│   │   ├── cache.py      # SQLite manifest + Parquet blobs
│   │   └── pipeline.py   # download, validate, align
│   └── analytics/
│       ├── returns.py
│       └── metrics.py
└── data_cache/            # created at runtime; gitignored
```

## Quickstart

```bash
cd risk-intelligence-platform
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Roadmap (later phases)

Tail risk (VaR/CVaR), regimes (HMM, GARCH), stress scenarios, portfolio optimisation, AI commentary (Claude), reinsurance framing — each as its own PR-sized phase on top of this data contract.

## Data contract (v1)

- **Prices**: split/dividend-adjusted closes from yfinance (`auto_adjust=True`) so returns are total-return consistent for broad indices.
- **Timezone**: dates normalised to **UTC midnight** index (naive) for stable caching keys.
- **Missing rows**: forward-fill limited to 5 consecutive sessions; gaps beyond that flagged in `DataManifest.warnings`.

## Disclaimer

Educational / research tooling. Not investment advice. Public data sources carry survivorship, backfill, and corporate-action semantics limitations.
