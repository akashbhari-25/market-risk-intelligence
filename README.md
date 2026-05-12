# Global Market Risk Intelligence Platform

Institutional-style market risk analytics in Python: live index data, log-return analytics, core risk metrics, and a Streamlit analyst shell. Built in **phases** so each layer stays reviewable and deployable.

## Phase 1

- Multi-index ingestion via **yfinance** with **SQLite audit trail** + on-disk Parquet cache
- **Log returns** and **simple returns** (Sharpe/Sortino use excess **simple** daily returns — industry-consistent)
- **CAGR**, annualised volatility, **Sharpe**, **Sortino**, **Calmar**, **max drawdown**, drawdown length
- Streamlit **Overview** tab: metrics table, normalised price chart, CSV export

## Phase 2A

- **Historical VaR / CVaR** (empirical quantile + tail mean) and **Gaussian VaR / CVaR** on simple returns
- **Skewness**, **excess kurtosis**, **Jarque–Bera** normality test
- **Annualised** VaR/CVaR via √252 scaling (documented IID shortcut)
- Streamlit **Tail risk** tab: summary table, histogram vs normal overlay, normal Q-Q plot, CSV export

## Phase 2B (current)

- **Full-sample correlation** heatmap (Pearson on simple returns)
- **Crisis-window** presets (GFC, COVID, 2022, 2018 Q4, Brexit) with Δ vs full-sample average pairwise correlation
- **Rolling** pairwise correlation and **rolling mean pairwise** correlation (63 / 126 / 252 sessions)
- CSV export of the full-sample correlation matrix

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
│       ├── metrics.py
│       ├── tail_risk.py
│       └── correlation.py
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

Regimes (HMM, GARCH), stress scenarios, portfolio optimisation, AI commentary (Claude), reinsurance framing — each as its own PR-sized phase on top of this data contract.

## Data contract (v1)

- **Prices**: split/dividend-adjusted closes from yfinance (`auto_adjust=True`) so returns are total-return consistent for broad indices.
- **Timezone**: dates normalised to **UTC midnight** index (naive) for stable caching keys.
- **Missing rows**: forward-fill limited to 5 consecutive sessions; gaps beyond that flagged in `DataManifest.warnings`.

## Disclaimer

Educational / research tooling. Not investment advice. Public data sources carry survivorship, backfill, and corporate-action semantics limitations.
