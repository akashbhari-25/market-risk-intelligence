"""Core risk and performance metrics (Phase 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_intel.analytics.returns import log_returns, simple_returns
from risk_intel.config import TRADING_DAYS_PER_YEAR


def _annualised_rf_daily(risk_free_annual: float) -> float:
    """Convert annual risk-free to implied daily simple rate (252-day convention)."""
    return (1.0 + float(risk_free_annual)) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


def _cagr_from_prices(prices: pd.Series) -> float:
    p = prices.dropna()
    if len(p) < 2:
        return float("nan")
    start, end = p.index[0], p.index[-1]
    years = (end - start).days / 365.25
    if years <= 0:
        return float("nan")
    return float((p.iloc[-1] / p.iloc[0]) ** (1.0 / years) - 1.0)


def _max_drawdown(prices: pd.Series) -> float:
    p = prices.dropna()
    if len(p) < 2:
        return float("nan")
    wealth = p / p.iloc[0]
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def _worst_drawdown_episode_days(prices: pd.Series) -> tuple[float, int | None]:
    """
    Return (max_drawdown, peak_to_trough_business_days).

    Peak is the maximum wealth date on or before the global max-drawdown trough (standard episode anchor).
    """
    p = prices.dropna()
    if len(p) < 2:
        return float("nan"), None
    wealth = p / p.iloc[0]
    roll_max = wealth.cummax()
    dd = wealth / roll_max - 1.0
    trough_loc = int(dd.values.argmin())
    trough_date = dd.index[trough_loc]
    sub = wealth.iloc[: trough_loc + 1]
    peak_date = sub.idxmax()
    days = int(np.busday_count(peak_date.date(), trough_date.date()))
    return float(dd.min()), days


def _sharpe(simple: pd.Series, rf_daily: float) -> float:
    r = simple.dropna()
    if len(r) < 5:
        return float("nan")
    excess = r - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(simple: pd.Series, rf_daily: float, mar: float | None = None) -> float:
    """Sortino using downside deviation vs MAR (default: risk-free daily)."""
    r = simple.dropna()
    if len(r) < 5:
        return float("nan")
    mar = rf_daily if mar is None else mar
    excess = r - rf_daily
    downside = r - mar
    downside = downside[downside < 0]
    if len(downside) < 2:
        return float("nan")
    ddev = float(np.sqrt((downside**2).mean()))
    if ddev == 0:
        return float("nan")
    return float(excess.mean() / ddev * np.sqrt(TRADING_DAYS_PER_YEAR))


def _calmar(cagr: float, max_dd: float) -> float:
    if max_dd >= 0 or np.isnan(max_dd) or np.isnan(cagr):
        return float("nan")
    return float(cagr / abs(max_dd))


def compute_core_metrics_table(
    prices: pd.DataFrame,
    risk_free_annual: float = 0.04,
) -> pd.DataFrame:
    """
    Per-column metrics aligned with common buy-side reporting.

    - CAGR and Calmar use **calendar** span on levels.
    - Volatility, Sharpe, Sortino use **simple** daily returns, annualised with sqrt(252).
    - Log-return volatility reported separately (some desks quote both).
    """
    rf_d = _annualised_rf_daily(risk_free_annual)
    rows: list[dict[str, float | str]] = []

    lr = log_returns(prices)
    sr = simple_returns(prices)

    for col in prices.columns:
        px = prices[col]
        mdd = _max_drawdown(px)
        cagr = _cagr_from_prices(px)
        mdd_depth, mdd_peak_to_trough_days = _worst_drawdown_episode_days(px)

        s = sr[col]
        vol_simple_ann = float(s.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(s.dropna()) > 5 else float("nan")
        lr_col = lr[col]
        vol_log_ann = (
            float(lr_col.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(lr_col.dropna()) > 5 else float("nan")
        )

        rows.append(
            {
                "ticker": col,
                "cagr": cagr,
                "ann_vol_simple": vol_simple_ann,
                "ann_vol_log": vol_log_ann,
                "sharpe": _sharpe(s, rf_d),
                "sortino": _sortino(s, rf_d),
                "calmar": _calmar(cagr, mdd),
                "max_drawdown": mdd,
                "worst_dd_peak_to_trough_days": float(mdd_peak_to_trough_days)
                if mdd_peak_to_trough_days is not None
                else float("nan"),
                "n_obs_prices": int(px.notna().sum()),
            }
        )

    return pd.DataFrame(rows).set_index("ticker")
