"""Rolling risk and drawdown-based regime analytics (Phase 3A)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_intel.config import TRADING_DAYS_PER_YEAR


def _rf_daily_from_annual(rf_annual: float) -> float:
    return (1.0 + float(rf_annual)) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


def rolling_volatility(simple_returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling annualized volatility on simple returns."""
    r = simple_returns.dropna()
    min_periods = max(10, window // 2)
    return r.rolling(window=window, min_periods=min_periods).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_sharpe(simple_returns: pd.Series, rf_annual: float, window: int = 252) -> pd.Series:
    """Rolling annualized Sharpe ratio using daily excess simple returns."""
    r = simple_returns.dropna()
    min_periods = max(10, window // 2)
    rf_daily = _rf_daily_from_annual(rf_annual)
    excess = r - rf_daily
    mean_excess = excess.rolling(window=window, min_periods=min_periods).mean()
    std_excess = excess.rolling(window=window, min_periods=min_periods).std()
    sharpe = mean_excess / std_excess
    return sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling beta = rolling cov(asset, benchmark) / rolling var(benchmark)."""
    pair = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if pair.shape[1] != 2:
        return pd.Series(dtype=float)
    asset = pair.iloc[:, 0]
    bench = pair.iloc[:, 1]
    min_periods = max(10, window // 2)
    cov = asset.rolling(window=window, min_periods=min_periods).cov(bench)
    var_bench = bench.rolling(window=window, min_periods=min_periods).var()
    return cov / var_bench


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Drawdown from running peak: price / cummax(price) - 1."""
    p = prices.dropna()
    if p.empty:
        return pd.Series(dtype=float)
    return p / p.cummax() - 1.0


def market_regime_from_drawdown(drawdown: pd.Series, bear_threshold: float = -0.20) -> pd.Series:
    """Label each date as Bull/Bear based on drawdown threshold."""
    dd = drawdown.dropna()
    out = pd.Series(index=dd.index, dtype=object)
    out[dd <= bear_threshold] = "Bear"
    out[dd > bear_threshold] = "Bull"
    return out
