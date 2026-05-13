"""Mean–variance portfolio optimisation (Phase 5) — PyPortfolioOpt."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models


def frequency_for_interval(interval: str) -> int:
    """Annualisation frequency for PyPortfolioOpt estimators."""
    if interval == "1wk":
        return 52
    if interval == "1mo":
        return 12
    return 252


def prepare_prices(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows/columns and forward-fill for stable estimation."""
    df = panel.sort_index().dropna(how="all").dropna(axis=1, how="all")
    return df.ffill().dropna()


def estimate_mu_cov(prices: pd.DataFrame, interval: str) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return (mu, cov, cleaned_prices) in annualised units for the given sampling frequency."""
    px = prepare_prices(prices)
    freq = frequency_for_interval(interval)
    mu = expected_returns.mean_historical_return(px, frequency=freq)
    cov = risk_models.sample_cov(px, frequency=freq)
    return mu, cov, px


def _perf(mu: pd.Series, cov: pd.DataFrame, weights: pd.Series, rf_annual: float) -> tuple[float, float, float]:
    w = weights.reindex(mu.index).fillna(0.0).to_numpy(dtype=float)
    exp_ret = float(mu.to_numpy(dtype=float) @ w)
    vol = float(np.sqrt(w.T @ cov.to_numpy(dtype=float) @ w))
    sharpe = (exp_ret - float(rf_annual)) / vol if vol > 0 else float("nan")
    return exp_ret, vol, sharpe


def efficient_frontier_cloud(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 35,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[list[float], list[float]]:
    """
    Approximate the long-only frontier by sweeping target returns.

    Returns parallel lists (vols, rets) in annualised units.
    """
    cols = list(mu.index)
    targets = np.linspace(float(mu.min()), float(mu.max()), n_points)
    vols: list[float] = []
    rets: list[float] = []
    for t in targets:
        try:
            ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)
            ef.efficient_return(target_return=float(t))
            cw = ef.clean_weights()
            w = pd.Series({c: float(cw.get(c, 0.0)) for c in cols}, index=cols)
            er, vol, _ = _perf(mu, cov, w, rf_annual=0.0)
            if np.isfinite(vol) and vol > 0:
                vols.append(vol)
                rets.append(er)
        except Exception:
            continue
    return vols, rets


@dataclass
class PortfolioOptimizeResult:
    weights: pd.DataFrame
    performance: pd.DataFrame
    notes: list[str]


def optimize_portfolios(
    panel: pd.DataFrame,
    interval: str,
    rf_annual: float,
    long_only: bool = True,
) -> PortfolioOptimizeResult:
    """
    Equal weight, minimum volatility, and maximum Sharpe (long-only by default).

    Expected returns and covariance use PyPortfolioOpt sample estimators with
    frequency matched to the app's sampling interval.
    """
    notes: list[str] = []
    mu, cov, prices = estimate_mu_cov(panel, interval)
    if prices.shape[1] < 2:
        notes.append("Select at least two tickers with overlapping price history.")
        return PortfolioOptimizeResult(pd.DataFrame(), pd.DataFrame(), notes)
    if prices.shape[0] < prices.shape[1] + 5:
        notes.append("Short history relative to number of assets — estimates may be unstable.")

    wb: tuple[float, float] = (0.0, 1.0) if long_only else (-1.0, 1.0)

    n = int(prices.shape[1])
    ew = pd.Series({c: 1.0 / n for c in prices.columns}, dtype=float)

    weight_map: dict[str, pd.Series] = {"Equal weight": ew}

    try:
        ef_mv = EfficientFrontier(mu, cov, weight_bounds=wb)
        ef_mv.min_volatility()
        w_mv = pd.Series(ef_mv.clean_weights(), dtype=float)
        weight_map["Min volatility"] = w_mv
    except Exception as exc:
        notes.append(f"Min volatility optimisation failed: {exc}")

    try:
        ef_ms = EfficientFrontier(mu, cov, weight_bounds=wb)
        ef_ms.max_sharpe(risk_free_rate=float(rf_annual))
        w_ms = pd.Series(ef_ms.clean_weights(), dtype=float)
        weight_map["Max Sharpe"] = w_ms
    except Exception as exc:
        notes.append(f"Max Sharpe optimisation failed: {exc}")

    weights_df = pd.DataFrame(weight_map).fillna(0.0)

    perf_rows: list[dict[str, float | str]] = []
    for name, w in weight_map.items():
        er, vol, sh = _perf(mu, cov, w, rf_annual)
        perf_rows.append({"method": name, "exp_return": er, "ann_vol": vol, "sharpe": sh})
    performance = pd.DataFrame(perf_rows).set_index("method")

    return PortfolioOptimizeResult(weights=weights_df, performance=performance, notes=notes)
