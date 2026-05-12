"""Correlation analytics on aligned simple-return panels."""

from __future__ import annotations

import pandas as pd

from risk_intel.analytics.returns import simple_returns

# Inclusive calendar bounds (works for daily / weekly / monthly indices).
CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "2008–09 Global Financial Crisis": ("2008-09-01", "2009-03-31"),
    "2020 COVID crash": ("2020-02-15", "2020-04-30"),
    "2022 Inflation / rates shock": ("2022-01-01", "2022-10-31"),
    "2018 Q4 selloff": ("2018-10-01", "2018-12-24"),
    "2016 Brexit window": ("2016-06-01", "2016-07-15"),
}


def return_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns aligned to `prices` index and columns."""
    return simple_returns(prices)


def full_sample_correlation(returns: pd.DataFrame, min_periods: int | None = None) -> pd.DataFrame:
    """
    Pearson correlation matrix with pairwise-complete observations.

    `min_periods` defaults to max(10, ~10% of rows) so short samples do not
    produce spurious 1.0 correlations.
    """
    r = returns.copy()
    n = int(r.shape[0])
    mp = min_periods if min_periods is not None else max(10, n // 10)
    return r.corr(min_periods=mp)


def slice_by_dates(returns: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    m = (returns.index >= t0) & (returns.index <= t1)
    return returns.loc[m]


def correlation_for_window(
    returns: pd.DataFrame,
    start: str,
    end: str,
    min_periods: int | None = None,
) -> pd.DataFrame:
    sub = slice_by_dates(returns, start, end)
    return full_sample_correlation(sub, min_periods=min_periods)


def rolling_pair_correlation(
    returns: pd.DataFrame,
    col_a: str,
    col_b: str,
    window: int,
) -> pd.Series:
    """Rolling Pearson correlation between two return series."""
    a = returns[col_a]
    b = returns[col_b]
    min_p = max(10, min(window // 2, window))
    return a.rolling(window=window, min_periods=min_p).corr(b)


def rolling_average_pairwise_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    """
    Mean of unique pairwise rolling correlations (upper triangle).

    Summarises how “typical” cross-asset correlation evolves; rises often
    coincide with stress / risk-off episodes (empirical, not guaranteed).
    """
    cols = list(returns.columns)
    if len(cols) < 2:
        return pd.Series(index=returns.index, dtype=float)
    parts: list[pd.Series] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            parts.append(rolling_pair_correlation(returns, cols[i], cols[j], window))
    mat = pd.concat(parts, axis=1)
    return mat.mean(axis=1)


def upper_triangle_mean(corr: pd.DataFrame) -> float:
    """Average pairwise correlation (excludes diagonal)."""
    c = corr.to_numpy()
    k = c.shape[0]
    if k < 2:
        return float("nan")
    tot = 0.0
    cnt = 0
    for i in range(k):
        for j in range(i + 1, k):
            v = c[i, j]
            if pd.notna(v):
                tot += float(v)
                cnt += 1
    return tot / cnt if cnt else float("nan")
