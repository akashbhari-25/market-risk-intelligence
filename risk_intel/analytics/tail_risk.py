"""Tail risk and return distribution analytics (Phase 2A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, norm

from risk_intel.analytics.returns import simple_returns
from risk_intel.config import TRADING_DAYS_PER_YEAR


def _require_returns(simple: pd.Series, min_obs: int = 30) -> pd.Series:
    r = simple.dropna()
    return r if len(r) >= min_obs else pd.Series(dtype=float)


def historical_var_cvar(
    simple: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Historical (non-parametric) one-period VaR and CVaR on **simple** returns.

    **Convention:** both values are **positive** and represent **loss magnitude**
    in the same units as returns (e.g. 0.02 = 2% one-day loss at the threshold).

    - VaR: loss level at the (1-confidence) empirical quantile of the return distribution.
    - CVaR / ES: mean loss in the tail, conditional on being at or beyond that quantile.
    """
    r = _require_returns(simple)
    if r.empty:
        return float("nan"), float("nan")
    arr = r.to_numpy(dtype=float)
    q_pct = (1.0 - float(confidence)) * 100.0
    cutoff = float(np.percentile(arr, q_pct))
    var = float(-cutoff)
    tail = arr[arr <= cutoff]
    if tail.size == 0:
        cvar = var
    else:
        cvar = float(-np.mean(tail))
    return var, cvar


def parametric_normal_var_cvar(
    simple: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Gaussian VaR / CVaR on simple returns (same loss sign convention as historical).

    CVaR under normality uses the closed-form tail conditional mean.
    """
    r = _require_returns(simple)
    if r.empty:
        return float("nan"), float("nan")
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma == 0 or np.isnan(sigma):
        return float("nan"), float("nan")
    alpha = 1.0 - float(confidence)
    z = norm.ppf(alpha)
    # VaR loss (positive): - (mu + z*sigma)  with z negative for left tail
    var = float(-(mu + z * sigma))
    # ES for normal: mu - sigma * phi(z) / alpha  (loss convention positive)
    phi_z = float(norm.pdf(z))
    es_level = mu - sigma * (phi_z / alpha)
    cvar = float(-es_level)
    return var, cvar


def skew_excess_kurtosis(simple: pd.Series) -> tuple[float, float]:
    """Sample skewness and **excess** kurtosis (Fisher definition)."""
    r = _require_returns(simple, min_obs=10)
    if r.empty:
        return float("nan"), float("nan")
    sk = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=True, bias=False))
    return sk, kurt


def jarque_bera_test(simple: pd.Series) -> tuple[float, float]:
    """Jarque-Bera statistic and two-sided p-value."""
    r = _require_returns(simple, min_obs=10)
    if r.empty:
        return float("nan"), float("nan")
    jb_stat, pvalue = jarque_bera(r)
    return float(jb_stat), float(pvalue)


def scale_var_sqrt_time(var_1p: float, periods: int) -> float:
    """IID √T scaling (common desk shortcut; not a copula model)."""
    if np.isnan(var_1p):
        return float("nan")
    return float(var_1p * np.sqrt(float(periods)))


def compute_tail_risk_table(
    prices: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Per-ticker tail metrics on **simple** returns derived from `prices`.

    Includes daily VaR/CVaR (historical + Gaussian), skew, excess kurtosis, JB test,
    and **annualised** VaR/CVaR using √252 scaling (documented shortcut).
    """
    sr = simple_returns(prices)
    rows: list[dict[str, float | str]] = []

    for col in prices.columns:
        s = sr[col]
        h_var, h_cvar = historical_var_cvar(s, confidence=confidence)
        p_var, p_cvar = parametric_normal_var_cvar(s, confidence=confidence)
        sk, ek = skew_excess_kurtosis(s)
        jb, jb_p = jarque_bera_test(s)

        rows.append(
            {
                "ticker": col,
                f"hist_var_{int(confidence*100)}_1d": h_var,
                f"hist_cvar_{int(confidence*100)}_1d": h_cvar,
                f"gauss_var_{int(confidence*100)}_1d": p_var,
                f"gauss_cvar_{int(confidence*100)}_1d": p_cvar,
                f"hist_var_{int(confidence*100)}_ann_sqrt252": scale_var_sqrt_time(h_var, TRADING_DAYS_PER_YEAR),
                f"hist_cvar_{int(confidence*100)}_ann_sqrt252": scale_var_sqrt_time(h_cvar, TRADING_DAYS_PER_YEAR),
                "skewness": sk,
                "excess_kurtosis": ek,
                "jarque_bera": jb,
                "jarque_bera_pvalue": jb_p,
                "n_obs_returns": int(s.dropna().shape[0]),
            }
        )

    return pd.DataFrame(rows).set_index("ticker")


def qq_plot_points(simple: pd.Series, n_max: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Ordered sample quantiles vs normal theoretical quantiles (subsample for speed)."""
    r = _require_returns(simple)
    if r.empty:
        return np.array([]), np.array([])
    x = r.to_numpy(dtype=float)
    if x.size > n_max:
        rng = np.random.default_rng(42)
        x = rng.choice(x, size=n_max, replace=False)
    x = np.sort(x)
    n = x.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = norm.ppf(probs)
    return theoretical, x
