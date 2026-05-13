"""Historical stress-window analytics on price panels (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_intel.analytics.correlation import CRISIS_WINDOWS
from risk_intel.analytics.regime import drawdown_series
from risk_intel.analytics.returns import simple_returns
from risk_intel.config import TRADING_DAYS_PER_YEAR


def periods_per_year_for_interval(interval: str) -> int:
    """Annualisation factor matching the sampling frequency."""
    if interval == "1wk":
        return 52
    if interval == "1mo":
        return 12
    return TRADING_DAYS_PER_YEAR


def _rf_daily_from_annual(rf_annual: float) -> float:
    return (1.0 + float(rf_annual)) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


def slice_panel(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    return panel.loc[(panel.index >= t0) & (panel.index <= t1)]


def _window_sharpe(r: pd.Series, rf_annual: float, periods_per_year: int) -> float:
    x = r.dropna()
    if len(x) < 5:
        return float("nan")
    rf_d = _rf_daily_from_annual(rf_annual)
    excess = x - rf_d
    std = float(excess.std(ddof=1))
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(float(periods_per_year)))


def _row_for_price_returns(
    name: str,
    px: pd.Series,
    r: pd.Series,
    periods_per_year: int,
    rf_annual: float,
) -> dict[str, float | str]:
    px_c = px.dropna()
    r_c = r.reindex(px_c.index).dropna()
    if len(px_c) < 2 or len(r_c) < 2:
        return {
            "name": name,
            "cum_return": float("nan"),
            "max_drawdown": float("nan"),
            "worst_period_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "n_periods": int(len(r_c)),
        }

    cum_ret = float(px_c.iloc[-1] / px_c.iloc[0] - 1.0)
    mdd = float(drawdown_series(px_c).min())
    worst = float(r_c.min())
    ann_vol = float(r_c.std(ddof=1) * np.sqrt(float(periods_per_year)))
    sh = _window_sharpe(r_c, rf_annual, periods_per_year)

    return {
        "name": name,
        "cum_return": cum_ret,
        "max_drawdown": mdd,
        "worst_period_return": worst,
        "ann_vol": ann_vol,
        "sharpe": sh,
        "n_periods": int(len(r_c)),
    }


def stress_table_for_window(
    panel: pd.DataFrame,
    start: str,
    end: str,
    interval: str,
    rf_annual: float,
) -> pd.DataFrame:
    """
    Per-asset stress stats plus an equal-weight **daily-rebalanced** proxy portfolio
    using cross-sectional mean of simple returns each period.
    """
    px_s = slice_panel(panel, start, end)
    if px_s.empty:
        return pd.DataFrame()

    rets = simple_returns(px_s)
    ppy = periods_per_year_for_interval(interval)

    rows: list[dict[str, float | str]] = []
    for col in px_s.columns:
        rows.append(
            _row_for_price_returns(str(col), px_s[col], rets[col], ppy, rf_annual)
        )

    ew = rets.mean(axis=1).dropna()
    if not ew.empty:
        wealth = (1.0 + ew).cumprod()
        rows.append(
            _row_for_price_returns(
                "Equal-weight (1/N, rebalanced each period)",
                wealth,
                ew,
                ppy,
                rf_annual,
            )
        )

    out = pd.DataFrame(rows).set_index("name")
    return out


def stress_scenario_names() -> list[str]:
    return ["Full sample (current fetch)"] + list(CRISIS_WINDOWS.keys())


def stress_table_for_scenario(
    panel: pd.DataFrame,
    scenario: str,
    interval: str,
    rf_annual: float,
) -> pd.DataFrame:
    if scenario == "Full sample (current fetch)":
        start = str(panel.index.min().date())
        end = str(panel.index.max().date())
        return stress_table_for_window(panel, start, end, interval, rf_annual)
    if scenario not in CRISIS_WINDOWS:
        return pd.DataFrame()
    t0, t1 = CRISIS_WINDOWS[scenario]
    return stress_table_for_window(panel, t0, t1, interval, rf_annual)
