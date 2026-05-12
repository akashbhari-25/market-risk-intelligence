"""Return series helpers — log and simple returns from price panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns: P_t / P_{t-1} - 1."""
    return prices.pct_change()
