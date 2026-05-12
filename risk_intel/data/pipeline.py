"""Download, clean, validate, and cache adjusted close panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import yfinance as yf

from risk_intel.config import MAX_FORWARD_FILL_SESSIONS
from risk_intel.data import cache as cache_mod


Interval = Literal["1d", "1wk", "1mo"]


@dataclass
class DataManifest:
    cache_key: str
    warnings: list[str]
    rows: int
    tickers: list[str]
    from_cache: bool


def _wide_close_from_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalise yfinance output to a wide Close dataframe (sorted datetime index)."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance multi-ticker: level 0 is usually OHLCV field, level 1 is ticker (or reversed).
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        if "Close" in level0:
            closes = raw["Close"]
        elif "Close" in level1:
            closes = raw.xs("Close", axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        closes = raw[["Close"]].copy()
        closes.columns = [tickers[0]]

    closes = closes.sort_index()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    cols = [c for c in tickers if c in closes.columns]
    closes = closes[cols]
    return closes


def _validate_and_clean(closes: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if closes.empty:
        return closes, ["empty_frame"]

    df = closes.copy()
    df = df[~df.index.duplicated(keep="last")]
    all_nan_rows = df.isna().all(axis=1)
    if all_nan_rows.any():
        dropped = int(all_nan_rows.sum())
        df = df.loc[~all_nan_rows]
        warnings.append(f"dropped_all_nan_rows:{dropped}")

    # Count consecutive NaNs before fill
    for col in df.columns:
        s = df[col]
        is_na = s.isna()
        if not is_na.any():
            continue
        # longest NaN streak
        groups = (is_na != is_na.shift()).cumsum()
        longest = int(is_na.groupby(groups).sum().max())
        if longest > MAX_FORWARD_FILL_SESSIONS:
            warnings.append(f"{col}:longest_nan_streak={longest}")

    filled = df.ffill(limit=MAX_FORWARD_FILL_SESSIONS)
    still_na = filled.isna().sum()
    for col, n in still_na.items():
        if n:
            warnings.append(f"{col}:remaining_na_after_ffill={int(n)}")
    return filled, warnings


def load_or_download(
    tickers: list[str],
    start: str,
    end: str,
    interval: Interval = "1d",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, DataManifest]:
    """
    Return wide adjusted Close panel and manifest.

    Prices use yfinance auto_adjust=True (total-return consistent for broad indices).
    """
    tickers = list(dict.fromkeys(tickers))  # preserve order, unique
    key = cache_mod.cache_key(tickers, start, end, interval)

    if not force_refresh:
        path = cache_mod.get_cached_parquet_path(key)
        if path is not None:
            frame = cache_mod.load_parquet(path)
            rec = cache_mod.describe_cache(key)
            w = rec.warnings if rec else []
            return frame, DataManifest(
                cache_key=key, warnings=w, rows=len(frame), tickers=tickers, from_cache=True
            )

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        threads=True,
        progress=False,
        group_by="column",
    )

    closes = _wide_close_from_download(raw, tickers)
    cleaned, warnings = _validate_and_clean(closes)

    if cleaned.empty and tickers:
        warnings.append("yfinance_returned_empty_or_unparseable")

    cache_mod.write_payload(key, tickers, start, end, interval, cleaned, warnings)
    return cleaned, DataManifest(
        cache_key=key, warnings=warnings, rows=len(cleaned), tickers=tickers, from_cache=False
    )
