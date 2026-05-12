"""Defaults and constants — single source of truth for Phase 1."""

from __future__ import annotations

from pathlib import Path

# Ten liquid global equity proxies (expand later via UI / config file)
DEFAULT_TICKERS: tuple[str, ...] = (
    "^GSPC",   # S&P 500
    "^IXIC",   # Nasdaq Composite
    "^DJI",    # Dow Jones
    "^FTSE",   # FTSE 100
    "^GDAXI",  # DAX
    "^FCHI",   # CAC 40
    "^N225",   # Nikkei 225
    "^HSI",    # Hang Seng
    "^AXJO",   # ASX 200
    "^BSESN",  # S&P BSE Sensex
)

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_ANNUAL_DEFAULT = 0.04  # 4% — user-editable in UI

# Project root (parent of package)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = PACKAGE_ROOT / "data_cache"

MAX_FORWARD_FILL_SESSIONS = 5
