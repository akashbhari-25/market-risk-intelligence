"""SQLite manifest + Parquet payloads for reproducible data pulls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from risk_intel.config import DATA_CACHE_DIR


MANIFEST_DB = DATA_CACHE_DIR / "manifest.sqlite3"


def _ensure_dirs() -> None:
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_key(tickers: list[str], start: str, end: str, interval: str) -> str:
    payload = json.dumps(
        {"tickers": sorted(tickers), "start": start, "end": end, "interval": interval},
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _connect() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(MANIFEST_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fetches (
            cache_key TEXT PRIMARY KEY,
            tickers_json TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            interval TEXT NOT NULL,
            parquet_path TEXT NOT NULL,
            row_count INTEGER,
            warnings_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get_cached_parquet_path(key: str) -> Path | None:
    conn = _connect()
    row = conn.execute(
        "SELECT parquet_path FROM fetches WHERE cache_key = ?", (key,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    path = Path(row[0])
    return path if path.is_file() else None


def write_payload(
    key: str,
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
    frame: pd.DataFrame,
    warnings: list[str],
) -> Path:
    _ensure_dirs()
    parquet_path = DATA_CACHE_DIR / f"{key}.parquet"
    frame.to_parquet(parquet_path)
    conn = _connect()
    conn.execute(
        """
        INSERT OR REPLACE INTO fetches
        (cache_key, tickers_json, start_date, end_date, interval, parquet_path,
         row_count, warnings_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            key,
            json.dumps(tickers),
            start,
            end,
            interval,
            str(parquet_path),
            int(len(frame)),
            json.dumps(warnings),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return parquet_path


@dataclass
class CacheRecord:
    tickers: list[str]
    start: str
    end: str
    interval: str
    parquet_path: Path
    row_count: int
    warnings: list[str]
    created_at: str


def describe_cache(key: str) -> CacheRecord | None:
    conn = _connect()
    row = conn.execute(
        """
        SELECT tickers_json, start_date, end_date, interval, parquet_path,
               row_count, warnings_json, created_at
        FROM fetches WHERE cache_key = ?
        """,
        (key,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    tj, sd, ed, iv, pp, rc, wj, ca = row
    return CacheRecord(
        tickers=json.loads(tj),
        start=sd,
        end=ed,
        interval=iv,
        parquet_path=Path(pp),
        row_count=int(rc or 0),
        warnings=list(json.loads(wj or "[]")),
        created_at=ca,
    )


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
