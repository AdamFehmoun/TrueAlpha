"""Reproducible OHLCV download: absolute dates, parquet + SHA-256 manifest.

Run with:  python -m data.download

Rules enforced here:
- ABSOLUTE dates only (START/END below). Relative ranges like ``period="2y"`` make the
  dataset depend on the day you ran the script and are banned in this repo.
- Every dataset is validated (complete daily calendar, no NaN, positive prices) before
  being written.
- A manifest records source, dates, download time and the SHA-256 of the data content,
  so any silent change to the data is detected by ``tests/test_data_repro.py``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import ccxt
import pandas as pd

from data.integrity import canonical_hash
from data.paths import MANIFEST_PATH, RAW_DIR

EXCHANGE_ID = "binance"
SYMBOLS = ("BTC/USDT", "ETH/USDT")
TIMEFRAME = "1d"
# Absolute dates only. END is the open time of the last daily candle, inclusive.
START = "2022-01-01T00:00:00Z"
END = "2024-12-31T00:00:00Z"

_DAY_MS = 24 * 60 * 60 * 1000
_COLUMNS = ("open", "high", "low", "close", "volume")


def fetch_ohlcv(exchange: Any, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch daily candles in [start_ms, end_ms] (candle open time, UTC), paginated."""
    rows: list[list[float]] = []
    since = start_ms
    while since <= end_ms:
        batch = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = int(batch[-1][0]) + _DAY_MS
        time.sleep(exchange.rateLimit / 1000.0)

    df = pd.DataFrame(rows, columns=["timestamp", *_COLUMNS])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    start_ts = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_ts = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return df.astype("float64")


def validate_ohlcv(df: pd.DataFrame, start_ms: int, end_ms: int) -> None:
    """Fail loudly if the dataset is incomplete or malformed. No silent gaps."""
    expected_index = pd.date_range(
        start=pd.Timestamp(start_ms, unit="ms", tz="UTC"),
        end=pd.Timestamp(end_ms, unit="ms", tz="UTC"),
        freq="D",
    )
    if not df.index.equals(expected_index):
        missing = expected_index.difference(pd.DatetimeIndex(df.index))
        raise ValueError(
            f"calendar incomplete: {len(missing)} missing daily candles: {missing[:5]}"
        )
    if df[list(_COLUMNS)].isna().any().any():
        raise ValueError("NaN values in OHLCV data")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive prices in OHLCV data")


def main() -> None:
    exchange = ccxt.binance({"enableRateLimit": True})
    start_ms = int(exchange.parse8601(START))
    end_ms = int(exchange.parse8601(END))
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        df = fetch_ohlcv(exchange, symbol, start_ms, end_ms)
        validate_ohlcv(df, start_ms, end_ms)
        filename = f"{symbol.replace('/', '_')}_{TIMEFRAME}.parquet"
        df.to_parquet(RAW_DIR / filename)
        sha = canonical_hash(df)
        datasets[symbol] = {
            "file": filename,
            "rows": int(len(df)),
            "first_timestamp": pd.Timestamp(df.index[0]).isoformat(),
            "last_timestamp": pd.Timestamp(df.index[-1]).isoformat(),
            "sha256": sha,
        }
        print(f"{symbol}: {len(df)} rows, sha256={sha[:16]}…")

    manifest = {
        "source": {
            "exchange": EXCHANGE_ID,
            "via": f"ccxt {ccxt.__version__}",
            "pandas": pd.__version__,
        },
        "timeframe": TIMEFRAME,
        "start": START,
        "end": END,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": datasets,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
