"""Reproducible OHLCV download: absolute dates, parquet + SHA-256 manifest.

Run with:  python -m data.download [--timeframe 1d|1h|all]

Rules enforced here:
- ABSOLUTE dates only (START/END below). Relative ranges like ``period="2y"`` make the
  dataset depend on the day you ran the script and are banned in this repo.
- Every dataset is validated (calendar, no NaN, positive prices) before being written.
- Daily data must have a COMPLETE calendar. Hourly data may have exchange-maintenance
  holes: they are detected, printed loudly, and PINNED in the manifest
  (``missing_bars``) -- never interpolated, never filled. Any hole that is not pinned
  makes the verified loader fail and list it.
- A manifest records source, dates, download time and the SHA-256 of the data content,
  so any silent change to the data is detected by ``tests/test_data_repro.py``.
- Re-downloading one timeframe preserves the other timeframes' manifest entries
  verbatim (the 1d entries -- and therefore every 1d number -- never move as a side
  effect of adding 1h data).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Final

import ccxt
import pandas as pd

from data.integrity import canonical_hash, expected_calendar, find_missing_bars
from data.paths import MANIFEST_PATH, RAW_DIR

EXCHANGE_ID = "binance"
SYMBOLS = ("BTC/USDT", "ETH/USDT")
TIMEFRAMES: Final = ("1d", "1h")
# Absolute dates only. END is the open time of the LAST candle, inclusive; both
# timeframes cover the identical calendar span (the last hourly candle of
# 2024-12-31 opens at 23:00).
START = "2022-01-01T00:00:00Z"
END: Final[dict[str, str]] = {"1d": "2024-12-31T00:00:00Z", "1h": "2024-12-31T23:00:00Z"}

_TIMEFRAME_MS: Final[dict[str, int]] = {"1d": 86_400_000, "1h": 3_600_000}
_COLUMNS = ("open", "high", "low", "close", "volume")


def fetch_ohlcv(
    exchange: Any, symbol: str, timeframe: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """Fetch candles in [start_ms, end_ms] (candle open time, UTC), paginated."""
    step_ms = _TIMEFRAME_MS[timeframe]
    rows: list[list[float]] = []
    since = start_ms
    while since <= end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = int(batch[-1][0]) + step_ms
        time.sleep(exchange.rateLimit / 1000.0)

    df = pd.DataFrame(rows, columns=["timestamp", *_COLUMNS])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    start_ts = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_ts = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return df.astype("float64")


def validate_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DatetimeIndex:
    """Fail loudly on malformed data; return the (possibly empty) list of holes.

    Daily data must be gap-free (hard guarantee, unchanged). Hourly holes are
    returned so the caller PINS them in the manifest -- kept as holes, never filled.
    """
    expected = expected_calendar(START, END[timeframe], timeframe)
    unexpected = pd.DatetimeIndex(df.index).difference(expected)
    if len(unexpected) > 0:
        raise ValueError(
            f"{len(unexpected)} bars outside the expected {timeframe} calendar: "
            f"{list(unexpected[:5])}"
        )
    missing = find_missing_bars(pd.DatetimeIndex(df.index), START, END[timeframe], timeframe)
    if timeframe == "1d" and len(missing) > 0:
        raise ValueError(
            f"calendar incomplete: {len(missing)} missing daily candles: {list(missing[:5])}"
        )
    if df[list(_COLUMNS)].isna().any().any():
        raise ValueError("NaN values in OHLCV data")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive prices in OHLCV data")
    return missing


def _load_or_init_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
        return manifest
    return {
        "source": {},
        "start": START,
        "end": dict(END),
        "timeframes": list(TIMEFRAMES),
        "downloaded_at": {},
        "datasets": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeframe",
        choices=[*TIMEFRAMES, "all"],
        default="all",
        help="download one timeframe only, preserving the others' manifest entries",
    )
    args = parser.parse_args()
    requested = TIMEFRAMES if args.timeframe == "all" else (args.timeframe,)

    exchange = ccxt.binance({"enableRateLimit": True})
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_init_manifest()
    manifest["source"] = {
        "exchange": EXCHANGE_ID,
        "via": f"ccxt {ccxt.__version__}",
        "pandas": pd.__version__,
    }
    manifest["start"] = START
    manifest["end"] = dict(END)
    manifest["timeframes"] = list(TIMEFRAMES)

    for timeframe in requested:
        start_ms = int(exchange.parse8601(START))
        end_ms = int(exchange.parse8601(END[timeframe]))
        for symbol in SYMBOLS:
            df = fetch_ohlcv(exchange, symbol, timeframe, start_ms, end_ms)
            missing = validate_ohlcv(df, timeframe)
            if len(missing) > 0:
                print(f"!! {symbol} {timeframe}: {len(missing)} missing bars, kept as holes")
                for ts in missing:
                    print(f"!!   missing: {ts.isoformat()}")
            filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
            df.to_parquet(RAW_DIR / filename)
            sha = canonical_hash(df)
            manifest.setdefault("datasets", {}).setdefault(symbol, {})[timeframe] = {
                "file": filename,
                "rows": int(len(df)),
                "first_timestamp": pd.Timestamp(df.index[0]).isoformat(),
                "last_timestamp": pd.Timestamp(df.index[-1]).isoformat(),
                "missing_bars": [ts.isoformat() for ts in missing],
                "sha256": sha,
            }
            print(f"{symbol} {timeframe}: {len(df)} rows, sha256={sha[:16]}…")
        manifest.setdefault("downloaded_at", {})[timeframe] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
