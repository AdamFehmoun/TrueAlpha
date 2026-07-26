"""Canonical content hashing and calendar integrity for market data.

The hash is computed on a canonical CSV serialization of the DataFrame (fixed float
format, fixed line terminator), not on the parquet bytes: parquet encoding is not
byte-stable across writer versions, while the canonical serialization only depends on
the actual data content.

Calendar integrity: every dataset is compared against the COMPLETE bar-open calendar
of its timeframe. Missing bars (exchange maintenance) are never interpolated; they
must be explicitly pinned in the manifest, and any unpinned hole fails loudly, listed
by timestamp.
"""

import hashlib
from typing import Final

import pandas as pd

PANDAS_FREQ: Final[dict[str, str]] = {"1d": "D", "1h": "h"}


def canonical_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the DataFrame's canonical serialization (index + all columns)."""
    payload = df.to_csv(float_format="%.10f", lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_calendar(start: str, end: str, timeframe: str) -> pd.DatetimeIndex:
    """The complete bar-open calendar for [start, end] (inclusive, UTC instants)."""
    try:
        freq = PANDAS_FREQ[timeframe]
    except KeyError as exc:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; supported: {sorted(PANDAS_FREQ)}"
        ) from exc
    return pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq=freq)


def find_missing_bars(
    index: pd.DatetimeIndex, start: str, end: str, timeframe: str
) -> pd.DatetimeIndex:
    """Bars of the expected calendar that are absent from ``index`` (exchange gaps)."""
    return expected_calendar(start, end, timeframe).difference(index)
