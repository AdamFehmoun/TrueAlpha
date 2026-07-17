"""Canonical content hashing for market data.

The hash is computed on a canonical CSV serialization of the DataFrame (fixed float
format, fixed line terminator), not on the parquet bytes: parquet encoding is not
byte-stable across writer versions, while the canonical serialization only depends on
the actual data content.
"""

import hashlib

import pandas as pd


def canonical_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the DataFrame's canonical serialization (index + all columns)."""
    payload = df.to_csv(float_format="%.10f", lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
