"""Hash-verified data loading. Every load re-checks the SHA-256 against the manifest,
and re-checks the calendar: any hole that is not explicitly pinned in the manifest
fails loudly, listed by timestamp. Pinned holes stay holes -- nothing is interpolated.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from data.integrity import canonical_hash, expected_calendar
from data.paths import MANIFEST_PATH, RAW_DIR


class DataIntegrityError(RuntimeError):
    """Raised when data on disk does not match the manifest hash or calendar."""


def read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} not found -- run `python -m data.download` first")
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
    return manifest


def _verify_calendar(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    manifest: dict[str, Any],
    entry: dict[str, Any],
) -> None:
    """Every hole must be pinned; every pinned hole must actually be a hole."""
    expected = expected_calendar(str(manifest["start"]), str(manifest["end"][timeframe]), timeframe)
    actual_missing = expected.difference(pd.DatetimeIndex(df.index))
    pinned = pd.DatetimeIndex(pd.to_datetime(entry.get("missing_bars", []), utc=True))
    # a pin outside the expected calendar is a bogus manifest entry, not a data
    # disagreement -- reported separately so the operator hunts the pin, not the disk
    out_of_calendar = pinned.difference(expected)
    if len(out_of_calendar) > 0:
        listed = ", ".join(ts.isoformat() for ts in out_of_calendar[:10])
        raise DataIntegrityError(
            f"{symbol} {timeframe}: {len(out_of_calendar)} pinned bar(s) outside the "
            f"expected {timeframe} calendar -- bogus manifest pin(s): {listed}"
            + ("…" if len(out_of_calendar) > 10 else "")
        )
    unpinned = actual_missing.difference(pinned)
    if len(unpinned) > 0:
        listed = ", ".join(ts.isoformat() for ts in unpinned[:10])
        raise DataIntegrityError(
            f"{symbol} {timeframe}: {len(unpinned)} missing bar(s) NOT pinned in the "
            f"manifest -- data gaps must be declared, never silent: {listed}"
            + ("…" if len(unpinned) > 10 else "")
        )
    phantom = pinned.difference(actual_missing)
    if len(phantom) > 0:
        listed = ", ".join(ts.isoformat() for ts in phantom[:10])
        raise DataIntegrityError(
            f"{symbol} {timeframe}: {len(phantom)} bar(s) pinned as missing but present "
            f"on disk -- manifest and data disagree: {listed}" + ("…" if len(phantom) > 10 else "")
        )


def load_ohlcv(symbol: str, timeframe: str = "1d", verify: bool = True) -> pd.DataFrame:
    """Load a symbol's OHLCV parquet; verify content hash and calendar integrity."""
    manifest = read_manifest()
    try:
        entry: dict[str, Any] = manifest["datasets"][symbol][timeframe]
    except KeyError as exc:
        available = {
            sym: sorted(frames) for sym, frames in sorted(manifest.get("datasets", {}).items())
        }
        raise KeyError(f"{symbol!r} {timeframe!r} not in manifest; available: {available}") from exc

    df = pd.read_parquet(RAW_DIR / entry["file"])
    if verify:
        actual = canonical_hash(df)
        if actual != entry["sha256"]:
            raise DataIntegrityError(
                f"hash mismatch for {symbol} {timeframe}: manifest={entry['sha256'][:16]}… "
                f"actual={actual[:16]}… -- data on disk was modified or corrupted"
            )
        _verify_calendar(symbol, timeframe, df, manifest, entry)
    return df
