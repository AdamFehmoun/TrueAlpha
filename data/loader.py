"""Hash-verified data loading. Every load re-checks the SHA-256 against the manifest."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from data.integrity import canonical_hash
from data.paths import MANIFEST_PATH, RAW_DIR


class DataIntegrityError(RuntimeError):
    """Raised when data on disk does not match the manifest hash."""


def read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} not found -- run `python -m data.download` first")
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
    return manifest


def load_ohlcv(symbol: str, verify: bool = True) -> pd.DataFrame:
    """Load a symbol's OHLCV parquet; verify its content hash against the manifest."""
    manifest = read_manifest()
    try:
        entry = manifest["datasets"][symbol]
    except KeyError as exc:
        available = sorted(manifest.get("datasets", {}))
        raise KeyError(f"{symbol!r} not in manifest; available: {available}") from exc

    df = pd.read_parquet(RAW_DIR / entry["file"])
    if verify:
        actual = canonical_hash(df)
        if actual != entry["sha256"]:
            raise DataIntegrityError(
                f"hash mismatch for {symbol}: manifest={entry['sha256'][:16]}… "
                f"actual={actual[:16]}… -- data on disk was modified or corrupted"
            )
    return df
