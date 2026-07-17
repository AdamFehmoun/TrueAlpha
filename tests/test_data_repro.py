"""Reproducibility guardrails on the committed dataset.

The core check: the SHA-256 of the data content on disk must match the manifest.
If someone (or something) silently alters the data, or a re-download returns different
history, this fails loudly instead of producing quietly different results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import data.loader
from data.download import END, START, SYMBOLS, TIMEFRAME
from data.integrity import canonical_hash
from data.loader import DataIntegrityError, load_ohlcv, read_manifest
from data.paths import MANIFEST_PATH, RAW_DIR
from tests.utils import make_prices


def test_manifest_exists() -> None:
    assert MANIFEST_PATH.exists(), (
        "manifest.json missing -- run `python -m data.download` and commit the data"
    )


def test_manifest_records_the_absolute_dates_from_code() -> None:
    manifest = read_manifest()
    assert manifest["start"] == START == "2022-01-01T00:00:00Z"
    assert manifest["end"] == END == "2024-12-31T00:00:00Z"
    assert manifest["timeframe"] == TIMEFRAME == "1d"
    # dates must parse as absolute, timezone-aware instants
    assert pd.Timestamp(manifest["start"]).tzinfo is not None
    assert pd.Timestamp(manifest["end"]).tzinfo is not None


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_data_hash_matches_manifest(symbol: str) -> None:
    """THE reproducibility test: content hash on disk == pinned hash in manifest."""
    manifest = read_manifest()
    entry = manifest["datasets"][symbol]
    df = pd.read_parquet(RAW_DIR / entry["file"])
    assert canonical_hash(df) == entry["sha256"]
    assert len(df) == entry["rows"]


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_calendar_is_complete_and_utc(symbol: str) -> None:
    df = load_ohlcv(symbol)
    expected = pd.date_range(
        start=pd.Timestamp(START), end=pd.Timestamp(END), freq="D", name=df.index.name
    )
    assert df.index.equals(expected)
    assert len(df) == 1096  # 2022 (365) + 2023 (365) + 2024 (366)


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_prices_are_positive_and_complete(symbol: str) -> None:
    df = load_ohlcv(symbol)
    assert not df.isna().any().any()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()


def test_hash_is_sensitive_to_single_value_change() -> None:
    df = make_prices([100.0, 101.0, 102.0])
    before = canonical_hash(df)
    tampered = df.copy()
    tampered.loc[tampered.index[1], "open"] = df["open"].iloc[1] + 0.0001
    assert canonical_hash(tampered) != before


def test_loader_raises_on_tampered_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: a manifest/data mismatch must make load_ohlcv raise, not warn."""
    df = make_prices([100.0, 101.0, 102.0])
    df.to_parquet(tmp_path / "FAKE_1d.parquet")
    manifest: dict[str, Any] = {
        "datasets": {"FAKE/USDT": {"file": "FAKE_1d.parquet", "rows": 3, "sha256": "0" * 64}}
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(data.loader, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(data.loader, "RAW_DIR", tmp_path)
    with pytest.raises(DataIntegrityError, match="hash mismatch"):
        load_ohlcv("FAKE/USDT")


def test_loader_accepts_data_matching_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = make_prices([100.0, 101.0, 102.0])
    df.to_parquet(tmp_path / "FAKE_1d.parquet")
    stored = pd.read_parquet(tmp_path / "FAKE_1d.parquet")
    manifest: dict[str, Any] = {
        "datasets": {
            "FAKE/USDT": {
                "file": "FAKE_1d.parquet",
                "rows": 3,
                "sha256": canonical_hash(stored),
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(data.loader, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(data.loader, "RAW_DIR", tmp_path)
    loaded = load_ohlcv("FAKE/USDT")
    # parquet does not persist the index `freq` attribute; data content must be identical
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)
