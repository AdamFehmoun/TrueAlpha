"""Reproducibility guardrails on the committed datasets (1d and 1h).

The core check: the SHA-256 of the data content on disk must match the manifest.
If someone (or something) silently alters the data, or a re-download returns different
history, this fails loudly instead of producing quietly different results.

Calendar policy: daily data is gap-free, full stop. Hourly data may have
exchange-maintenance holes; every hole must be PINNED in the manifest
(``missing_bars``), an unpinned hole fails loudly and is LISTED by timestamp, and a
pinned hole stays a hole -- no bar is ever interpolated or invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import data.loader
from data.download import END, START, SYMBOLS, TIMEFRAMES
from data.integrity import canonical_hash, expected_calendar
from data.loader import DataIntegrityError, load_ohlcv, read_manifest
from data.paths import MANIFEST_PATH, RAW_DIR
from tests.utils import make_prices

SYMBOL_TIMEFRAMES = [(symbol, tf) for symbol in SYMBOLS for tf in TIMEFRAMES]


def test_manifest_exists() -> None:
    assert MANIFEST_PATH.exists(), (
        "manifest.json missing -- run `python -m data.download` and commit the data"
    )


def test_manifest_records_the_absolute_dates_from_code() -> None:
    manifest = read_manifest()
    assert manifest["start"] == START == "2022-01-01T00:00:00Z"
    assert manifest["end"] == END == {"1d": "2024-12-31T00:00:00Z", "1h": "2024-12-31T23:00:00Z"}
    assert sorted(manifest["timeframes"]) == sorted(TIMEFRAMES) == ["1d", "1h"]
    # dates must parse as absolute, timezone-aware instants
    assert pd.Timestamp(manifest["start"]).tzinfo is not None
    for timeframe in TIMEFRAMES:
        assert pd.Timestamp(manifest["end"][timeframe]).tzinfo is not None


@pytest.mark.parametrize(("symbol", "timeframe"), SYMBOL_TIMEFRAMES)
def test_data_hash_matches_manifest(symbol: str, timeframe: str) -> None:
    """THE reproducibility test: content hash on disk == pinned hash in manifest."""
    manifest = read_manifest()
    entry = manifest["datasets"][symbol][timeframe]
    df = pd.read_parquet(RAW_DIR / entry["file"])
    assert canonical_hash(df) == entry["sha256"]
    assert len(df) == entry["rows"]


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_daily_calendar_is_complete_and_utc(symbol: str) -> None:
    df = load_ohlcv(symbol)
    expected = pd.date_range(
        start=pd.Timestamp(START), end=pd.Timestamp(END["1d"]), freq="D", name=df.index.name
    )
    assert df.index.equals(expected)
    assert len(df) == 1096  # 2022 (365) + 2023 (365) + 2024 (366)
    manifest = read_manifest()
    assert manifest["datasets"][symbol]["1d"]["missing_bars"] == []


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_hourly_calendar_is_the_full_span_minus_pinned_holes_only(symbol: str) -> None:
    """26,304 expected hourly bars (1096 days x 24); every absent bar is pinned."""
    manifest = read_manifest()
    entry = manifest["datasets"][symbol]["1h"]
    df = load_ohlcv(symbol, timeframe="1h")  # loader re-verifies hash AND calendar
    expected = expected_calendar(START, END["1h"], "1h")
    assert len(expected) == 26_304
    pinned = pd.DatetimeIndex(pd.to_datetime(entry["missing_bars"], utc=True))
    assert df.index.equals(expected.difference(pinned))
    assert len(df) == 26_304 - len(pinned)


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_pinned_hourly_holes_stay_holes(symbol: str) -> None:
    """No interpolation, ever: a pinned missing bar must be truly absent on disk."""
    manifest = read_manifest()
    entry = manifest["datasets"][symbol]["1h"]
    df = load_ohlcv(symbol, timeframe="1h")
    for ts in entry["missing_bars"]:
        assert pd.Timestamp(ts) not in df.index


@pytest.mark.parametrize(("symbol", "timeframe"), SYMBOL_TIMEFRAMES)
def test_prices_are_positive_and_have_no_nan(symbol: str, timeframe: str) -> None:
    df = load_ohlcv(symbol, timeframe=timeframe)
    assert not df.isna().any().any()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()


def test_hash_is_sensitive_to_single_value_change() -> None:
    df = make_prices([100.0, 101.0, 102.0])
    before = canonical_hash(df)
    tampered = df.copy()
    tampered.loc[tampered.index[1], "open"] = df["open"].iloc[1] + 0.0001
    assert canonical_hash(tampered) != before


def _install_fake_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    df: pd.DataFrame,
    missing_bars: list[str],
    sha256: str | None = None,
) -> pd.DataFrame:
    """Write a parquet + v2 manifest for FAKE/USDT over 2023-01-01..df-span (daily)."""
    df.to_parquet(tmp_path / "FAKE_1d.parquet")
    stored = pd.read_parquet(tmp_path / "FAKE_1d.parquet")
    end = pd.Timestamp("2023-01-01T00:00:00Z") + pd.Timedelta(days=3)
    manifest: dict[str, Any] = {
        "start": "2023-01-01T00:00:00Z",
        "end": {"1d": end.isoformat()},
        "timeframes": ["1d"],
        "datasets": {
            "FAKE/USDT": {
                "1d": {
                    "file": "FAKE_1d.parquet",
                    "rows": int(len(stored)),
                    "missing_bars": missing_bars,
                    "sha256": sha256 if sha256 is not None else canonical_hash(stored),
                }
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(data.loader, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(data.loader, "RAW_DIR", tmp_path)
    return stored


def test_loader_raises_on_tampered_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: a manifest/data hash mismatch must make load_ohlcv raise, not warn."""
    df = make_prices([100.0, 101.0, 102.0, 103.0])
    _install_fake_manifest(tmp_path, monkeypatch, df, missing_bars=[], sha256="0" * 64)
    with pytest.raises(DataIntegrityError, match="hash mismatch"):
        load_ohlcv("FAKE/USDT")


def test_loader_accepts_data_matching_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = make_prices([100.0, 101.0, 102.0, 103.0])
    _install_fake_manifest(tmp_path, monkeypatch, df, missing_bars=[])
    loaded = load_ohlcv("FAKE/USDT")
    # parquet does not persist the index `freq` attribute; data content must be identical
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_loader_fails_loudly_and_lists_unpinned_holes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bar absent on disk but NOT pinned in the manifest fails and is LISTED."""
    df = make_prices([100.0, 101.0, 102.0, 103.0])
    gapped = df.drop(index=df.index[2])  # 2023-01-03 silently missing
    _install_fake_manifest(tmp_path, monkeypatch, gapped, missing_bars=[])
    with pytest.raises(DataIntegrityError, match="2023-01-03"):
        load_ohlcv("FAKE/USDT")


def test_loader_accepts_pinned_holes_and_keeps_them_as_holes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = make_prices([100.0, 101.0, 102.0, 103.0])
    gapped = df.drop(index=df.index[2])
    _install_fake_manifest(
        tmp_path, monkeypatch, gapped, missing_bars=["2023-01-03T00:00:00+00:00"]
    )
    loaded = load_ohlcv("FAKE/USDT")
    assert len(loaded) == 3
    assert pd.Timestamp("2023-01-03T00:00:00Z") not in loaded.index  # still a hole


def test_loader_rejects_phantom_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bar pinned as missing but actually present = manifest/data disagreement."""
    df = make_prices([100.0, 101.0, 102.0, 103.0])
    _install_fake_manifest(tmp_path, monkeypatch, df, missing_bars=["2023-01-03T00:00:00+00:00"])
    with pytest.raises(DataIntegrityError, match="present"):
        load_ohlcv("FAKE/USDT")


def test_loader_rejects_pins_outside_the_calendar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pin before start / after end is a bogus manifest entry and is named as such,
    not mislabeled as a bar 'present on disk' the operator would then hunt in vain."""
    df = make_prices([100.0, 101.0, 102.0, 103.0])
    _install_fake_manifest(tmp_path, monkeypatch, df, missing_bars=["2021-06-15T00:00:00+00:00"])
    with pytest.raises(DataIntegrityError, match="outside the expected"):
        load_ohlcv("FAKE/USDT")
