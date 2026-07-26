"""Regenerate the auto-generated results: README sections + results/metrics.json.

Every number displayed in the README comes from this script -- numbers are never
written by hand. Two README blocks are generated (the full-sample baselines between
the RESULTS markers and the walk-forward out-of-sample evaluation between the
WALKFORWARD markers), plus ``results/metrics.json``: the golden numbers file, every
metric at full float precision, committed and asserted byte-for-byte (zero
tolerance) by ``tests/test_metrics_golden.py``. README and metrics.json are rendered
from ONE shared computation, so they cannot disagree with each other. Run:

    python -m scripts.generate_results          # rewrite README sections + metrics.json
    python -m scripts.generate_results --check  # fail if either is out of sync (CI)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from data.download import END, START, SYMBOLS
from data.loader import load_ohlcv, read_manifest
from engine.backtest import BacktestConfig, run_backtest
from engine.evaluate import WalkForwardResult, walk_forward_evaluate
from engine.metrics import (
    bars_per_year,
    calendar_bars_per_year,
    sharpe_ratio,
    sharpe_standard_error,
    sharpe_tstat,
    summarize,
)
from engine.splits import walk_forward_splits
from strategies import buy_and_hold_signal, ma_crossover_signal

README_PATH = Path(__file__).resolve().parent.parent / "README.md"
METRICS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics.json"
BEGIN_MARK = "<!-- RESULTS:BEGIN -->"
END_MARK = "<!-- RESULTS:END -->"
WF_BEGIN_MARK = "<!-- WALKFORWARD:BEGIN -->"
WF_END_MARK = "<!-- WALKFORWARD:END -->"

# Every result below is computed on DAILY data. The hourly datasets are pinned in the
# manifest for the next milestone; they feed no number here yet.
RESULTS_TIMEFRAME: Final = "1d"

CONFIG = BacktestConfig(fee_bps=10.0, slippage_bps=0.0)

STRATEGIES: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
    ("Buy & hold", buy_and_hold_signal),
    # named to match the code: the signal is {0, 1} = long-only (long/flat), never short
    ("MA crossover 20/50 (long-only)", ma_crossover_signal),
]

# --- walk-forward protocol: declared here, before any result ------------------- #
WF_N_FOLDS = 5
WF_TEST_SIZE = 0.2
# purge = slow_max, the longest look-back any grid candidate uses; never reduced to
# make folds fit (the split constructor raises instead)
WF_PURGE = 200
WF_EMBARGO = 5
WF_SELECTION_METRIC = "sharpe"
WF_GRID: list[dict[str, Any]] = [
    {"fast": fast, "slow": slow}
    for fast in (5, 10, 20, 50)
    for slow in (20, 50, 100, 200)
    if fast < slow
]
# both walk-forward configurations are published: anchored is the pre-registered
# protocol; rolling exercises the fixed-length-train path on real data and is the
# configuration where the selection has a chance to move (early regimes slide out
# of the train instead of being diluted forever)
WF_CONFIG_LABELS: Final[tuple[tuple[str, bool, str], ...]] = (
    ("anchored", True, "expanding train — the pre-registered protocol"),
    ("rolling", False, "fixed-length sliding train"),
)

# --- pre-registered expectations (see the README's dated pre-registration) ----- #
PREREG_SHARPE_LO = -0.3
PREREG_SHARPE_HI = 0.4
PREREG_TSTAT_MAX = 1.0
PREREG_ALARM_SHARPE = 1.0


@dataclass(frozen=True)
class Computed:
    """Everything the README blocks and metrics.json are rendered from -- computed once."""

    manifest: dict[str, Any]
    prices: dict[str, pd.DataFrame]  # 1d, hash- and calendar-verified loads
    baseline_rows: list[dict[str, Any]]
    walkforward: dict[str, dict[str, WalkForwardResult]]  # symbol -> config -> result
    splice: dict[str, dict[str, dict[str, Any]]]  # symbol -> config -> splice stats


def _splice_stats(prices: pd.DataFrame, result: WalkForwardResult) -> dict[str, Any]:
    """Quantify the artifact the splice rule removes, by running the alternative.

    Re-runs every fold STANDALONE (forced flat restart -- the pre-splice
    convention) and reports what that convention would have cost on this run:
    how many same-parameter boundaries actually carried a position, and the OOS
    total-return delta in bps (phantom re-entry fees + forced-flat first bars).
    """
    standalone_returns: list[pd.Series] = []
    for f in result.folds:
        window = f.split.test
        history = prices.iloc[: int(window[-1]) + 1]
        signal = _ma_factory(history, f.params)
        standalone_returns.append(
            run_backtest(prices.iloc[window], signal.iloc[window], CONFIG).returns
        )
    standalone_series = pd.concat(standalone_returns)
    standalone_total = float((1.0 + standalone_series).prod() - 1.0)
    spliced_total = float(result.metrics["total_return"])

    boundaries = 0
    for k in range(1, len(result.folds)):
        prev, cur = result.folds[k - 1], result.folds[k]
        contiguous = int(cur.split.test[0]) == int(prev.split.test[-1]) + 1
        if contiguous and cur.params == prev.params:
            boundaries += 1
    return {
        "same_param_boundaries": boundaries,
        # single source of truth: the engine counts carried boundaries itself
        # (WalkForwardResult.n_carried_boundaries); this script only publishes it
        "carried_positions": result.n_carried_boundaries,
        "spliced_total_return": spliced_total,
        "standalone_total_return": standalone_total,
        "standalone_sharpe": sharpe_ratio(standalone_series, RESULTS_TIMEFRAME),
        "standalone_sharpe_se": sharpe_standard_error(standalone_series, RESULTS_TIMEFRAME),
        "splice_delta_bps": (spliced_total - standalone_total) * 1e4,
    }


def compute_all() -> Computed:
    manifest = read_manifest()
    prices = {symbol: load_ohlcv(symbol, timeframe=RESULTS_TIMEFRAME) for symbol in SYMBOLS}

    rows: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for name, strategy in STRATEGIES:
            result = run_backtest(prices[symbol], strategy(prices[symbol]), CONFIG)
            metrics = summarize(result, RESULTS_TIMEFRAME)
            rows.append({"symbol": symbol, "name": name, **metrics})

    walkforward: dict[str, dict[str, WalkForwardResult]] = {}
    splice_stats: dict[str, dict[str, dict[str, Any]]] = {}
    for symbol in SYMBOLS:
        walkforward[symbol] = {}
        splice_stats[symbol] = {}
        # the manifest's pinned missing_bars list IS the hole whitelist consumed by
        # the calendar-aware validator (empty for the gap-free daily data); it is
        # generated by data.download and cross-checked by the loader, never by hand
        known_holes = pd.DatetimeIndex(
            pd.to_datetime(
                list(manifest["datasets"][symbol][RESULTS_TIMEFRAME]["missing_bars"]), utc=True
            )
        )
        for config_name, anchored, _desc in WF_CONFIG_LABELS:
            splits = walk_forward_splits(
                len(prices[symbol]),
                WF_N_FOLDS,
                WF_TEST_SIZE,
                WF_PURGE,
                WF_EMBARGO,
                anchored=anchored,
            )
            wf_result = walk_forward_evaluate(
                prices[symbol],
                _ma_factory,
                WF_GRID,
                splits,
                CONFIG,
                WF_SELECTION_METRIC,
                RESULTS_TIMEFRAME,
                known_holes=known_holes,
            )
            walkforward[symbol][config_name] = wf_result
            splice_stats[symbol][config_name] = _splice_stats(prices[symbol], wf_result)
    return Computed(
        manifest=manifest,
        prices=prices,
        baseline_rows=rows,
        walkforward=walkforward,
        splice=splice_stats,
    )


def _pct(x: float) -> str:
    return "—" if not math.isfinite(x) else f"{x:+.2%}"


def _pct_unsigned(x: float) -> str:
    return "—" if not math.isfinite(x) else f"{x:.2%}"


def _num(x: float) -> str:
    return "—" if not math.isfinite(x) else f"{x:.2f}"


def _notes(rows: list[dict[str, Any]]) -> list[str]:
    """Interpretation notes, generated from the same numbers as the table."""
    notes: list[str] = []
    with_t = [r for r in rows if math.isfinite(r["t_stat"])]
    if with_t:
        worst = max(with_t, key=lambda r: abs(float(r["t_stat"])))
        if all(abs(float(r["t_stat"])) < 2.0 for r in with_t):
            notes.append(
                f"**Significance:** every baseline has |t-stat| < 2 "
                f"(max: {abs(worst['t_stat']):.2f}, {worst['symbol']} {worst['name']}) — "
                "none is statistically distinguishable from zero. That is the honest "
                "takeaway for untuned baselines, and it is stated deliberately."
            )
        else:
            significant = ", ".join(
                f"{r['symbol']} {r['name']} (t = {r['t_stat']:.2f})"
                for r in with_t
                if abs(float(r["t_stat"])) >= 2.0
            )
            notes.append(
                f"**Significance:** |t-stat| ≥ 2 for: {significant}. "
                "All other rows are not statistically distinguishable from zero."
            )
    for r in rows:
        if math.isfinite(r["sharpe"]) and r["sharpe"] > 0.0 and r["total_return"] < 0.0:
            notes.append(
                f"**Volatility drag:** {r['symbol']} {r['name']} shows a positive Sharpe "
                f"({r['sharpe']:.2f}) with a negative total return "
                f"({r['total_return']:+.2%}): the arithmetic mean of per-bar returns is "
                "positive while the compounded (geometric) return is negative. "
                "Expected behavior, not a bug."
            )
    return notes


def build_block(computed: Computed) -> str:
    timeframe = RESULTS_TIMEFRAME
    rows = computed.baseline_rows

    lines = [
        BEGIN_MARK,
        "_Generated by `python -m scripts.generate_results` — do not edit by hand"
        " (CI enforces sync)._",
        "",
        f"**Period:** {START[:10]} → {END[timeframe][:10]} ({timeframe} candles, UTC) · "
        f"**Costs:** {CONFIG.fee_bps:.0f} bps fee + {CONFIG.slippage_bps:.0f} bps slippage"
        " per trade · **Execution:** signal at close[t], fill at open[t+1] · "
        f"**Annualization:** √{bars_per_year(timeframe)} · "
        "**Risk-free rate:** 0 (Sharpe, Sortino, t-stat) · "
        f"**t-stat:** Sharpe · √(n_returns / {calendar_bars_per_year(timeframe)})",
        "",
        "| Symbol | Strategy | Total return | CAGR | Sharpe | t-stat | Sortino"
        " | Max drawdown | Ann. turnover | Win rate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['symbol']} | {r['name']} | {_pct(r['total_return'])} | {_pct(r['cagr'])} "
            f"| {_num(r['sharpe'])} | {_num(r['t_stat'])} | {_num(r['sortino'])} "
            f"| {_pct(r['max_drawdown'])} | {r['annualized_turnover']:.1f} "
            f"| {_pct_unsigned(r['win_rate'])} |"
        )
    for note in _notes(rows):
        lines += ["", note]
    hashes = " · ".join(
        f"{symbol} `{computed.manifest['datasets'][symbol][timeframe]['sha256'][:12]}…`"
        for symbol in SYMBOLS
    )
    lines += ["", f"Data integrity (SHA-256): {hashes}", END_MARK]
    return "\n".join(lines)


def _ma_factory(prices: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    return ma_crossover_signal(prices, fast=int(params["fast"]), slow=int(params["slow"]))


def _fold_rows(prices: pd.DataFrame, result: WalkForwardResult) -> list[str]:
    rows: list[str] = []
    for f in result.folds:
        train, test = f.split.train, f.split.test
        train_0, train_1 = prices.index[train[0]], prices.index[train[-1]]
        test_0, test_1 = prices.index[test[0]], prices.index[test[-1]]
        test_sharpe = sharpe_ratio(f.test_result.returns, result.timeframe)
        test_tstat = sharpe_tstat(f.test_result.returns, result.timeframe)
        rows.append(
            f"| {f.fold + 1} | {train_0:%Y-%m-%d} → {train_1:%Y-%m-%d} ({len(train)}) "
            f"| {test_0:%Y-%m-%d} → {test_1:%Y-%m-%d} ({len(test)}) "
            f"| {f.params['fast']}/{f.params['slow']} "
            f"| {_num(f.train_metric)} | {_num(test_sharpe)} | {_num(test_tstat)} |"
        )
    return rows


def _overfit_note(result: WalkForwardResult) -> str:
    gaps: list[float] = []
    below = 0
    unmeasurable_folds: list[int] = []
    for f in result.folds:
        test_sharpe = sharpe_ratio(f.test_result.returns, result.timeframe)
        if math.isfinite(test_sharpe):
            gaps.append(f.train_metric - test_sharpe)
            if test_sharpe < f.train_metric:
                below += 1
        else:
            unmeasurable_folds.append(f.fold + 1)
    if not gaps:
        return "**Overfit gap (train − test Sharpe):** not computable (no finite test Sharpe)."
    mean_gap = sum(gaps) / len(gaps)
    # the honest denominator is the number of MEASURABLE folds: counting an
    # unmeasurable fold in the ratio would understate the regularity of the result
    if unmeasurable_folds:
        listed = ", ".join(str(k) for k in unmeasurable_folds)
        plural = "s" if len(unmeasurable_folds) > 1 else ""
        ratio = (
            f"test Sharpe below train Sharpe on {below} of the {len(gaps)} measurable folds "
            f"(fold{plural} {listed} not measurable: strategy flat over the whole test "
            "window, excluded from the mean)"
        )
    else:
        ratio = f"test Sharpe below train Sharpe on {below}/{len(result.folds)} folds"
    return (
        f"**Overfit gap (train − test Sharpe):** mean {mean_gap:+.2f} · {ratio} — "
        "the gap is the overfit measure, displayed on purpose."
    )


def _prereg_note(symbol: str, metrics: Mapping[str, float]) -> str:
    oos_sharpe, oos_tstat = metrics["sharpe"], metrics["t_stat"]
    in_range = PREREG_SHARPE_LO <= oos_sharpe <= PREREG_SHARPE_HI
    t_ok = abs(oos_tstat) < PREREG_TSTAT_MAX
    note = (
        f"**{symbol}:** OOS Sharpe {oos_sharpe:.2f} "
        f"{'inside' if in_range else '**OUTSIDE**'} the pre-registered "
        f"[{PREREG_SHARPE_LO:+.1f}, {PREREG_SHARPE_HI:+.1f}] · "
        f"|t| = {abs(oos_tstat):.2f} {'<' if t_ok else '≥'} {PREREG_TSTAT_MAX:.0f}"
    )
    if oos_sharpe > PREREG_ALARM_SHARPE:
        note += (
            " — **ALARM: an OOS Sharpe > 1.0 on this sample means suspect a bug or a "
            "leak; re-check the two blindness invariants before believing it**"
        )
    return note


def _ci95(m: Mapping[str, float]) -> tuple[float, float]:
    return (m["sharpe"] - 1.96 * m["sharpe_se"], m["sharpe"] + 1.96 * m["sharpe_se"])


def _oos_line(m: Mapping[str, float], n_bars: int) -> str:
    lo, hi = _ci95(m)
    return (
        f"**OOS, {n_bars} concatenated test bars:** "
        f"Sharpe {_num(m['sharpe'])} ± {_num(m['sharpe_se'])} (SE), "
        f"CI95 [{_num(lo)}, {_num(hi)}] · t-stat {_num(m['t_stat'])} · "
        f"total return {_pct(m['total_return'])} · "
        f"max drawdown {_pct(m['max_drawdown'])} · "
        f"ann. turnover {m['annualized_turnover']:.1f} · Sortino {_num(m['sortino'])} · "
        f"win rate {_pct_unsigned(m['win_rate'])} (non-flat bars only) · "
        f"flat bars {_pct_unsigned(m['flat_share'])}"
    )


def _power_note(oos_bars: int) -> str:
    """B3: both numbers are computed here, never hand-written."""
    years = oos_bars / calendar_bars_per_year(RESULTS_TIMEFRAME)
    minimum_detectable_sharpe = 2.0 / math.sqrt(years)
    return (
        f"**Statistical power (computed, not hand-written):** on an OOS span of "
        f"{years:.3f} years, the smallest Sharpe detectable at |t| > 2 is "
        f"2/√{years:.3f} = {minimum_detectable_sharpe:.2f}. No Sharpe below "
        f"{minimum_detectable_sharpe:.2f} measured on this window can be "
        "distinguished from luck. This backtest does not have the statistical power "
        "to demonstrate an edge; it has the power to demonstrate a method."
    )


def _selection_note(walkforward: Mapping[str, Mapping[str, WalkForwardResult]]) -> str:
    """Say explicitly whether the selection step actually discriminated anything."""
    picks = {
        (int(f.params["fast"]), int(f.params["slow"]))
        for by_config in walkforward.values()
        for result in by_config.values()
        for f in result.folds
    }
    if len(picks) == 1:
        fast, slow = next(iter(picks))
        return (
            f"**Selection is degenerate on this grid and this sample:** every fold of "
            f"every configuration (anchored and rolling, both symbols) selects "
            f"{fast}/{slow}, so this walk-forward is observationally equivalent to "
            f"evaluating MA {fast}/{slow} in OOS. The protocol is correct; its "
            "discriminating power is NOT demonstrated by this run (it is demonstrated "
            "on synthetic data by the moving-optimum fixture of "
            "`tests/test_walk_forward_evaluate.py`, where the argmax provably moves)."
        )
    listed = ", ".join(f"{fast}/{slow}" for fast, slow in sorted(picks))
    return (
        f"**Selection moved on this sample:** {len(picks)} distinct parameter sets "
        f"retained across folds and configurations: {listed}."
    )


def _splice_note(stats: Mapping[str, Any]) -> str:
    return (
        f"**Fold splicing (`n_carried_boundaries`, counted by the engine):** "
        f"{stats['carried_positions']} of "
        f"{stats['same_param_boundaries']} same-parameter fold boundaries carried a "
        "non-zero position into the next fold (identical parameters → one continuous "
        "backtest; a parameter change restarts flat). Forcing a flat restart at every "
        "fold — the pre-splice convention — would have produced a total return of "
        f"{_pct(stats['standalone_total_return'])} instead of "
        f"{_pct(stats['spliced_total_return'])}: "
        f"{stats['splice_delta_bps']:+.0f} bps of splice artifact (phantom re-entry "
        "fees + forced-flat first bars), not trades."
    )


def build_walkforward_block(computed: Computed) -> str:
    (n_samples,) = {len(p) for p in computed.prices.values()}
    n_test = int(round(n_samples * WF_TEST_SIZE))
    (oos_bars,) = {
        len(result.oos.returns)
        for by_config in computed.walkforward.values()
        for result in by_config.values()
    }

    lines = [
        WF_BEGIN_MARK,
        "_Generated by `python -m scripts.generate_results` — do not edit by hand"
        " (CI enforces sync)._",
        "",
        f"**Protocol:** {WF_N_FOLDS} walk-forward folds in TWO configurations — "
        "anchored (expanding train from bar 0; the pre-registered protocol) and "
        "rolling (fixed-length train sliding forward) · OOS region = last "
        f"{n_test} of {n_samples} bars (test_size {WF_TEST_SIZE}), tiled by contiguous "
        f"test windows · purge {WF_PURGE} bars (= slow_max, the grid's longest "
        f"look-back) · embargo {WF_EMBARGO} bars · selection: argmax of TRAIN "
        f"{WF_SELECTION_METRIC} over the declared grid ({len(WF_GRID)} combos: "
        "fast ∈ {5, 10, 20, 50}, slow ∈ {20, 50, 100, 200}, fast < slow) · "
        f"costs {CONFIG.fee_bps:.0f} bps fee + {CONFIG.slippage_bps:.0f} bps slippage "
        "per trade · execution: signal at close[t], fill at open[t+1] · bars before a "
        "window feed the MAs as look-back; parameter choice sees train only · fold "
        "boundary: identical selected parameters splice into one continuous backtest "
        "(the position carries over; no phantom flat bar, no phantom re-entry fee), a "
        "parameter change restarts flat · Sharpe error bar: Lo (2002) i.i.d.-normal "
        "SE, ≈ 1/√(OOS years) — frequency-independent; serial dependence widens it "
        "(conservative for a no-edge conclusion).",
        "",
        _power_note(oos_bars),
        "",
        _selection_note(computed.walkforward),
    ]
    prereg_notes: list[str] = []
    for symbol in SYMBOLS:
        prices = computed.prices[symbol]
        for config_name, _anchored, config_desc in WF_CONFIG_LABELS:
            result = computed.walkforward[symbol][config_name]
            m = result.metrics
            lines += [
                "",
                f"### {symbol} — {config_name} ({config_desc})",
                "",
                "| Fold | Train (bars) | Test (bars) | Params fast/slow | Sharpe train "
                "| Sharpe test | t-stat test |",
                "|---|---|---|---|---|---|---|",
                *_fold_rows(prices, result),
                "",
                _oos_line(m, len(result.oos.returns)),
                "",
                _overfit_note(result),
                "",
                _splice_note(computed.splice[symbol][config_name]),
            ]
            if config_name == "anchored":
                prereg_notes.append(_prereg_note(symbol, m))
    lines += [
        "",
        "**Pre-registration check (anchored runs — the pre-registered protocol; auto-generated):**",
        "",
    ]
    lines += [f"- {note}" for note in prereg_notes]
    lines.append(WF_END_MARK)
    return "\n".join(lines)


def _json_ready(obj: Any) -> Any:
    """Make ``obj`` serializable as STRICT JSON without losing information.

    Strict JSON has no NaN/Infinity tokens, so non-finite floats become the explicit
    strings "NaN" / "Infinity" / "-Infinity" (a deliberate, lossless convention --
    ``allow_nan=False`` below guarantees no bare token can slip through). Finite
    floats are left untouched: ``json.dumps`` serializes them via ``repr``, the
    shortest representation that round-trips bit-for-bit.
    """
    if isinstance(obj, dict):
        return {key: _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_ready(value) for value in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        if math.isnan(obj):
            return "NaN"
        return "Infinity" if obj > 0 else "-Infinity"
    return obj


def _fold_payload(prices: pd.DataFrame, result: WalkForwardResult) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for f in result.folds:
        train, test = f.split.train, f.split.test
        folds.append(
            {
                "fold": f.fold + 1,
                "train_start": pd.Timestamp(prices.index[train[0]]).isoformat(),
                "train_end": pd.Timestamp(prices.index[train[-1]]).isoformat(),
                "train_bars": len(train),
                "test_start": pd.Timestamp(prices.index[test[0]]).isoformat(),
                "test_end": pd.Timestamp(prices.index[test[-1]]).isoformat(),
                "test_bars": len(test),
                "params": dict(f.params),
                "train_sharpe": f.train_metric,
                "test_sharpe": sharpe_ratio(f.test_result.returns, result.timeframe),
                "test_t_stat": sharpe_tstat(f.test_result.returns, result.timeframe),
            }
        )
    return folds


def build_metrics_payload(computed: Computed) -> dict[str, Any]:
    """The golden numbers: full float precision, plus the config that produced them."""
    full_sample: dict[str, dict[str, dict[str, float]]] = {}
    for r in computed.baseline_rows:
        metrics = {k: v for k, v in r.items() if k not in ("symbol", "name")}
        full_sample.setdefault(str(r["symbol"]), {})[str(r["name"])] = metrics

    walk_forward: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        walk_forward[symbol] = {}
        for config_name, anchored, _desc in WF_CONFIG_LABELS:
            result = computed.walkforward[symbol][config_name]
            ci_lo, ci_hi = _ci95(result.metrics)
            walk_forward[symbol][config_name] = {
                "anchored": anchored,
                "folds": _fold_payload(computed.prices[symbol], result),
                "oos_bars": len(result.oos.returns),
                "oos": dict(result.metrics),
                "oos_sharpe_ci95": [ci_lo, ci_hi],
                "n_carried_boundaries": result.n_carried_boundaries,
                "splice": dict(computed.splice[symbol][config_name]),
            }

    datasets: dict[str, Any] = {}
    for symbol in SYMBOLS:
        datasets[symbol] = {
            timeframe: {
                "file": entry["file"],
                "rows": entry["rows"],
                "missing_bars": list(entry["missing_bars"]),
                "sha256": entry["sha256"],
            }
            for timeframe, entry in computed.manifest["datasets"][symbol].items()
        }

    return {
        "config": {
            "start": START,
            "end": dict(END),
            "results_timeframe": RESULTS_TIMEFRAME,
            "fee_bps": CONFIG.fee_bps,
            "slippage_bps": CONFIG.slippage_bps,
            "execution": "signal at close[t], fill at open[t+1]",
            "annualization_bars_per_year": bars_per_year(RESULTS_TIMEFRAME),
            "tstat_calendar_bars_per_year": calendar_bars_per_year(RESULTS_TIMEFRAME),
            "risk_free_rate": 0.0,
            "walk_forward": {
                "n_folds": WF_N_FOLDS,
                "test_size": WF_TEST_SIZE,
                "purge": WF_PURGE,
                "embargo": WF_EMBARGO,
                "configurations": [name for name, _anchored, _desc in WF_CONFIG_LABELS],
                "selection_metric": WF_SELECTION_METRIC,
                "grid": WF_GRID,
            },
        },
        "data": datasets,
        "full_sample": full_sample,
        "walk_forward": walk_forward,
    }


def render_metrics_json(computed: Computed) -> str:
    payload = _json_ready(build_metrics_payload(computed))
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def splice(readme: str, block: str, begin: str, end: str) -> str:
    begin_at = readme.index(begin)
    end_at = readme.index(end) + len(end)
    return readme[:begin_at] + block + readme[end_at:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if README or results/metrics.json is out of sync with the data",
    )
    args = parser.parse_args()

    # bytes in, bytes out: read_text()'s universal-newline translation would let a
    # CRLF-rewritten file pass a "byte-for-byte" comparison, so the comparison is
    # done on raw bytes with an explicit utf-8 codec (.gitattributes pins eol=lf)
    readme_bytes = README_PATH.read_bytes()
    readme = readme_bytes.decode("utf-8")
    for begin, end in ((BEGIN_MARK, END_MARK), (WF_BEGIN_MARK, WF_END_MARK)):
        if begin not in readme or end not in readme:
            print(f"ERROR: markers {begin} … {end} not found in {README_PATH}", file=sys.stderr)
            return 1

    computed = compute_all()
    updated = splice(readme, build_block(computed), BEGIN_MARK, END_MARK)
    updated = splice(updated, build_walkforward_block(computed), WF_BEGIN_MARK, WF_END_MARK)
    metrics_bytes = render_metrics_json(computed).encode("utf-8")

    if args.check:
        ok = True
        if updated.encode("utf-8") != readme_bytes:
            print(
                "ERROR: README results are out of sync with the committed data.\n"
                "Run `python -m scripts.generate_results` and commit the diff.",
                file=sys.stderr,
            )
            ok = False
        committed = METRICS_PATH.read_bytes() if METRICS_PATH.exists() else None
        if committed != metrics_bytes:
            print(
                "ERROR: results/metrics.json is out of sync (or missing) -- a number "
                "moved, byte-for-byte.\n"
                "Run `python -m scripts.generate_results` and commit the diff.",
                file=sys.stderr,
            )
            ok = False
        if not ok:
            return 1
        print("README and results/metrics.json are in sync.")
        return 0

    README_PATH.write_bytes(updated.encode("utf-8"))
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_bytes(metrics_bytes)
    print(f"README results sections regenerated in {README_PATH}")
    print(f"golden metrics written to {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
