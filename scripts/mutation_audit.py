"""Mutation audit of the walk-forward pipeline: 12 deliberate sabotages, one at a time.

Each mutation is applied to the source, the full pytest suite is rerun, and the dead
tests are counted -- TARGETED tests (everything except tests/test_metrics_golden.py)
separately from GOLDEN deaths, because the golden only says "some number changed",
not "this behavior is guarded". The pristine source is restored from an in-memory
backup after each mutation (never git checkout: it would clobber uncommitted work).

A mutation that kills zero targeted tests is a guardrail hole: unless it is listed
in ``EXPECTED_SURVIVORS`` with its reason, the script exits 1 and the missing test
must be written (the non-adjacent-splice test and the non-causal-factory truncation
test were both born exactly this way, from surviving M7 and M8). An expected
survivor that starts dying also fails the audit: its entry has gone stale.

Not CI-enforced: it reruns the full suite once per mutation (a few minutes). Rerun
after any change to engine/evaluate.py or engine/splits.py, and update the README's
mutation table from the output:

    python -m scripts.mutation_audit
"""

from __future__ import annotations

import hashlib
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import FrameType

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "engine" / "evaluate.py"
SPLITS = REPO / "engine" / "splits.py"

MUTATIONS: list[tuple[str, Path, str, str]] = [
    (
        "M1 argmax selects on TEST instead of TRAIN",
        EVAL,
        "train_result = _window_backtest(prices, strategy_factory, params, split.train, cfg)",
        "train_result = _window_backtest(prices, strategy_factory, params, split.test, cfg)",
    ),
    (
        "M2 fold measurement uses TRAIN instead of TEST",
        EVAL,
        "window = np.concatenate([np.asarray(splits[k].test) for k in segment])",
        "window = np.concatenate([np.asarray(splits[k].train) for k in segment])",
    ),
    (
        "M3 purge/embargo silently zeroed in walk_forward_splits",
        SPLITS,
        "    n_test_total = int(round(n_samples * test_size))",
        "    purge = 0\n    embargo = 0\n    n_test_total = int(round(n_samples * test_size))",
    ),
    (
        "M4 unlagged positions: fold windows see one bar of future",
        EVAL,
        "    signal = factory(history, params)\n",
        "    signal = factory(history, params).shift(-1).ffill()\n",
    ),
    (
        "M5 assert_no_leakage removed from the pipeline",
        EVAL,
        "        assert_no_leakage(split.train, split.test, min_gap=split.purge + split.embargo)\n",
        "        pass\n",
    ),
    (
        "M6 splice reset removed: position carried across a parameter change",
        EVAL,
        "        if same_params and contiguous:",
        "        if contiguous:",
    ),
    (
        "M7 splice contiguity removed: non-adjacent windows glued",
        EVAL,
        "        if same_params and contiguous:",
        "        if same_params:",
    ),
    (
        "M8 history truncation removed: the factory is shown the future",
        EVAL,
        "    history = prices.iloc[: int(window[-1]) + 1]",
        "    history = prices",
    ),
    (
        "M9 positional gapped-window rejection disabled",
        EVAL,
        "            if not bool((np.diff(np.asarray(window)) == 1).all()):",
        "            if False:",
    ),
    (
        "M10 chronological all_test check removed",
        EVAL,
        "    if not bool((np.diff(all_test) > 0).all()):",
        "    if False:",
    ),
    # M11 is deliberately absent: the external audit's M11 (select params on TEST
    # instead of TRAIN) is this table's M1, present since the commit-5 campaign.
    (
        "M12 union calendar re-validation removed at segment level",
        EVAL,
        "        _assert_window_calendar(prices, window, timeframe, known_holes, segment[0], "
        '"segment")',
        "        pass",
    ),
    (
        "M13 segment-level positional contiguity assertion disabled",
        EVAL,
        "        if not bool((np.diff(window) == 1).all()):",
        "        if False:",
    ),
]

# Mutations EXPECTED to survive, with the reason. The mechanism is deliberate:
# an UNEXPECTED survivor fails the audit (a guardrail hole to fix), and an expected
# survivor that starts DYING fails it too (its entry is stale and must be removed).
# A mutation is never deleted to fake a 100% table.
# M8 sat here when introduced ("every committed strategy is causal, so the history
# truncation is redundant for them") and was then evicted:
# test_history_truncation_hides_bars_beyond_the_window_even_from_a_non_causal_factory
# kills it with a deliberately non-causal global-mean factory.
EXPECTED_SURVIVORS: dict[str, str] = {
    "M13": (
        "unreachable by construction: the splice rule appends fold k to the current "
        "segment only when splits[k].test[0] == splits[k-1].test[-1] + 1, and every "
        "fold window is contiguity-checked per fold, so the concatenated segment "
        "window is contiguous by construction. The assertion guards the invariant, "
        "not an input. If a test ever starts killing it, the splice rule has changed "
        "and the invariant no longer holds -- investigate before removing this entry."
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_pristine_base(paths: Sequence[Path]) -> None:
    """Refuse to audit from an unknown base (B13).

    Every file this audit will mutate must be byte-identical to HEAD, checked
    BEFORE any mutation is applied and before a single kill count is printed: a
    count computed from a dirty base is a published number of unknown origin.
    The incident this guards against: a timeout-killed run left the engine
    mutated on disk, and the re-audit published five inflated counts from it.
    """
    rel = [str(path.relative_to(REPO)) for path in paths]
    proc = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *rel], cwd=REPO, check=False)
    if proc.returncode == 0:
        return
    names = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *rel],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = ", ".join(names.stdout.split()) or ", ".join(rel)
    print(
        f"REFUSED: dirty audit base -- target file(s) differ from HEAD: {dirty}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> int:
    # the exact set of files this audit will mutate, derived from MUTATIONS --
    # never a hard-coded list that could drift from the table
    targets = sorted({path for _name, path, _old, _new in MUTATIONS}, key=str)
    assert_pristine_base(targets)
    baseline = {path: _sha256(path) for path in targets}
    originals = {path: path.read_bytes() for path in targets}

    def restore_all() -> None:
        for path, data in originals.items():
            path.write_bytes(data)

    def on_signal(signum: int, _frame: FrameType | None) -> None:
        # an audit that dies must leave the tree exactly as it found it (B13)
        restore_all()
        raise SystemExit(128 + signum)

    previous_term = signal.signal(signal.SIGTERM, on_signal)
    previous_int = signal.signal(signal.SIGINT, on_signal)

    failures = 0
    restoration_ok = False
    try:
        for name, path, old, new in MUTATIONS:
            src = path.read_text(encoding="utf-8")
            if old not in src:
                print(
                    f"{name}: PATTERN NOT FOUND -- the audit is stale, fix it first",
                    file=sys.stderr,
                )
                return 1
            path.write_bytes(src.replace(old, new, 1).encode("utf-8"))
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-rfE", "--tb=no"],
                    cwd=REPO,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                lines = proc.stdout.splitlines()
                dead = [line for line in lines if line.startswith(("FAILED", "ERROR"))]
                golden = [line for line in dead if "test_metrics_golden" in line]
                targeted = [line for line in dead if "test_metrics_golden" not in line]
                mutation_id = name.split(" ", 1)[0]
                expected_survivor = mutation_id in EXPECTED_SURVIVORS
                print(f"=== {name}")
                print(f"    targeted kills: {len(targeted)}   golden kills: {len(golden)}")
                for line in targeted:
                    print(f"      {line.split(' ', 1)[-1]}")
                if not targeted and expected_survivor:
                    print(f"    ASSUMED survivor: {EXPECTED_SURVIVORS[mutation_id]}")
                elif not targeted:
                    failures += 1
                    print(
                        "    !!! UNEXPECTED SURVIVOR (0 targeted kills): write the missing test !!!"
                    )
                elif expected_survivor:
                    failures += 1
                    print(
                        "    !!! EXPECTED SURVIVOR NOW DIES: remove its stale entry from "
                        "EXPECTED_SURVIVORS !!!"
                    )
            finally:
                path.write_bytes(src.encode("utf-8"))
    finally:
        # unconditional: reached on success, on any exception, and on
        # SIGTERM/SIGINT (whose handlers restore then re-raise as SystemExit)
        restore_all()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        mismatched = [str(path) for path in targets if _sha256(path) != baseline[path]]
        if mismatched:
            print(f"sources restored: FAILED for {', '.join(mismatched)}", file=sys.stderr)
        else:
            restoration_ok = True
            print("sources restored: OK")
    if not restoration_ok:
        return 2
    if failures:
        print(f"{failures} audit failure(s) -- fix before trusting the table", file=sys.stderr)
        return 1
    print("mutation audit clean (kills as expected, assumed survivors documented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
