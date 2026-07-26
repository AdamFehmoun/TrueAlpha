"""Mutation audit of the walk-forward pipeline: 7 deliberate sabotages, one at a time.

Each mutation is applied to the source, the full pytest suite is rerun, and the dead
tests are counted -- TARGETED tests (everything except tests/test_metrics_golden.py)
separately from GOLDEN deaths, because the golden only says "some number changed",
not "this behavior is guarded". The pristine source is restored from an in-memory
backup after each mutation (never git checkout: it would clobber uncommitted work).

A mutation that kills zero targeted tests is a guardrail hole: the script exits 1
and the missing test must be written (M7 was born exactly this way).

Not CI-enforced: it reruns the full suite once per mutation (a few minutes). Rerun
after any change to engine/evaluate.py or engine/splits.py, and update the README's
mutation table from the output:

    python -m scripts.mutation_audit
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
]


def main() -> int:
    survivors = 0
    for name, path, old, new in MUTATIONS:
        src = path.read_text(encoding="utf-8")
        if old not in src:
            print(f"{name}: PATTERN NOT FOUND -- the audit is stale, fix it first", file=sys.stderr)
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
            print(f"=== {name}")
            print(f"    targeted kills: {len(targeted)}   golden kills: {len(golden)}")
            for line in targeted:
                print(f"      {line.split(' ', 1)[-1]}")
            if not targeted:
                survivors += 1
                print("    !!! SURVIVING MUTATION (0 targeted kills): write the missing test !!!")
        finally:
            path.write_bytes(src.encode("utf-8"))
    if survivors:
        print(f"{survivors} surviving mutation(s) -- a guardrail hole", file=sys.stderr)
        return 1
    print("all mutations killed by at least one targeted test; sources restored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
