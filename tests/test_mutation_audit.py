"""The mutation audit's own guardrails (B13): refuse a dirty base, restore on death.

Born from a real, logged incident: the audit was launched with a 120 s timeout
on a ~3 minute job; the SIGTERM killed it mid-flight LEAVING engine/evaluate.py
MUTATED on disk, the next pytest produced two phantom failures, and the
re-audit published five inflated kill counts from the dirty base before
catching itself. Two defects, both in the instrument, none in the engine:
(1) it started without verifying its base was clean; (2) it did not guarantee
restoration when dying. These tests make both impossible to regress silently.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

import scripts.mutation_audit as mutation_audit


def _audit_targets() -> list[Path]:
    return sorted({path for _name, path, _old, _new in mutation_audit.MUTATIONS}, key=str)


def test_dirty_base_is_refused_before_any_kill_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dirty target file must abort the audit non-zero BEFORE any mutation is
    applied and before a single kill count is printed: never publish a number
    computed from an unknown base. The fake git reports 'dirty'; the fake suite
    invocation fails the test outright if it is ever reached."""

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "pytest" in cmd:
            pytest.fail("the suite must never be invoked on a dirty base")
        if "--quiet" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if "--name-only" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="engine/evaluate.py\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    # mutation_audit does `import subprocess`, so patching the module's `run`
    # globally is exactly what its calls will see
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as excinfo:
        mutation_audit.main()
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "targeted kills" not in captured.out  # ZERO kill counts printed
    assert "REFUSED" in captured.err
    assert "engine/evaluate.py" in captured.err  # the dirty file is NAMED


def test_sources_restored_even_when_the_suite_invocation_raises(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE incident test: an audit that dies mid-loop -- here the suite
    invocation raises, the same effect as the SIGTERM that killed it for real,
    and it fires while the first mutation IS applied on disk -- must leave every
    target file byte-identical to its pre-audit state, print the restoration
    proof, and let the exception propagate instead of swallowing it."""
    targets = _audit_targets()
    before = {path: path.read_bytes() for path in targets}

    # R4: run the audit over ONE synthetic mutation whose anchor no real mutation
    # consumes. With the real MUTATIONS table, running under audit mutation M1
    # finds M1's anchor already consumed -> PATTERN NOT FOUND -> return 1 -> the
    # RuntimeError never fires and this test dies by SELF-REFERENCE, not by
    # detection (and only ever exercises the table's first entry).
    anchor = "from __future__ import annotations"
    assert mutation_audit.EVAL.read_text(encoding="utf-8").count(anchor) == 1, (
        "the synthetic anchor must appear EXACTLY once in engine/evaluate.py; "
        "anything else silently turns this test into the no-op it exists to prevent"
    )
    synthetic = [("R4 synthetic mutation", mutation_audit.EVAL, anchor, anchor + "  # mutated")]
    monkeypatch.setattr(mutation_audit, "MUTATIONS", synthetic)

    monkeypatch.setattr(mutation_audit, "assert_pristine_base", lambda paths: None)

    def exploding_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise RuntimeError("suite invocation killed mid-audit")

    monkeypatch.setattr(subprocess, "run", exploding_run)
    with pytest.raises(RuntimeError, match="killed mid-audit"):
        mutation_audit.main()

    after = {path: path.read_bytes() for path in targets}
    assert after == before  # byte-identical, i.e. sha256-identical
    assert "sources restored: OK" in capsys.readouterr().out
