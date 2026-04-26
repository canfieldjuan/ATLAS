"""Wiring tests for the EvidenceClaim audit task.

Pin the operational invariants the module-only Step 6 commit missed:

  - The handler is in the _BUILTIN_TASKS tuple, so HeadlessRunner can
    dispatch it.
  - register_builtin_tasks() actually registers it on a runner so a
    scheduled row with builtin_handler='b2b_evidence_claim_audit'
    resolves instead of erroring 'unknown handler'.
  - There is a default scheduler seed in
    TaskScheduler._DEFAULT_TASKS that paints the cron, the timeout,
    and the metadata defaults the audit task expects.
  - That seed lands disabled by default, since enabling it before
    ATLAS_B2B_CHURN_EVIDENCE_CLAIM_SHADOW_ENABLED=true would alert on
    zero rows every day.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_audit_task_in_builtin_registry():
    from atlas_brain.autonomous.tasks import _BUILTIN_TASKS

    assert (
        "b2b_evidence_claim_audit",
        "run",
        "b2b_evidence_claim_audit",
    ) in _BUILTIN_TASKS


def test_audit_task_registers_on_runner():
    """register_builtin_tasks() must surface the audit handler under the
    name the scheduled row will pass as builtin_handler."""

    class _Recorder:
        def __init__(self) -> None:
            self.registered: dict[str, object] = {}

        def register_builtin(self, name: str, handler) -> None:
            self.registered[name] = handler

    from atlas_brain.autonomous.tasks import register_builtin_tasks

    runner = _Recorder()
    register_builtin_tasks(runner)
    assert "b2b_evidence_claim_audit" in runner.registered
    handler = runner.registered["b2b_evidence_claim_audit"]
    # The registered handler is the run() coroutine function.
    assert callable(handler)
    assert getattr(handler, "__name__", "") == "run"
    assert handler.__module__.endswith("b2b_evidence_claim_audit")


def test_audit_task_seeded_in_default_tasks_disabled():
    """The seed must exist in _DEFAULT_TASKS, point at the right handler,
    have a cron expression, and be disabled by default. Enabling it
    before shadow capture is on would page on zero rows daily."""
    from atlas_brain.autonomous.scheduler import TaskScheduler

    seed = next(
        (
            t
            for t in TaskScheduler._DEFAULT_TASKS
            if t.get("name") == "b2b_evidence_claim_audit"
        ),
        None,
    )
    assert seed is not None, (
        "_DEFAULT_TASKS missing b2b_evidence_claim_audit seed"
    )
    assert seed["task_type"] == "builtin"
    assert seed["schedule_type"] == "cron"
    assert seed.get("cron_expression"), "seed must declare a cron_expression"
    assert seed.get("enabled") is False, (
        "audit task must be disabled by default; it alerts on zero rows "
        "and shadow capture is also off by default"
    )
    metadata = seed.get("metadata") or {}
    assert metadata.get("builtin_handler") == "b2b_evidence_claim_audit"


def test_audit_task_seed_runs_after_witness_quality_maintenance():
    """The audit reads back rows the synthesis cycle wrote. To avoid a
    race, it must schedule AFTER b2b_witness_quality_maintenance, which
    itself runs after the synthesis cycle. Enforce by comparing cron
    HH:MM tuples, not just text equality."""
    from atlas_brain.autonomous.scheduler import TaskScheduler

    def _hhmm(cron: str) -> tuple[int, int]:
        # cron format: "M H * * *" -> (H, M)
        parts = cron.split()
        return (int(parts[1]), int(parts[0]))

    audit = next(
        t for t in TaskScheduler._DEFAULT_TASKS
        if t.get("name") == "b2b_evidence_claim_audit"
    )
    witness = next(
        t for t in TaskScheduler._DEFAULT_TASKS
        if t.get("name") == "b2b_witness_quality_maintenance"
    )
    assert _hhmm(audit["cron_expression"]) > _hhmm(witness["cron_expression"]), (
        "audit must run after witness_quality_maintenance to avoid races"
    )
