"""Phase 1 verification harness for the inline pre-scrape gate evaluators.

This is the snapshot/parity layer for the scrape-eligibility refactor laid
out in
docs/progress/b2b_scrape_architecture_refactor_plan_2026-04-28.md.

What this harness asserts today:

  - Each fixture under tests/fixtures/scrape_eligibility/<gate>/*.json
    drives one of the three current inline evaluators in
    atlas_brain.autonomous.tasks.b2b_scrape_intake:
      - _evaluate_pre_scrape_skip                        (cross-source)
      - _evaluate_pre_scrape_low_yield_skip              (low yield)
      - _evaluate_pre_scrape_recent_zero_insert_skip     (recent zero-insert
                                                          page-cap)

  - The evaluator's dict-or-None return is normalized into the unified
    {kind, status, stop_reason, reason, detail} payload that the future
    should_scrape_now() will return after Phase 1 lands. The fixtures
    therefore encode the post-refactor shape, not the pre-refactor dict.

  - Fixtures stay reusable unchanged. After Phase 1, a parallel test
    layer parametrizes over should_scrape_now() and asserts the same
    fixtures produce identical decisions. Decision parity proves the
    refactor is behavior-preserving.

The mocked pool only needs to return aggregated rows -- no per-row scrape
log, no DB connection. Each gate runs a single SQL aggregation against
b2b_scrape_log; the fixture encodes the column-shaped output of that
aggregation directly. If the SQL changes during Phase 1, fixtures with the
same logical inputs still produce the same decision because the fixture
captures aggregated state, not raw rows.

Fixture format (see fixtures/scrape_eligibility/*/*.json):

  {
    "name": str,
    "description": str,
    "gate": "pre_scrape_cross_source_coverage"
          | "pre_scrape_low_incremental_yield"
          | "pre_scrape_recent_zero_insert_page_cap",
    "inputs": {target_id, source, vendor_name, parser_version?},
    "cfg":   {<gate-specific config flags>},
    "pool_state": {
        "fetchrow_returns": {
            <gate-specific aggregate columns>,
            "last_real_scrape_at_offset_days": int | null
        }
    },
    "expected_decision":
        {"kind": "allow"}
        | {"kind": "skip",
           "status": str, "stop_reason": str, "reason": str,
           "detail_subset": {<key: value pairs that must appear in detail>}}
  }

last_real_scrape_at_offset_days is resolved at test runtime to a real
datetime relative to datetime.now(timezone.utc) so the gate's age-window
arithmetic produces deterministic results without the fixture carrying a
hard-coded timestamp that would drift.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# b2b_scrape_intake imports asyncpg at module load time. Mock it before
# the production import so the test harness does not need a real DB
# driver. Mirrors the pattern used by tests/test_b2b_vendor_briefing*.py.
_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

from atlas_brain.autonomous.tasks.b2b_scrape_intake import (  # noqa: E402
    _evaluate_pre_scrape_low_yield_skip,
    _evaluate_pre_scrape_recent_zero_insert_skip,
    _evaluate_pre_scrape_skip,
)
from atlas_brain.services.scraping.eligibility import (  # noqa: E402
    Allow,
    ScrapeContext,
    Skip,
    apply_skip_decision,
    should_scrape_now,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "scrape_eligibility"


# Maps the canonical gate name (as used in skip "reason" strings and as
# the future GateName enum value) to the inline evaluator + which kwargs
# that evaluator accepts. Cross-source dedup does not take parser_version;
# the other two do.
_INLINE_EVALUATORS: dict[str, tuple[Any, frozenset[str]]] = {
    "pre_scrape_cross_source_coverage": (
        _evaluate_pre_scrape_skip,
        frozenset({"target_id", "source", "vendor_name"}),
    ),
    "pre_scrape_low_incremental_yield": (
        _evaluate_pre_scrape_low_yield_skip,
        frozenset({"target_id", "source", "vendor_name", "parser_version"}),
    ),
    "pre_scrape_recent_zero_insert_page_cap": (
        _evaluate_pre_scrape_recent_zero_insert_skip,
        frozenset({"target_id", "source", "vendor_name", "parser_version"}),
    ),
}


# Maps gate name to the unified Skip "status" string. Status values are
# the orchestrator's skip-status (the value persisted to b2b_scrape_log
# when the gate fires); they are stable across the refactor.
_GATE_SKIP_STATUS: dict[str, str] = {
    "pre_scrape_cross_source_coverage": "skipped_redundant",
    "pre_scrape_low_incremental_yield": "skipped_low_incremental_yield",
    "pre_scrape_recent_zero_insert_page_cap": "skipped_recent_zero_insert_page_cap",
}


def _load_fixtures() -> list[tuple[str, dict[str, Any]]]:
    fixtures: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(FIXTURE_ROOT.glob("**/*.json")):
        if path.name.startswith("_"):
            continue
        data = json.loads(path.read_text())
        rel = path.relative_to(FIXTURE_ROOT).as_posix()
        fixtures.append((rel, data))
    return fixtures


def _resolve_pool_row(fetchrow_returns: dict[str, Any]) -> dict[str, Any]:
    """Translate fixture-relative fields into concrete pool-row values.

    last_real_scrape_at_offset_days becomes a datetime relative to
    datetime.now(timezone.utc); other keys pass through unchanged.
    """
    resolved: dict[str, Any] = {}
    has_explicit_offset = False
    for key, value in fetchrow_returns.items():
        if key == "last_real_scrape_at_offset_days":
            has_explicit_offset = True
            if value is None:
                resolved["last_real_scrape_at"] = None
            else:
                resolved["last_real_scrape_at"] = (
                    datetime.now(timezone.utc) + timedelta(days=int(value))
                )
        else:
            resolved[key] = value
    if not has_explicit_offset:
        resolved.setdefault("last_real_scrape_at", None)
    return resolved


def _make_pool(fixture: dict[str, Any]) -> MagicMock:
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value=_resolve_pool_row(fixture["pool_state"]["fetchrow_returns"])
    )
    return pool


def _normalize_inline_decision(
    gate: str, raw: dict[str, Any] | None
) -> dict[str, Any]:
    """Adapt the inline evaluator's dict-or-None into the unified Decision
    shape that should_scrape_now() returns."""
    if raw is None:
        return {"kind": "allow"}
    status = _GATE_SKIP_STATUS[gate]
    reason = str(raw.get("reason") or "")
    detail = {k: v for k, v in raw.items() if k != "reason"}
    return {
        "kind": "skip",
        "status": status,
        "stop_reason": reason,
        "reason": reason,
        "detail": detail,
    }


def _normalize_dataclass_decision(decision: Any) -> dict[str, Any]:
    """Normalize a Decision dataclass (Allow|Skip from
    atlas_brain.services.scraping.eligibility) into the same dict shape
    the inline path produces. Lets layer 1 and layer 3 share assertion
    logic and lets fixtures stay path-agnostic."""
    if isinstance(decision, Allow):
        return {"kind": "allow"}
    if isinstance(decision, Skip):
        return {
            "kind": "skip",
            "status": decision.status,
            "stop_reason": decision.stop_reason,
            "reason": decision.reason,
            "detail": dict(decision.detail),
        }
    raise TypeError(f"Unknown Decision subtype: {type(decision)!r}")


def _assert_decision_matches_fixture(
    fixture_name: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Shared assertion logic across decision-parity layers. The
    actual/expected payloads must already be normalized to the unified
    {kind, status?, stop_reason?, reason?, detail?} dict shape."""
    if expected["kind"] == "allow":
        assert actual == {"kind": "allow"}, (
            f"{fixture_name}: expected allow, got {actual!r}"
        )
        return

    assert actual["kind"] == "skip", (
        f"{fixture_name}: expected skip, got {actual!r}"
    )
    assert actual["status"] == expected["status"], (
        f"{fixture_name}: status mismatch."
        f" expected {expected['status']!r}, got {actual['status']!r}"
    )
    assert actual["stop_reason"] == expected["stop_reason"], (
        f"{fixture_name}: stop_reason mismatch."
        f" expected {expected['stop_reason']!r}, got {actual['stop_reason']!r}"
    )
    assert actual["reason"] == expected["reason"], (
        f"{fixture_name}: reason mismatch."
        f" expected {expected['reason']!r}, got {actual['reason']!r}"
    )

    detail_subset = expected.get("detail_subset", {})
    for key, expected_value in detail_subset.items():
        actual_value = actual["detail"].get(key)
        assert actual_value == expected_value, (
            f"{fixture_name}: detail[{key!r}]={actual_value!r},"
            f" expected {expected_value!r}"
        )


_FIXTURES = _load_fixtures()
_FIXTURE_IDS = [name for name, _ in _FIXTURES]


# ---------------------------------------------------------------------------
# Test layer 1: direct gate behavior + final normalized decision shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name,fixture",
    _FIXTURES,
    ids=_FIXTURE_IDS,
)
async def test_inline_evaluator_decision_matches_fixture(
    fixture_name: str,
    fixture: dict[str, Any],
) -> None:
    """Layer 1: the inline gate evaluator (current production path)
    produces the expected normalized decision payload.

    Same fixtures are exercised against ``should_scrape_now()`` in
    layer 3; decision parity across the two layers is the Phase 1
    behavior-preservation gate.
    """
    gate = fixture["gate"]
    if gate not in _INLINE_EVALUATORS:
        pytest.fail(f"{fixture_name}: unknown gate name {gate!r}")
    evaluator, accepted_kwargs = _INLINE_EVALUATORS[gate]

    pool = _make_pool(fixture)
    cfg = SimpleNamespace(**fixture["cfg"])
    inputs = fixture["inputs"]

    call_kwargs = {k: v for k, v in inputs.items() if k in accepted_kwargs}
    call_kwargs["cfg"] = cfg

    raw = await evaluator(pool, **call_kwargs)
    actual = _normalize_inline_decision(gate, raw)
    _assert_decision_matches_fixture(fixture_name, actual, fixture["expected_decision"])


# ---------------------------------------------------------------------------
# Test layer 2: every fixture's normalized decision is well-formed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name,fixture",
    _FIXTURES,
    ids=_FIXTURE_IDS,
)
def test_fixture_expected_decision_is_well_formed(
    fixture_name: str,
    fixture: dict[str, Any],
) -> None:
    """The expected_decision payload itself follows the unified shape.

    This is a fixture-shape regression test: it catches
    fixtures that drift away from the canonical Decision contract before
    the inline-evaluator parity test ever runs them. Cheap to run, fails
    fast with a clear message.
    """
    expected = fixture.get("expected_decision")
    assert isinstance(expected, dict), f"{fixture_name}: expected_decision must be a dict"

    kind = expected.get("kind")
    assert kind in {"allow", "skip"}, (
        f"{fixture_name}: expected_decision.kind must be 'allow' or 'skip', got {kind!r}"
    )

    if kind == "allow":
        assert set(expected.keys()) == {"kind"}, (
            f"{fixture_name}: allow expected_decision must contain only 'kind'"
        )
        return

    required_skip_keys = {"kind", "status", "stop_reason", "reason"}
    missing = required_skip_keys - set(expected.keys())
    assert not missing, (
        f"{fixture_name}: skip expected_decision missing keys {sorted(missing)!r}"
    )
    for key in ("status", "stop_reason", "reason"):
        assert isinstance(expected[key], str) and expected[key], (
            f"{fixture_name}: skip expected_decision[{key!r}] must be a non-empty string"
        )

    detail_subset = expected.get("detail_subset", {})
    assert isinstance(detail_subset, dict), (
        f"{fixture_name}: detail_subset must be a dict if present"
    )

    gate = fixture.get("gate")
    assert gate in _GATE_SKIP_STATUS, (
        f"{fixture_name}: gate {gate!r} not in registry"
    )
    assert expected["status"] == _GATE_SKIP_STATUS[gate], (
        f"{fixture_name}: status {expected['status']!r}"
        f" does not match gate {gate!r}'s canonical status"
        f" {_GATE_SKIP_STATUS[gate]!r}"
    )


# ---------------------------------------------------------------------------
# Test layer 3: should_scrape_now() decision parity vs the inline evaluators
#
# Phase 1 acceptance gate. The same fixtures must produce the same decision
# whether routed through the inline evaluator path (layer 1) or through the
# new shared chain (this layer). If layer 1 and layer 3 diverge on any
# fixture, the eligibility refactor is not behavior-preserving and Phase 1
# orchestration replacement (Turn N+3) cannot proceed.
#
# After Turn N+3 deletes the inline evaluator functions, layer 1 will be
# removed from this file. Layer 3 then becomes the canonical decision-shape
# test against ``should_scrape_now()``. The fixtures stay unchanged across
# both transitions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name,fixture",
    _FIXTURES,
    ids=_FIXTURE_IDS,
)
async def test_should_scrape_now_matches_fixture(
    fixture_name: str,
    fixture: dict[str, Any],
) -> None:
    """Layer 3: ``should_scrape_now()`` produces the same decision as the
    inline evaluator for every fixture.

    The chain runs all three pre-scrape rules in order. Fixtures isolate
    gate behavior via cfg flags -- each fixture only sets the
    ``_enabled`` flag for the gate it tests, so the chain naturally
    short-circuits at the other rules' enabled-checks before consulting
    the pool. No fixture changes were required to support this layer.
    """
    pool = _make_pool(fixture)
    cfg = SimpleNamespace(**fixture["cfg"])
    inputs = fixture["inputs"]

    ctx = ScrapeContext(
        target_id=inputs["target_id"],
        source=inputs["source"],
        vendor_name=inputs["vendor_name"],
        parser_version=inputs.get("parser_version"),
        scrape_mode="default",
        target_metadata={},
        cfg=cfg,
        pool=pool,
    )

    decision = await should_scrape_now(ctx)
    actual = _normalize_dataclass_decision(decision)
    _assert_decision_matches_fixture(fixture_name, actual, fixture["expected_decision"])


# ---------------------------------------------------------------------------
# apply_skip_decision integration -- verifies the persistence side calls
# _log_pre_scrape_skip / _update_target_cooldown_only / record_dedup
# with arguments correctly derived from the Skip dataclass + ScrapeContext.
# Single test, focused on wiring rather than per-gate combinatorics.
# ---------------------------------------------------------------------------


async def test_apply_skip_decision_invokes_persistence_helpers(monkeypatch):
    """apply_skip_decision adapts a Skip dataclass back to the dict shape
    that ``_log_pre_scrape_skip`` expects, calls
    ``_update_target_cooldown_only`` for the target, and emits a dedup
    record stamped with the originating rule's ``dedup_stage``.

    Patches the source modules' attributes so the deferred-local-imports
    inside apply_skip_decision pick up the mocks at call time.
    """
    log_skip_mock = AsyncMock(return_value=None)
    update_cooldown_mock = AsyncMock(return_value=None)
    record_dedup_mock = AsyncMock(return_value=None)

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake._log_pre_scrape_skip",
        log_skip_mock,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake._update_target_cooldown_only",
        update_cooldown_mock,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.visibility.record_dedup",
        record_dedup_mock,
    )

    pool = MagicMock()
    cfg = SimpleNamespace()
    ctx = ScrapeContext(
        target_id="00000000-0000-0000-0000-000000000010",
        source="g2",
        vendor_name="Shopify",
        parser_version="g2.v3.2",
        scrape_mode="default",
        target_metadata={},
        cfg=cfg,
        pool=pool,
    )
    decision = Skip(
        status="skipped_redundant",
        stop_reason="pre_scrape_cross_source_coverage",
        reason="pre_scrape_cross_source_coverage",
        detail={"duplicate_ratio": 0.92, "real_runs": 5, "total_found": 50},
    )

    await apply_skip_decision(pool, ctx=ctx, decision=decision)

    # _log_pre_scrape_skip got the raw decision dict reconstructed from
    # the Skip dataclass + the canonical status/stop_reason kwargs.
    log_skip_mock.assert_awaited_once()
    log_kwargs = log_skip_mock.await_args.kwargs
    assert log_kwargs["target_id"] == ctx.target_id
    assert log_kwargs["source"] == ctx.source
    assert log_kwargs["parser_version"] == ctx.parser_version
    assert log_kwargs["status"] == decision.status
    assert log_kwargs["stop_reason"] == decision.stop_reason
    raw_decision = log_kwargs["decision"]
    assert raw_decision["reason"] == decision.reason
    assert raw_decision["duplicate_ratio"] == 0.92
    assert raw_decision["real_runs"] == 5
    assert raw_decision["total_found"] == 50

    # Cooldown helper called with the target id only.
    update_cooldown_mock.assert_awaited_once_with(pool, ctx.target_id)

    # record_dedup gets the originating rule's dedup_stage; the mapping
    # comes from _PRE_SCRAPE_RULES_BY_NAME[reason].dedup_stage.
    record_dedup_mock.assert_awaited_once()
    dedup_kwargs = record_dedup_mock.await_args.kwargs
    assert dedup_kwargs["stage"] == "b2b_scrape_pre_skip"
    assert dedup_kwargs["entity_type"] == "scrape_target"
    assert dedup_kwargs["entity_id"] == str(ctx.target_id)
    assert dedup_kwargs["reason"] == decision.reason
    assert dedup_kwargs["detail"]["duplicate_ratio"] == 0.92


async def test_apply_skip_decision_swallows_record_dedup_failure(monkeypatch):
    """A record_dedup failure must NOT prevent the scrape_log row or
    cooldown update. Telemetry failure is non-fatal; behavior preserved
    from the inline orchestration's try/except."""
    log_skip_mock = AsyncMock(return_value=None)
    update_cooldown_mock = AsyncMock(return_value=None)
    record_dedup_mock = AsyncMock(side_effect=RuntimeError("dedup table missing"))

    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake._log_pre_scrape_skip",
        log_skip_mock,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.tasks.b2b_scrape_intake._update_target_cooldown_only",
        update_cooldown_mock,
    )
    monkeypatch.setattr(
        "atlas_brain.autonomous.visibility.record_dedup",
        record_dedup_mock,
    )

    pool = MagicMock()
    ctx = ScrapeContext(
        target_id="00000000-0000-0000-0000-000000000011",
        source="stackoverflow",
        vendor_name="AWS",
        parser_version="stackoverflow.v1.0",
        scrape_mode="default",
        target_metadata={},
        cfg=SimpleNamespace(),
        pool=pool,
    )
    decision = Skip(
        status="skipped_recent_zero_insert_page_cap",
        stop_reason="pre_scrape_recent_zero_insert_page_cap",
        reason="pre_scrape_recent_zero_insert_page_cap",
        detail={"real_runs": 3, "total_pages_scraped": 9},
    )

    # Must not raise -- the dedup failure is logged at debug and swallowed.
    await apply_skip_decision(pool, ctx=ctx, decision=decision)

    log_skip_mock.assert_awaited_once()
    update_cooldown_mock.assert_awaited_once()
    record_dedup_mock.assert_awaited_once()  # fired even though it raised
