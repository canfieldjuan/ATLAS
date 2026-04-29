"""Phase 1 of the B2B scrape architecture refactor.

Single eligibility decision surface for the autonomous scrape intake task
and the manual scrape API. After this module ships, both paths must
call ``should_scrape_now(ctx)`` instead of inlining gate orchestration.

Design (per docs/progress/b2b_scrape_architecture_refactor_plan_2026-04-28.md):

  - ``Decision = Allow | Skip`` -- sealed union returned by every rule
    and by ``should_scrape_now()``. Pattern-match at the call site.
  - ``RuleChain[Ctx]`` -- generic ordered evaluator. First rule that
    returns Skip wins; if no rule returns Skip, the chain returns Allow.
    The same primitive will back ``should_run_maintenance(ctx)`` in
    Phase 3 with its own context type and rule list.
  - ``ScrapeContext`` -- inputs to ``should_scrape_now()``. Mixes static
    target data with IO handles (cfg, pool) for now; the doc's Phase 1
    Non-Goals explicitly say "leave surrounding Any types alone."
  - ``MaintenanceContext`` -- stub for Phase 3.

Phase 1 migration window
------------------------

The three pre-scrape gates currently live as inline evaluators in
``atlas_brain.autonomous.tasks.b2b_scrape_intake`` (``_evaluate_pre_scrape_skip``,
``_evaluate_pre_scrape_low_yield_skip``,
``_evaluate_pre_scrape_recent_zero_insert_skip``). The Rule classes here
wrap those evaluators via deferred local imports.

The deprecation plan (Phase 1 in the design doc) specifies that the
inline evaluator bodies relocate INTO this module as Rule bodies and the
originals are deleted. That lands as part of Turn N+3, after both
autonomous intake and manual API have been routed to call
``should_scrape_now()``. During this migration window the wrapping is
load-bearing scaffold, not a permanent shim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable


# ---------------------------------------------------------------------------
# Decision: Allow | Skip
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Allow:
    """The eligibility decision is to allow the scrape (or maintenance run)
    to proceed. Carries no data."""


@dataclass(frozen=True)
class Skip:
    """The eligibility decision is to skip the scrape (or maintenance run).

    Fields:
      - ``status``: persisted to ``b2b_scrape_log.status`` when the gate
        fires (e.g., ``"skipped_redundant"``). Stable across the refactor.
      - ``stop_reason``: persisted to ``b2b_scrape_log.stop_reason``
        (e.g., ``"pre_scrape_cross_source_coverage"``).
      - ``reason``: canonical gate name, used as the dedup key. In the
        current pre-scrape gates ``reason == stop_reason``.
      - ``detail``: gate-specific evidence carried for observability and
        for downstream telemetry (ratios, counts, parser_version, etc.).
        Treat as read-only by convention; ``frozen=True`` does not deep-
        freeze the dict.
    """

    status: str
    stop_reason: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


Decision = Allow | Skip


# ---------------------------------------------------------------------------
# Enums: GateName, DecisionSurface
# ---------------------------------------------------------------------------


class GateName(str, Enum):
    """Canonical gate identifiers.

    The string value is the gate's ``reason`` field, which already appears
    in production telemetry (``b2b_scrape_log.stop_reason``,
    ``record_dedup`` reasons). Keeping the enum value identical to the
    legacy reason string means external observability is stable across
    the refactor.

    Stringly-typed gate enrollment is forbidden per the design doc's
    Design Rule 6. ``GateName`` exists so source enrollment in
    ``SourceSpec`` (Phase 2) cannot silently typo a gate name.
    """

    # Pre-scrape gates -- Phase 1 migrated.
    CROSS_SOURCE_COVERAGE = "pre_scrape_cross_source_coverage"
    LOW_INCREMENTAL_YIELD = "pre_scrape_low_incremental_yield"
    RECENT_ZERO_INSERT_PAGE_CAP = "pre_scrape_recent_zero_insert_page_cap"

    # In-scrape and post-parse gates -- Phase 2/3 placeholders. They do
    # not yet have Rule implementations; they are reserved here so the
    # ``SourceSpec.gates_by_surface`` registry can reference them with
    # type safety.
    KNOWN_REVIEW_PAGE_STOP = "known_review_page_stop"
    SOURCE_QUALITY_GATE = "source_quality_gate"


class DecisionSurface(str, Enum):
    """Lifecycle phase a gate fires in.

    Used by ``SourceSpec.gates_by_surface`` (Phase 2) to declare which
    gates apply to which lifecycle phase per source. The two decision
    primitives (``should_scrape_now``, ``should_run_maintenance``) each
    read their own surface enrollment.
    """

    PRE_SCRAPE = "pre_scrape"
    IN_SCRAPE = "in_scrape"
    POST_PARSE = "post_parse"
    MAINTENANCE = "maintenance"


# Maps the canonical gate name to the persisted ``b2b_scrape_log.status``
# string the orchestrator writes when that gate fires. The mapping is
# load-bearing during Phase 1 because the inline evaluator's raw return
# dict does not include ``status`` -- the orchestrator hardcodes it per
# call site. After Phase 1 the Rule emits Skip with status set directly.
_GATE_SKIP_STATUS: dict[str, str] = {
    GateName.CROSS_SOURCE_COVERAGE.value: "skipped_redundant",
    GateName.LOW_INCREMENTAL_YIELD.value: "skipped_low_incremental_yield",
    GateName.RECENT_ZERO_INSERT_PAGE_CAP.value: "skipped_recent_zero_insert_page_cap",
}


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


@dataclass
class ScrapeContext:
    """Inputs to ``should_scrape_now()``.

    Mixes static target data with IO handles (``cfg``, ``pool``). The
    design doc accepts this for Phase 1 and flags it for revisit after
    the SourceSpec registry lands. Do not refactor surrounding ``Any``
    types in this slice.
    """

    target_id: Any
    source: str
    vendor_name: str
    parser_version: str | None
    scrape_mode: str
    target_metadata: dict[str, Any]
    cfg: Any
    pool: Any


@dataclass
class MaintenanceContext:
    """Inputs to ``should_run_maintenance()``. Phase 3 fills in the rule
    set; this dataclass is published now so SourceSpec (Phase 2) can
    declare maintenance gate enrollment against a concrete type.
    """

    target_id: Any
    source: str
    vendor_name: str
    parser_version: str | None
    target_metadata: dict[str, Any]
    cfg: Any
    pool: Any


# ---------------------------------------------------------------------------
# Rule + RuleChain primitives
# ---------------------------------------------------------------------------


Ctx = TypeVar("Ctx")


@runtime_checkable
class Rule(Protocol[Ctx]):
    """A rule evaluates a context and returns a Decision.

    Contract:
      - ``name``: canonical gate name (typically a ``GateName.value``).
        Used for telemetry and for the SourceSpec registry's gate
        enrollment lookup in Phase 2.
      - ``evaluate(ctx)``: deterministic pure-ish function of the
        context. May read from ``ctx.pool`` and ``ctx.cfg``. Must return
        Skip iff the conditions to suppress the action hold; Allow
        otherwise. Failures should propagate -- the chain does not catch
        exceptions on the rule's behalf.
    """

    name: str

    async def evaluate(self, ctx: Ctx) -> Decision: ...


class RuleChain(Generic[Ctx]):
    """Ordered list of rules. First Skip wins; if no rule returns Skip,
    the chain returns Allow.

    The chain is the shared decision primitive across surfaces (scrape
    eligibility, maintenance eligibility, etc.). Each surface defines
    its own context type, its own ordered rule list, and its own
    top-level ``should_*`` entry point that delegates to a chain
    instance.
    """

    def __init__(self, rules: list[Rule[Ctx]]) -> None:
        self._rules: list[Rule[Ctx]] = list(rules)

    @property
    def rules(self) -> list[Rule[Ctx]]:
        """Return a defensive copy of the rule list (introspection)."""
        return list(self._rules)

    async def evaluate(self, ctx: Ctx) -> Decision:
        for rule in self._rules:
            decision = await rule.evaluate(ctx)
            if isinstance(decision, Skip):
                return decision
        return Allow()


# ---------------------------------------------------------------------------
# Pre-scrape rules
# ---------------------------------------------------------------------------


def _decision_from_inline_raw(
    gate_name: str, raw: dict[str, Any] | None,
) -> Decision:
    """Adapt an inline evaluator's dict-or-None return into a ``Decision``.

    The inline evaluators in ``b2b_scrape_intake`` return ``None`` for
    "allow" or a dict with a ``reason`` key plus gate-specific detail
    fields for "skip". The unified Skip dataclass takes ``status`` from
    the canonical gate->status mapping (because the inline evaluator's
    raw dict does not include status -- the orchestrator hardcoded it
    per call site).

    Used during the Phase 1 migration window where Rule bodies wrap
    existing evaluators. After Turn N+3 the evaluator bodies relocate
    here and emit ``Skip`` directly.
    """
    if raw is None:
        return Allow()
    reason = str(raw.get("reason") or "")
    return Skip(
        status=_GATE_SKIP_STATUS[gate_name],
        stop_reason=reason,
        reason=reason,
        detail={k: v for k, v in raw.items() if k != "reason"},
    )


class _CrossSourceCoverageRule:
    """Skip when the last N runs all show high cross-source duplicate ratio.

    Phase 1: wraps ``b2b_scrape_intake._evaluate_pre_scrape_skip``.
    Phase 1 deprecation moves the body here in Turn N+3+.
    """

    name: str = GateName.CROSS_SOURCE_COVERAGE.value

    async def evaluate(self, ctx: ScrapeContext) -> Decision:
        # Local import: ``b2b_scrape_intake`` is large and after Turn N+3
        # will import from this module (to call ``should_scrape_now``).
        # Module-level import here would create a circular dependency.
        from ...autonomous.tasks.b2b_scrape_intake import (
            _evaluate_pre_scrape_skip,
        )

        raw = await _evaluate_pre_scrape_skip(
            ctx.pool,
            target_id=ctx.target_id,
            source=ctx.source,
            vendor_name=ctx.vendor_name,
            cfg=ctx.cfg,
        )
        return _decision_from_inline_raw(self.name, raw)


class _LowIncrementalYieldRule:
    """Skip when recent same-parser-version runs produced almost no
    NET-NEW reviews (inserts minus cross-source duplicates) relative
    to found.

    Phase 1: wraps ``b2b_scrape_intake._evaluate_pre_scrape_low_yield_skip``.
    """

    name: str = GateName.LOW_INCREMENTAL_YIELD.value

    async def evaluate(self, ctx: ScrapeContext) -> Decision:
        from ...autonomous.tasks.b2b_scrape_intake import (
            _evaluate_pre_scrape_low_yield_skip,
        )

        raw = await _evaluate_pre_scrape_low_yield_skip(
            ctx.pool,
            target_id=ctx.target_id,
            source=ctx.source,
            vendor_name=ctx.vendor_name,
            parser_version=ctx.parser_version,
            cfg=ctx.cfg,
        )
        return _decision_from_inline_raw(self.name, raw)


class _RecentZeroInsertPageCapRule:
    """Skip when N consecutive same-parser-version runs all hit page_cap
    with zero inserts and total pages exceed the minimum.

    Phase 1: wraps
    ``b2b_scrape_intake._evaluate_pre_scrape_recent_zero_insert_skip``.
    """

    name: str = GateName.RECENT_ZERO_INSERT_PAGE_CAP.value

    async def evaluate(self, ctx: ScrapeContext) -> Decision:
        from ...autonomous.tasks.b2b_scrape_intake import (
            _evaluate_pre_scrape_recent_zero_insert_skip,
        )

        raw = await _evaluate_pre_scrape_recent_zero_insert_skip(
            ctx.pool,
            target_id=ctx.target_id,
            source=ctx.source,
            vendor_name=ctx.vendor_name,
            parser_version=ctx.parser_version,
            cfg=ctx.cfg,
        )
        return _decision_from_inline_raw(self.name, raw)


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


# Order matches the inline orchestration in
# ``b2b_scrape_intake.py:2162-2342``: cross-source coverage runs first,
# low-yield second, recent-zero-insert page-cap third. Behavior is
# preserved when more than one gate would fire on the same target --
# whichever gate the inline path returned first, this chain returns
# first too.
_PRE_SCRAPE_RULES: list[Rule[ScrapeContext]] = [
    _CrossSourceCoverageRule(),
    _LowIncrementalYieldRule(),
    _RecentZeroInsertPageCapRule(),
]

_PRE_SCRAPE_CHAIN: RuleChain[ScrapeContext] = RuleChain(_PRE_SCRAPE_RULES)


# Maintenance chain is empty in Phase 1. Phase 3 fills it in with rules
# wrapping the planner's filter chain
# (``scripts/plan_parser_upgrade_rescrape_targets.py:_run()``).
_MAINTENANCE_RULES: list[Rule[MaintenanceContext]] = []
_MAINTENANCE_CHAIN: RuleChain[MaintenanceContext] = RuleChain(_MAINTENANCE_RULES)


async def should_scrape_now(ctx: ScrapeContext) -> Decision:
    """Single eligibility decision used by autonomous intake and manual
    API scrape paths.

    Returns ``Allow`` to proceed with the scrape, or ``Skip`` with a
    structured reason that carries the gate name, the persisted status
    string, the stop_reason, and gate-specific detail.

    Caller responsibilities on Skip:
      - persist the skip via the orchestrator's existing logging path
        (Phase 1) or via ``apply_skip_decision()`` (Turn N+3 helper).
      - update target cooldown so the next scheduler tick does not
        immediately re-attempt the same target.
    """
    return await _PRE_SCRAPE_CHAIN.evaluate(ctx)


async def should_run_maintenance(ctx: MaintenanceContext) -> Decision:
    """Phase 3 stub. Currently always Allow.

    Maintenance eligibility today lives in
    ``scripts/plan_parser_upgrade_rescrape_targets.py:_run()`` and runs
    its own filter chain. Phase 3 migrates that chain into rules
    registered against this primitive.

    During Phase 1 and 2, no caller invokes this function. The stub
    returning ``Allow`` is safe in that no-caller state. If a future
    caller starts invoking it before Phase 3 rules are registered, the
    Allow result is the conservative default (the legacy planner chain
    still runs whatever filters are wired into the script).
    """
    return await _MAINTENANCE_CHAIN.evaluate(ctx)


__all__ = [
    "Allow",
    "Decision",
    "DecisionSurface",
    "GateName",
    "MaintenanceContext",
    "Rule",
    "RuleChain",
    "ScrapeContext",
    "Skip",
    "should_run_maintenance",
    "should_scrape_now",
]
