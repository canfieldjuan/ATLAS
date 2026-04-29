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

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable


logger = logging.getLogger("atlas.services.scraping.eligibility")


# ---------------------------------------------------------------------------
# Source-allowlist resolvers
#
# Relocated from b2b_scrape_intake.py in Turn N+3b. These helpers were
# previously used only by the inline _evaluate_pre_scrape_* functions; now
# they are private to this module and consumed exclusively by the Rule
# implementations below.
# ---------------------------------------------------------------------------


# Cached static default derived from capability metadata. The override knob
# (cfg.pre_scrape_skip_paid_sources_override) is read per-call, not cached.
_PRE_SCRAPE_SKIP_PAID_SOURCES_DEFAULT: frozenset[str] | None = None


def _derive_pre_scrape_skip_paid_sources_default() -> frozenset[str]:
    """Paid-source set derived from services.scraping.capabilities.

    Includes a source iff data_quality == verified AND
    (web_unlocker in access_patterns OR proxy_class == residential).
    Computed once and memoized -- capability profiles are immutable code.
    """
    global _PRE_SCRAPE_SKIP_PAID_SOURCES_DEFAULT
    if _PRE_SCRAPE_SKIP_PAID_SOURCES_DEFAULT is not None:
        return _PRE_SCRAPE_SKIP_PAID_SOURCES_DEFAULT
    from .capabilities import (
        AccessPattern,
        DataQuality,
        ProxyClass,
        get_all_capabilities,
    )
    paid: set[str] = set()
    for source, profile in get_all_capabilities().items():
        if profile.data_quality is not DataQuality.verified:
            continue
        uses_web_unlocker = AccessPattern.web_unlocker in profile.access_patterns
        uses_residential = profile.proxy_class is ProxyClass.residential
        if uses_web_unlocker or uses_residential:
            paid.add(str(source).strip().lower())
    _PRE_SCRAPE_SKIP_PAID_SOURCES_DEFAULT = frozenset(paid)
    return _PRE_SCRAPE_SKIP_PAID_SOURCES_DEFAULT


def _resolve_pre_scrape_skip_paid_sources(cfg: Any) -> frozenset[str]:
    """Resolve the paid-source allowlist for cross-source + low-yield gates.

    Per-call read of cfg.pre_scrape_skip_paid_sources_override; empty/blank
    falls back to the capability-derived default.
    """
    raw_override = str(getattr(cfg, "pre_scrape_skip_paid_sources_override", "") or "").strip()
    if not raw_override:
        return _derive_pre_scrape_skip_paid_sources_default()
    items = {part.strip().lower() for part in raw_override.split(",") if part.strip()}
    if not items:
        return _derive_pre_scrape_skip_paid_sources_default()
    return frozenset(items)


def _resolve_pre_scrape_recent_zero_insert_sources(cfg: Any) -> frozenset[str]:
    """Resolve the source allowlist for the recent zero-insert gate."""
    from .sources import parse_source_allowlist
    raw_sources = str(
        getattr(cfg, "pre_scrape_recent_zero_insert_skip_sources", "") or ""
    ).strip()
    if not raw_sources:
        return frozenset()
    return frozenset(parse_source_allowlist(raw_sources))


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
      - ``dedup_stage``: ``stage`` argument passed to
        ``record_dedup`` when this rule fires. Locating the dedup-stage
        identifier on the rule keeps gate-specific telemetry coupled
        to the rule, not scattered across orchestrator dispatch tables.
      - ``evaluate(ctx)``: deterministic pure-ish function of the
        context. May read from ``ctx.pool`` and ``ctx.cfg``. Must return
        Skip iff the conditions to suppress the action hold; Allow
        otherwise. Failures should propagate -- the chain does not catch
        exceptions on the rule's behalf.
      - ``format_skip_log(ctx, decision)``: returns the operator-visible
        log message emitted by ``apply_skip_decision`` when the rule
        produced this Skip. Operator-format string lives here, with the
        rule, instead of in a separate dispatch table.
    """

    name: str
    dedup_stage: str

    async def evaluate(self, ctx: Ctx) -> Decision: ...

    def format_skip_log(self, ctx: Ctx, decision: Skip) -> str: ...


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


class _CrossSourceCoverageRule:
    """Skip when the last N runs all show high cross-source duplicate ratio.

    Six conditions must ALL hold to skip (the original
    _evaluate_pre_scrape_skip contract preserved verbatim):
      1. cfg.pre_scrape_skip_enabled
      2. source is in the paid-source allowlist (override or default)
      3. real_runs in lookback window >= configured lookback
      4. total reviews_found across that window >= min_found
      5. total cross_source_duplicates across that window >= min_dups
      6. duplicate_ratio (dupes/found) >= configured threshold
      7. last real scrape within max_age_days (escape hatch)

    Reads only b2b_scrape_log -- a single index-only scan over <=
    lookback_runs rows.
    """

    name: str = GateName.CROSS_SOURCE_COVERAGE.value
    dedup_stage: str = "b2b_scrape_pre_skip"

    async def evaluate(self, ctx: ScrapeContext) -> Decision:
        cfg = ctx.cfg
        if not getattr(cfg, "pre_scrape_skip_enabled", False):
            return Allow()
        source_key = str(ctx.source or "").strip().lower()
        if not source_key:
            return Allow()
        paid_sources = _resolve_pre_scrape_skip_paid_sources(cfg)
        if source_key not in paid_sources:
            return Allow()
        lookback = max(2, int(getattr(cfg, "pre_scrape_skip_lookback_runs", 5) or 5))
        row = await ctx.pool.fetchrow(
            """
            WITH recent AS (
                SELECT status, reviews_found, cross_source_duplicates, started_at
                FROM b2b_scrape_log
                WHERE target_id = $1
                  AND COALESCE(status, '') NOT LIKE 'skipped%'
                ORDER BY started_at DESC
                LIMIT $2
            )
            SELECT
                COUNT(*)                                  AS real_runs,
                COALESCE(SUM(reviews_found), 0)           AS total_found,
                COALESCE(SUM(cross_source_duplicates), 0) AS total_dupes,
                MAX(started_at)                           AS last_real_scrape_at
            FROM recent
            """,
            ctx.target_id,
            lookback,
        )
        if row is None:
            return Allow()
        real_runs = int(row["real_runs"] or 0)
        total_found = int(row["total_found"] or 0)
        total_dupes = int(row["total_dupes"] or 0)
        last_real_scrape_at = row["last_real_scrape_at"]
        if real_runs < lookback:
            return Allow()
        min_found = int(getattr(cfg, "pre_scrape_skip_min_reviews_found_total", 20) or 20)
        if total_found < min_found:
            return Allow()
        min_dups = int(getattr(cfg, "pre_scrape_skip_min_dup_rows", 10) or 10)
        if total_dupes < min_dups:
            return Allow()
        if total_found <= 0:
            return Allow()
        ratio = total_dupes / total_found
        threshold = float(getattr(cfg, "pre_scrape_skip_dup_ratio", 0.90) or 0.90)
        if ratio < threshold:
            return Allow()
        max_age_days = int(getattr(cfg, "pre_scrape_skip_max_age_days", 14) or 14)
        if last_real_scrape_at is not None:
            age = datetime.now(timezone.utc) - last_real_scrape_at
            if age.days > max_age_days:
                return Allow()
        return Skip(
            status=_GATE_SKIP_STATUS[self.name],
            stop_reason=self.name,
            reason=self.name,
            detail={
                "source": source_key,
                "vendor_name": ctx.vendor_name,
                "real_runs": real_runs,
                "total_found": total_found,
                "total_dupes": total_dupes,
                "duplicate_ratio": round(ratio, 4),
                "ratio_threshold": threshold,
                "lookback_runs": lookback,
                "min_reviews_found_total": min_found,
                "min_dup_rows": min_dups,
                "max_age_days": max_age_days,
                "last_real_scrape_at": (
                    last_real_scrape_at.isoformat()
                    if last_real_scrape_at is not None else None
                ),
            },
        )

    def format_skip_log(self, ctx: ScrapeContext, decision: Skip) -> str:
        d = decision.detail
        return (
            "Pre-scrape skip for %s/%s: ratio=%.2f over %d runs (saved 1 paid call)"
            % (
                ctx.source,
                ctx.vendor_name,
                float(d.get("duplicate_ratio") or 0.0),
                int(d.get("real_runs") or 0),
            )
        )


class _LowIncrementalYieldRule:
    """Skip when recent same-parser-version runs produced almost no
    NET-NEW reviews (inserts minus cross-source duplicates) relative to
    found.

    Net-new = reviews_inserted minus cross_source_duplicates (cross-source
    dup rows are inserted with duplicate_of_review_id but contribute zero
    actionable churn signal -- they should not flatter the yield ratio).

    The lookback is filtered to rows that match the current parser_version
    so historical broken-parser logs cannot suppress a newly fixed parser.
    """

    name: str = GateName.LOW_INCREMENTAL_YIELD.value
    dedup_stage: str = "b2b_scrape_pre_skip_low_yield"

    async def evaluate(self, ctx: ScrapeContext) -> Decision:
        cfg = ctx.cfg
        if not getattr(cfg, "pre_scrape_low_yield_skip_enabled", False):
            return Allow()
        source_key = str(ctx.source or "").strip().lower()
        if not source_key:
            return Allow()
        paid_sources = _resolve_pre_scrape_skip_paid_sources(cfg)
        if source_key not in paid_sources:
            return Allow()
        current_parser_version = str(ctx.parser_version or "").strip()
        if not current_parser_version:
            # No baseline to compare against -- never suppress.
            return Allow()
        lookback = max(2, int(getattr(cfg, "pre_scrape_low_yield_skip_lookback_runs", 5) or 5))
        row = await ctx.pool.fetchrow(
            """
            WITH recent AS (
                SELECT status, reviews_found, reviews_inserted,
                       COALESCE(cross_source_duplicates, 0) AS cross_source_duplicates,
                       started_at
                FROM b2b_scrape_log
                WHERE target_id = $1
                  AND COALESCE(status, '') NOT LIKE 'skipped%'
                  AND parser_version = $3
                ORDER BY started_at DESC
                LIMIT $2
            )
            SELECT
                COUNT(*)                                              AS real_runs,
                COALESCE(SUM(reviews_found), 0)                       AS total_found,
                COALESCE(SUM(reviews_inserted), 0)                    AS total_inserted,
                COALESCE(SUM(cross_source_duplicates), 0)             AS total_cross_source_duplicates,
                COALESCE(SUM(GREATEST(reviews_inserted - COALESCE(cross_source_duplicates, 0), 0)), 0)
                                                                      AS total_unique_inserted,
                MAX(started_at)                                       AS last_real_scrape_at
            FROM recent
            """,
            ctx.target_id,
            lookback,
            current_parser_version,
        )
        if row is None:
            return Allow()
        real_runs = int(row["real_runs"] or 0)
        total_found = int(row["total_found"] or 0)
        total_inserted = int(row["total_inserted"] or 0)
        total_cross_source_duplicates = int(row["total_cross_source_duplicates"] or 0)
        total_unique_inserted = int(row["total_unique_inserted"] or 0)
        last_real_scrape_at = row["last_real_scrape_at"]
        if real_runs < lookback:
            return Allow()
        min_found = int(getattr(cfg, "pre_scrape_low_yield_skip_min_found_total", 50) or 50)
        if total_found < min_found:
            return Allow()
        if total_found <= 0:
            # Defense in depth: zero-found alone must never trigger this gate.
            return Allow()
        ratio = total_unique_inserted / total_found
        raw_threshold = getattr(cfg, "pre_scrape_low_yield_skip_ratio_threshold", 0.05)
        threshold = 0.05 if raw_threshold is None else float(raw_threshold)
        if ratio >= threshold:
            return Allow()
        max_age_days = int(getattr(cfg, "pre_scrape_low_yield_skip_max_age_days", 14) or 14)
        if last_real_scrape_at is not None:
            age = datetime.now(timezone.utc) - last_real_scrape_at
            if age.total_seconds() > max_age_days * 24 * 60 * 60:
                return Allow()
        return Skip(
            status=_GATE_SKIP_STATUS[self.name],
            stop_reason=self.name,
            reason=self.name,
            detail={
                "source": source_key,
                "vendor_name": ctx.vendor_name,
                "parser_version": current_parser_version,
                "real_runs": real_runs,
                "total_found": total_found,
                "total_inserted": total_inserted,
                "total_cross_source_duplicates": total_cross_source_duplicates,
                "total_unique_inserted": total_unique_inserted,
                "unique_insert_ratio": round(ratio, 4),
                "ratio_threshold": threshold,
                "lookback_runs": lookback,
                "min_found_total": min_found,
                "max_age_days": max_age_days,
                "last_real_scrape_at": (
                    last_real_scrape_at.isoformat()
                    if last_real_scrape_at is not None else None
                ),
            },
        )

    def format_skip_log(self, ctx: ScrapeContext, decision: Skip) -> str:
        d = decision.detail
        return (
            "Pre-scrape low-yield skip for %s/%s: ratio=%.3f over %d runs (saved 1 paid call)"
            % (
                ctx.source,
                ctx.vendor_name,
                float(d.get("unique_insert_ratio") or 0.0),
                int(d.get("real_runs") or 0),
            )
        )


class _RecentZeroInsertPageCapRule:
    """Skip when N consecutive same-parser-version runs all hit page_cap
    with zero inserts and total pages exceed the minimum. Catches
    repetitive search/API sweeps that keep walking the same already-known
    result window. Source allowlist is independent of the cross-source /
    low-yield paid-source set."""

    name: str = GateName.RECENT_ZERO_INSERT_PAGE_CAP.value
    dedup_stage: str = "b2b_scrape_pre_skip_recent_zero_insert_page_cap"

    async def evaluate(self, ctx: ScrapeContext) -> Decision:
        cfg = ctx.cfg
        if not getattr(cfg, "pre_scrape_recent_zero_insert_skip_enabled", False):
            return Allow()
        source_key = str(ctx.source or "").strip().lower()
        if not source_key:
            return Allow()
        allowed_sources = _resolve_pre_scrape_recent_zero_insert_sources(cfg)
        if source_key not in allowed_sources:
            return Allow()
        current_parser_version = str(ctx.parser_version or "").strip()
        if not current_parser_version:
            return Allow()
        consecutive_runs = max(
            2,
            int(getattr(cfg, "pre_scrape_recent_zero_insert_skip_consecutive_runs", 3) or 3),
        )
        row = await ctx.pool.fetchrow(
            """
            WITH recent AS (
                SELECT
                    status,
                    reviews_found,
                    reviews_inserted,
                    pages_scraped,
                    stop_reason,
                    parser_version,
                    started_at
                FROM b2b_scrape_log
                WHERE target_id = $1
                  AND COALESCE(status, '') NOT LIKE 'skipped%'
                ORDER BY started_at DESC, id DESC
                LIMIT $2
            )
            SELECT
                COUNT(*) AS real_runs,
                COALESCE(SUM(reviews_found), 0) AS total_found,
                COALESCE(SUM(reviews_inserted), 0) AS total_inserted,
                COALESCE(SUM(pages_scraped), 0) AS total_pages_scraped,
                BOOL_AND(COALESCE(status, '') IN ('success', 'partial')) AS all_successish,
                BOOL_AND(COALESCE(reviews_inserted, 0) = 0) AS all_zero_insert,
                BOOL_AND(COALESCE(stop_reason, '') = 'page_cap') AS all_page_cap,
                BOOL_AND(COALESCE(parser_version, '') = $3) AS all_current_parser,
                MAX(started_at) AS last_real_scrape_at
            FROM recent
            """,
            ctx.target_id,
            consecutive_runs,
            current_parser_version,
        )
        if row is None:
            return Allow()

        real_runs = int(row["real_runs"] or 0)
        if real_runs < consecutive_runs:
            return Allow()
        if not bool(row["all_successish"]):
            return Allow()
        if not bool(row["all_zero_insert"]):
            return Allow()
        if not bool(row["all_page_cap"]):
            return Allow()
        if not bool(row["all_current_parser"]):
            return Allow()

        total_pages_scraped = int(row["total_pages_scraped"] or 0)
        min_total_pages = max(
            1,
            int(getattr(cfg, "pre_scrape_recent_zero_insert_skip_min_total_pages", 6) or 6),
        )
        if total_pages_scraped < min_total_pages:
            return Allow()

        last_real_scrape_at = row["last_real_scrape_at"]
        max_age_days = int(
            getattr(cfg, "pre_scrape_recent_zero_insert_skip_max_age_days", 14) or 14
        )
        if last_real_scrape_at is not None:
            age = datetime.now(timezone.utc) - last_real_scrape_at
            if age.total_seconds() > max_age_days * 24 * 60 * 60:
                return Allow()

        return Skip(
            status=_GATE_SKIP_STATUS[self.name],
            stop_reason=self.name,
            reason=self.name,
            detail={
                "source": source_key,
                "vendor_name": ctx.vendor_name,
                "parser_version": current_parser_version,
                "real_runs": real_runs,
                "total_found": int(row["total_found"] or 0),
                "total_inserted": int(row["total_inserted"] or 0),
                "total_pages_scraped": total_pages_scraped,
                "consecutive_runs": consecutive_runs,
                "min_total_pages": min_total_pages,
                "max_age_days": max_age_days,
                "last_real_scrape_at": (
                    last_real_scrape_at.isoformat()
                    if last_real_scrape_at is not None else None
                ),
            },
        )

    def format_skip_log(self, ctx: ScrapeContext, decision: Skip) -> str:
        d = decision.detail
        return (
            "Pre-scrape repeated zero-insert skip for %s/%s: %d runs, %d pages (saved 1 scrape)"
            % (
                ctx.source,
                ctx.vendor_name,
                int(d.get("real_runs") or 0),
                int(d.get("total_pages_scraped") or 0),
            )
        )


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


# Lookup from gate name (== Skip.reason) to the originating Rule. Used by
# ``apply_skip_decision`` to dispatch operator-visible logging and dedup
# stage to the rule that produced the Skip. Built once at import time
# from the canonical pre-scrape rule list.
_PRE_SCRAPE_RULES_BY_NAME: dict[str, Rule[ScrapeContext]] = {
    rule.name: rule for rule in _PRE_SCRAPE_RULES
}


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


async def apply_skip_decision(
    pool: Any,
    *,
    ctx: ScrapeContext,
    decision: Skip,
) -> None:
    """Persist a Skip decision: scrape_log row, target cooldown, dedup
    record, and operator-visible log line.

    Encapsulates the persistence side that previously lived inline in
    ``b2b_scrape_intake``'s gate orchestration block and in
    ``api/b2b_scrape.py:_apply_manual_pre_scrape_gates``. After this
    helper lands, both call sites share one persistence path.

    Helpers (``_log_pre_scrape_skip``, ``_update_target_cooldown_only``,
    ``record_dedup``) live in ``b2b_scrape_intake`` and ``visibility``
    today and are imported via deferred local imports to avoid the
    circular dependency that arises when ``b2b_scrape_intake`` imports
    from this module.
    """
    from ...autonomous.tasks.b2b_scrape_intake import (
        _log_pre_scrape_skip,
        _update_target_cooldown_only,
    )
    from ...autonomous.visibility import record_dedup

    raw_decision = {"reason": decision.reason, **decision.detail}

    await _log_pre_scrape_skip(
        pool,
        target_id=ctx.target_id,
        source=ctx.source,
        parser_version=ctx.parser_version,
        decision=raw_decision,
        status=decision.status,
        stop_reason=decision.stop_reason,
    )
    await _update_target_cooldown_only(pool, ctx.target_id)

    rule = _PRE_SCRAPE_RULES_BY_NAME.get(decision.reason)
    stage = rule.dedup_stage if rule is not None else "b2b_scrape_pre_skip"
    try:
        await record_dedup(
            pool,
            stage=stage,
            entity_type="scrape_target",
            entity_id=str(ctx.target_id),
            reason=decision.reason,
            detail=raw_decision,
        )
    except Exception:
        logger.debug(
            "record_dedup failed for skip on %s/%s",
            ctx.source,
            ctx.vendor_name,
            exc_info=True,
        )

    if rule is not None:
        try:
            logger.info("%s", rule.format_skip_log(ctx, decision))
            return
        except Exception:
            logger.debug(
                "format_skip_log failed for %s; using generic format",
                decision.reason,
                exc_info=True,
            )
    logger.info(
        "Pre-scrape skip for %s/%s: gate=%s status=%s",
        ctx.source,
        ctx.vendor_name,
        decision.reason,
        decision.status,
    )


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
    "apply_skip_decision",
    "should_run_maintenance",
    "should_scrape_now",
]
