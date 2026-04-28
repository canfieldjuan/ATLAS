# B2B Scrape Architecture Refactor Plan

**Created**: 2026-04-28
**Status**: Proposed
**Owner**: Codex + Juan

---

## Goal

Raise the scrape system to this product bar:

- Vendor add requires zero developer touches.
- Source add requires a parser file, one source registration, and tests.
- Autonomous intake and manual API runs use one eligibility decision path.
- Source behavior enrollment is defined in one contract, not scattered through code and config.

## Non-Goals

These are explicitly out of scope for the first refactor slice:

- No new reactive scrape-cost gates.
- No parser-family rewrite in the first slice.
- No broad type cleanup or forced strict typing.
- No changing scrape semantics unless needed to unify duplicated decision paths.

## Current Problem

The current system is paying real cost to maintain duplicated source logic across:

- autonomous intake
- manual `/run`
- manual `/run-all`
- parser-upgrade maintenance
- source-yield pruning
- parser-level page-stop behavior

That creates three failures:

1. Path divergence: one path gets a gate, another path bypasses it.
2. Source onboarding checklist: adding a source requires remembering multiple disconnected memberships.
3. Silent failure modes: scrape still "works" but loses the cost-saving behavior because one source enrollment point was missed.

## Verified Current State

### Eligibility logic is duplicated

Autonomous intake applies pre-scrape gates inline in:
- `atlas_brain/autonomous/tasks/b2b_scrape_intake.py:2162-2342`

The gate functions live in the same file:
- `_evaluate_pre_scrape_skip` at `1666-1751`
- `_evaluate_pre_scrape_low_yield_skip` at `1754-1859`
- `_evaluate_pre_scrape_recent_zero_insert_skip` at `1862-1968`
- `_log_pre_scrape_skip` at `1971+`

Manual API now has its own orchestration helper in:
- `atlas_brain/api/b2b_scrape.py:169-304`

and uses it in:
- single-target run: `1427-1504`
- bulk run: `1782-1855`

This is better than before, but it is still duplicated orchestration rather than a shared decision contract.

### Source behavior enrollment is fragmented

Hard-coded source memberships currently exist in:

- known-review page-stop set:
  - `atlas_brain/autonomous/tasks/b2b_scrape_intake.py:76-88`
- default source-quality-gate set:
  - `atlas_brain/autonomous/tasks/b2b_scrape_intake.py:89-91`
- source classifications and default allowlists:
  - `atlas_brain/services/scraping/sources.py:79-165`

Config-driven source memberships also exist in:

- `deprecated_sources`:
  - `atlas_brain/config.py:3964-3967`
- `source_low_yield_pruning_source`:
  - `atlas_brain/config.py:4004-4007`
- `parser_upgrade_deferred_sources`:
  - `atlas_brain/config.py:4054-4060`
- `parser_upgrade_maintenance_sources`:
  - `atlas_brain/config.py:4145-4151`
- `parser_upgrade_maintenance_deep_sources`:
  - `atlas_brain/config.py:4198-4204`
- `pre_scrape_recent_zero_insert_skip_sources`:
  - `atlas_brain/config.py:4413-4416`

### Parser registry is separate from source behavior

Parser registration currently lives in:
- `atlas_brain/services/scraping/parsers/__init__.py:344-366`

with auto-import registration at:
- `atlas_brain/services/scraping/parsers/__init__.py:369-370`

This registry knows how to find a parser, but it does not know:
- which gates apply to the source
- whether known-review page stop is enabled
- whether the source participates in maintenance
- whether the source uses quality gating

### Maintenance uses another source-control surface

Parser-upgrade maintenance is driven by:
- `atlas_brain/autonomous/tasks/b2b_parser_upgrade_maintenance.py:33-66`
- `scripts/plan_parser_upgrade_rescrape_targets.py`

It relies on parser lookup and config-backed source lists instead of a shared source contract.

## Target Architecture

The refactor should introduce three layers.

### Layer 1: Decision Primitive

Create a shared rule-evaluation primitive that every decision surface uses.

```python
@dataclass
class Allow:
    pass


@dataclass
class Skip:
    status: str
    stop_reason: str
    reason: str
    detail: dict[str, Any]


Decision = Allow | Skip
```

Use one shared evaluator primitive:

```python
class RuleChain[Ctx]:
    def __init__(self, rules: list[Rule[Ctx]]) -> None: ...
    async def evaluate(self, ctx: Ctx) -> Decision: ...
```

This is the shared primitive, not a single shared rule set.

### Layer 2: Surface Contexts

Create surface-specific context objects.

```python
@dataclass
class ScrapeContext:
    target_id: Any
    source: str
    vendor_name: str
    parser_version: str | None
    scrape_mode: str
    target_metadata: dict[str, Any]
    cfg: Any
    pool: Any
```

Notes:
- Leave surrounding `Any` types alone when wiring this into existing code.
- This context is provisional and should not force a broad typing cleanup.
- It currently mixes static target data with IO handles. That is acceptable for
  Phase 1 and should be revisited after the registry lands.

Maintenance gets its own context:

```python
@dataclass
class MaintenanceContext:
    target_id: Any
    source: str
    vendor_name: str
    parser_version: str | None
    target_metadata: dict[str, Any]
    cfg: Any
    pool: Any
```

### Layer 3: SourceSpec

Create one typed source behavior contract.

```python
class GateName(str, Enum):
    CROSS_SOURCE_COVERAGE = "pre_scrape_cross_source_coverage"
    LOW_INCREMENTAL_YIELD = "pre_scrape_low_incremental_yield"
    RECENT_ZERO_INSERT_PAGE_CAP = "pre_scrape_recent_zero_insert_page_cap"
    KNOWN_REVIEW_PAGE_STOP = "known_review_page_stop"
    SOURCE_QUALITY_GATE = "source_quality_gate"


class DecisionSurface(str, Enum):
    PRE_SCRAPE = "pre_scrape"
    IN_SCRAPE = "in_scrape"
    POST_PARSE = "post_parse"
    MAINTENANCE = "maintenance"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    parser_key: str
    gates_by_surface: Mapping[DecisionSurface, frozenset[GateName]]
```

Registration must fail fast if a source references an unknown gate.

The final schema may grow, but the registry should start with typed behavior
enrollment, not a broad collection of booleans and free-form strings.

## Design Rules

1. Thresholds stay in config.
2. Source participation moves to `SourceSpec`.
3. Manual and autonomous paths call the same eligibility function.
4. Parser registry compatibility is preserved during migration.
5. Do not force all sources into one parser abstraction at once.
6. Stringly-typed gate enrollment is not allowed.

## Proposed Phases

### Phase 1: Shared Eligibility Decision

Create a new module:
- `atlas_brain/services/scraping/eligibility.py`

Responsibilities:
- define `Decision`
- define `RuleChain[Ctx]`
- define `ScrapeContext`
- define `MaintenanceContext` stub
- define `should_scrape_now(context)`
- define `should_run_maintenance(context)` stub
- define `apply_skip_decision(...)`
- register existing gates as rules instead of inlining them in task/api code

Initial migrated gates:
- `pre_scrape_cross_source_coverage`
- `pre_scrape_low_incremental_yield`
- `pre_scrape_recent_zero_insert_page_cap`

Current insertion points to replace:
- `atlas_brain/autonomous/tasks/b2b_scrape_intake.py:2162-2342`
- `atlas_brain/api/b2b_scrape.py:169-304`

Acceptance criteria:
- autonomous intake calls `should_scrape_now()`
- manual `/run` calls `should_scrape_now()`
- manual `/run-all` calls `should_scrape_now()`
- only one skip logging / cooldown update path remains
- behavior preserved for current skip cases
- verification mechanism is defined before code:
  - snapshot-style tests over `(target, recent_log_history, cfg) -> decision`
  - plus one replay sample from recent production-like scrape rows for each gate family

### Phase 2: SourceSpec Registry

Create a new module:
- `atlas_brain/services/scraping/source_specs.py`

Keep `ReviewSource` in `sources.py`, but add a registry that defines source behavior.

First migration targets:
- `_KNOWN_REVIEWS_PAGE_STOP_SOURCES`
- `_DEFAULT_SOURCE_QUALITY_GATE_SOURCES`
- `pre_scrape_recent_zero_insert_skip_sources`
- maintenance source enrollments
- any other source-participation memberships that are currently split across
  config or frozensets

Do not migrate threshold values into the registry.
Do not add `family` or `source_tier` to the first `SourceSpec` schema.
If those are still needed later, add them as typed enums in Phase 3 or Phase 4.

Acceptance criteria:
- adding a source no longer requires editing multiple scattered memberships
- source behavior enrollment can be read from one file
- parser lookup remains backward-compatible through `get_parser()` / `get_all_parsers()`

### Phase 3: Maintenance And Pruning Alignment

Update these components to consult `SourceSpec`:
- `atlas_brain/autonomous/tasks/b2b_parser_upgrade_maintenance.py`
- `scripts/plan_parser_upgrade_rescrape_targets.py`
- `atlas_brain/services/scraping/source_yield.py`

Keep policy thresholds in config.
Remove source-participation decisions from free-floating CSVs where the registry should own them.

Maintenance should not reuse the scrape rule set, but it should reuse the same
`RuleChain[Ctx]` primitive and typed gate enrollment model:

- `should_scrape_now: RuleChain[ScrapeContext]`
- `should_run_maintenance: RuleChain[MaintenanceContext]`

Acceptance criteria:
- maintenance source enrollment is not hidden in independent config fields
- source-yield pruning tiers read from a shared source contract or a shared derived mapping

### Phase 4: Parser Family Abstraction

Do not start here.

After the registry exists, introduce a shared scrape engine only for the structured paginated review family:
- `g2`
- `capterra`
- `gartner`
- `getapp`
- `peerspot`
- `software_advice`
- `trustpilot`
- `trustradius`

Likely family split:
- structured review platforms
- community/API/search sources
- news/feed sources

Do not try to force Reddit, StackOverflow, GitHub, Hacker News, or RSS into the same base loop in the first parser abstraction pass.

Acceptance criteria:
- shared page accounting
- shared stop-reason propagation
- shared known-review page-stop support
- source-specific extraction remains source-local

## Migration Strategy

### Slice 1
- design doc review
- implement `eligibility.py`
- replace autonomous and manual gate orchestration
- no registry yet

### Slice 2
- implement `source_specs.py`
- move source memberships there
- keep compatibility shims so existing parser lookups continue working

### Slice 3
- align maintenance and pruning with source registry

### Slice 4
- migrate structured review parsers to a shared engine in small batches

## Deprecation Plan

The refactor must remove old paths as each phase lands. Soft deprecation with no
removal trigger is how dual systems become permanent. Each phase below names
exactly what becomes legacy, whether it is shimmed or removed, and the exit
criteria that authorize deletion.

The four-step rule applies in every phase:

1. route callers to the new path
2. keep compatibility shims briefly only where needed
3. add explicit comments naming the transitional surface and its removal trigger
4. remove the old path once parity is proven

### Phase 1 deprecations: inline gate orchestration

**Becomes legacy:**
- inline gate orchestration block in `b2b_scrape_intake.py:2162-2342`
- `_apply_manual_pre_scrape_gates(...)` parameterized dispatch helper in
  `b2b_scrape.py:169-307` (itself a transitional artifact added during the
  manual-path bypass fix)
- direct callers of `_evaluate_pre_scrape_skip`,
  `_evaluate_pre_scrape_low_yield_skip`,
  `_evaluate_pre_scrape_recent_zero_insert_skip` outside the new
  `eligibility.py` module

**Disposition:**
- gate evaluator functions: relocate into `eligibility.py` as registered
  `Rule` implementations. Original signatures are NOT shimmed back into the
  old module -- `eligibility.should_scrape_now()` is the only callable surface.
- inline orchestration in autonomous intake: removed outright. No shim -- the
  block was internal, not an exported API.
- `_apply_manual_pre_scrape_gates` helper: removed outright once
  `b2b_scrape.py:trigger_scrape` and `trigger_scrape_all` call
  `should_scrape_now()` directly.

**Exit criteria for deletion:**
- snapshot tests pass for all three gate families against the fixture set
- one production-shape replay run shows zero decision deltas vs old path
- both autonomous and manual paths route through `should_scrape_now()`
- grep confirms no caller imports `_evaluate_pre_scrape_*` from outside
  `eligibility.py`
- removal PR cites: snapshot test pass, replay verification log, grep result.

### Phase 2 deprecations: scattered source memberships

**Becomes legacy:**
- frozensets:
  - `_KNOWN_REVIEWS_PAGE_STOP_SOURCES`
  - `_DEFAULT_SOURCE_QUALITY_GATE_SOURCES`
- config CSV fields:
  - `pre_scrape_recent_zero_insert_skip_sources`
  - `parser_upgrade_maintenance_sources`
  - `parser_upgrade_maintenance_deep_sources`
  - `parser_upgrade_deferred_sources`
  - `source_low_yield_pruning_source`

**Disposition:**
- frozensets: removed outright. The `SourceSpec` registry replaces them.
- config CSV fields: SHIMMED for one Phase 2 release as operator overrides
  only. If a CSV is non-empty, its value overrides the corresponding
  `SourceSpec` default for that source's gate enrollment. This preserves the
  operator escape hatch for a runaway source without requiring a code deploy.
- `deprecated_sources` (currently a config CSV at `config.py:3964-3967`) is a
  governance kill-switch surface, not a gate enrollment. Its disposition is
  decided by the Phase 2 review: it likely moves into a
  `SourceSpec.governance` field rather than being deprecated outright.

**Exit criteria for deletion:**
- frozensets: deleted as soon as grep confirms zero references after Phase 2
  lands.
- CSV fields: deleted in the slice immediately AFTER Phase 2 ships, gated on
  override-frequency telemetry showing no operator usage during the shim
  window. The deletion PR explicitly cites that telemetry.

### Phase 3 deprecations: maintenance filter chain

**Becomes legacy:**
- inline maintenance filter chain in `plan_parser_upgrade_rescrape_targets.py:_run()`
  (the recent-cooldown / noop / recent-zero-insert / stalled-partial /
  low-backlog filter sequence)
- standalone predicate functions: `_is_noop_parser_upgrade_deferred`,
  `_is_recent_exhaustive_zero_insert_run`,
  `_load_recent_stalled_partial_target_ids`,
  `_is_blocked_parser_upgrade_deferred`

**Disposition:**
- inline filter chain in `_run()`: removed once `should_run_maintenance(ctx)`
  is wired in.
- predicate functions: each becomes the body of a `MaintenanceRule`
  implementation registered into the maintenance rule chain. The predicates
  relocate into `eligibility.py` to keep all rule logic in one module.

**Exit criteria for deletion:**
- maintenance snapshot/replay verification proves identical target-selection
  output for the maintenance planner across at least one production cycle
- `b2b_parser_upgrade_maintenance.py` task wrapper consumes the rule chain
- delete: inline filter-chain code in `_run()` and the standalone predicate
  functions.

### Phase 4 deprecations: per-parser pagination loops

This phase deprecates per-source pagination ONLY for the structured-review
family (g2, capterra, gartner, getapp, peerspot, software_advice, trustpilot,
trustradius). The migration is per-source, not en masse -- until a source
migrates to the new `BaseParser` engine, its existing pagination loop stays
operational without warnings or shims.

**Becomes legacy (per migrated source):**
- the parser's pagination loop
- inline `page_has_only_known_source_reviews` short-circuit branches
- inline `stop_reason` propagation

**Disposition:**
- per-source: replaced by `BaseParser._scrape()` method. Source-specific
  extraction stays in the parser subclass.
- non-structured families (Reddit / StackOverflow / GitHub / HN / RSS) are
  NOT marked as legacy. They are explicitly out of Phase 4 scope and are NOT
  subject to deprecation under this plan.

**Exit criteria for deletion (per source):**
- parser migrated, fixture-driven extraction tests pass
- one production cycle confirms no behavior drift on that source
- delete: legacy pagination loop in that parser only

### Cross-cutting deprecation policy

These rules apply to every phase.

1. **No soft deprecation.** Every deprecated surface has either a removal
   target (a named slice) or a removal trigger (after parity verification +
   stated condition). A `DEPRECATED` comment without a removal target is not
   allowed.

2. **Source-level annotation.** Every deprecated function, dataclass,
   frozenset, or config field gets an inline comment of the form:

   ```python
   # DEPRECATED Phase 1 (2026-04-28). Removal target: after Phase 1 parity
   # verification. Use atlas_brain.services.scraping.eligibility.should_scrape_now().
   ```

   Removal PRs cite this comment.

3. **Single migration path.** Once a surface is deprecated, no new code may
   use it -- including in tests. Existing tests get migrated in the same PR
   that introduces the new path.

4. **Compatibility shims are time-bounded.** Any shim is named with its
   removal trigger inline. Example:
   `_TRANSITIONAL_source_membership_csv_override` makes the lifetime visible.

5. **No dual writes.** If a surface is migrated, only the new path writes to
   it. The old path becomes read-only via shim, never re-implemented in
   parallel.

## Risks

### Risk 1: Over-generalizing parser abstraction too early
Mitigation:
- keep parser-family abstraction after registry
- migrate only the structured family first

### Risk 2: Breaking maintenance while unifying source behavior
Mitigation:
- preserve current maintenance runner behavior in Phase 1
- move planner enrollment in a dedicated Phase 3

### Risk 3: Mixing thresholds and source enrollment in one place
Mitigation:
- config owns thresholds
- registry owns source participation

### Risk 4: Reintroducing silent-failure gate enrollment
Mitigation:
- use `GateName` enum, not free-form strings
- validate source registration at import time

### Risk 5: Refactor fatigue from current uncommitted patch stack
Mitigation:
- stop adding new reactive patches
- treat current patches as reference behavior, not architecture

## Open Decisions

1. Should `SourceSpec` live beside `sources.py` or eventually replace parts of it?
   - recommendation: live beside it first

2. Should maintenance use the same rule primitive as scrape?
   - recommendation: yes
   - use the same `RuleChain[Ctx]` primitive and typed gate enrollment model
   - do not use the same rule set or the same context type

3. Should parser registration move into `SourceSpec` immediately?
   - recommendation: no
   - keep parser registration backward-compatible first, then optionally derive parser registry from `SourceSpec` later

## Acceptance Bar

This refactor is successful when:

- adding a new pre-scrape gate requires one rule registration, not path-by-path wiring
- adding a new source requires one parser and one `SourceSpec` registration, not scattered source memberships
- autonomous intake and manual API cannot silently drift apart on eligibility behavior
- source behavior enrollment is visible in one file

## Follow-Up Work

### Vendor-Add Audit

The product bar also depends on vendor onboarding, not only source onboarding.
That audit is not part of Phase 1, but it should be tracked explicitly as a
follow-up slice:

- slug-format validation
- operator-visible failure states
- governance/source-fit feedback on vendor add
- retryability and diagnosis for failed initial scrapes

## Immediate Next Step

Start Phase 1 only:
- add `atlas_brain/services/scraping/eligibility.py`
- move the three existing pre-scrape gates behind `should_scrape_now()`
- route both autonomous and manual paths through it

Do not begin parser-family abstraction until the shared decision path and source registry exist.
