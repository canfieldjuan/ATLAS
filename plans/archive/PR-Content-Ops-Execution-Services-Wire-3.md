# PR: wire `campaign`, `report`, and `sales_brief` services (E3 of N)

## Why this slice exists

After PR #455 (E2.5), the host's `execution_services_provider`
bundle wires 2 of 6 outputs: `signal_extraction` + `landing_page`.
The remaining 4 outputs are:

- `campaign`, `report`, `sales_brief` -- all share an identical
  dependency shape: `IntelligenceRepository` (read campaign
  opportunities) + the per-output `*Repository` (persist drafts) +
  `LLMClient` + `SkillStore`.
- `blog_post` -- different shape: `BlogBlueprintRepository` +
  `BlogPostRepository` + `LLMClient` + `SkillStore`. No
  `IntelligenceRepository`. Separate slice (E4).

This slice wires all 3 IntelligenceRepository-dependent
generators in a single PR -- the wiring is mechanical and
identical across the three, so combining them avoids
shipping three near-duplicate slices. After E3 lands, the
bundle advertises 5 of 6 outputs; only `blog_post` remains.

## Scope (this PR)

The bundle factory and the test file. No frontend or route
changes.

### Files touched

1. `atlas_brain/_content_ops_services.py`:
   - Add `_build_campaign_service`, `_build_report_service`,
     `_build_sales_brief_service` helpers. Each follows the
     same shape as `_build_landing_page_service` (PR #454):
     short-circuit to `None` when LLM or pool is absent;
     otherwise construct the service from
     `PostgresIntelligenceRepository` + the per-output Postgres
     repo + the LLM/Skill adapters.
   - Update `build_content_ops_execution_services()` to call
     all three helpers when `enable_db_services=True`. Single
     `PostgresIntelligenceRepository(pool=pool)` instance is
     shared across the three services (Postgres pool is the
     real connection budget; the dataclass wrapper is
     lightweight).
   - ~80 LOC delta.

2. `tests/test_atlas_content_ops_execution_services.py`:
   - 3 new "wired-when-LLM-active-and-db-enabled" tests (one
     per output) + 3 new "slot stays None when no LLM" tests
     + update the existing `configured_outputs` tests to
     reflect the new tuple `(campaign, landing_page, report,
     sales_brief, signal_extraction)`.
   - Rewrite the unwired-output canary to pick `blog_post`
     since it's the only output left in the unwired set.
   - ~150 LOC delta.

3. `plans/PR-Content-Ops-Execution-Services-Wire-3.md`
   (this file).

### What's NOT in this slice

- **`blog_post` service.** Different repo shape
  (`BlogBlueprintRepository` -- not Postgres-backed by the
  same pool). Separate slice (E4).
- **Per-service config tuning.** Each
  `*GenerationConfig` has defaults (skill_name,
  default_brief_type, etc.); the host accepts the defaults.
  Operator-level tuning (e.g. `b2b_growth` plan caps a
  different temperature) is out of scope.
- **Reasoning context provider host wiring.** Route seam
  shipped in PR #402; the bundle's
  `with_reasoning_context()` derivation handles per-request
  rebinding when the host wires a provider. Not part of E3.
- **End-to-end smoke test posting to `/api/v1/content-ops/execute`**
  with each new output. Today's tests stop at the
  bundle/service layer.

## Mechanism

Each helper follows the established `_build_landing_page_service`
template -- short-circuit on missing dependency, otherwise
construct the service:

```python
def _build_report_service(
    *,
    intelligence: IntelligenceRepository | None,
    llm: LLMClient | None,
    skills: SkillStore,
    pool: Any,
) -> ReportGenerationService | None:
    if llm is None or pool is None or intelligence is None:
        return None
    return ReportGenerationService(
        intelligence=intelligence,
        reports=PostgresReportRepository(pool=pool),
        llm=llm,
        skills=skills,
    )
```

`build_content_ops_execution_services()` resolves the shared
`PostgresIntelligenceRepository(pool=pool)` once and passes
it to all three service builders -- same pool, same wrapper
instance. The repository is a frozen dataclass with no
per-call state, so sharing is safe.

The `enable_db_services` flag still gates all DB-backed
generators behind the production scope-wiring contract from
PR #455. Tests pass `enable_db_services=True` explicitly to
exercise the wiring path.

## Intentional

- **All 3 IntelligenceRepository-dependent services in one
  PR.** The wiring is identical across the three; shipping
  three near-duplicate slices would be process overhead with
  no review value. Each helper is ~10 LOC; the only delta is
  the per-output Postgres repo class.
- **Shared `PostgresIntelligenceRepository` instance.** The
  dataclass is immutable and the underlying pool is shared
  anyway; constructing three copies adds no isolation and
  one-extra-allocation per request churn.
- **Defensive `intelligence is None` guard** matches the
  existing LLM/pool guards from PR #454. Today the
  intelligence repo is always built when the pool is
  available, but the guard keeps the slot-skipping shape
  uniform across all four DB-backed generators (landing_page
  + the three this PR adds).
- **Pool-only `PostgresIntelligenceRepository(pool=pool)`
  construction**, omitting `opportunity_table` -- accepts
  the package's `"campaign_opportunities"` default. Operator
  override would be a separate slice if needed.
- **`blog_post` deliberately deferred.** Different repo
  shape (`BlogBlueprintRepository`); plugging it in alongside
  the IntelligenceRepository services would conflate two
  unrelated wiring patterns. Each gets its own slice.

## Deferred

- `blog_post` service wiring (E4) -- needs
  `BlogBlueprintRepository` + `BlogPostRepository` + LLM +
  Skills.
- Multi-pass reasoning provider host wiring -- separate
  follow-up; the bundle's per-request `with_reasoning_context`
  derivation is already in place (PR #402).
- End-to-end smoke test posting to
  `/api/v1/content-ops/execute` with each generator and
  asserting real drafts persist under the authenticated
  tenant. Today's tests stop at the bundle layer.
- Per-service config tuning (e.g. operator-level
  temperature, max_tokens overrides).

## Verification

- `pytest tests/test_atlas_content_ops_execution_services.py`
  -- updated tests pass.
- AST + ASCII checks on the modified module + test file.

## Estimated diff size

- `_content_ops_services.py`: ~80 LOC delta.
- Test: ~190 LOC delta (13 tests rather than the projected
  ~10; docstring inventory expanded).
- Plan doc: ~165 LOC.

Total actual: +414 / -44 = 458 changes. Marginally over the
400-LOC soft cap. Indivisible -- the Intentional bullet
about combining the 3 IntelligenceRepository-dependent
services in one PR is the justification: each helper is
~10 LOC, three near-duplicate slices would be process
overhead with no review value. Tests dominate the line
count; the production-code delta itself is well under cap.
