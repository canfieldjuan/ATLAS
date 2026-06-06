# PR: prepare `landing_page` wiring (E2 of N) — gated off in production until E2.5 wires scope

## Updates from review

Codex P1 review on the initial commit (`3a215645`) flagged
that turning on `landing_page` in production while the host
mount in `atlas_brain/api/__init__.py` doesn't pass a
`scope_provider` would persist drafts under empty
`account_id` -- cross-tenant leakage in any authenticated
B2B deployment.

Response: the `_build_landing_page_service` helper, the DI
kwargs, and the test inventory all stay as-is (proves the
wiring works mechanically). The production code path now
gates the actual wiring behind a new
`enable_db_services` kwarg defaulting to `False`. Production
callers (`atlas_brain/api/__init__.py`'s
`build_content_ops_execution_services()`) leave the bundle's
`landing_page` slot `None` until the follow-up slice (E2.5)
wires `scope_provider` from the authenticated `AuthUser`
and flips the flag on. New regression test
`test_landing_page_slot_stays_none_in_production_default`
pins the safe default.

This slice's value: the wiring helper is in place and tested,
ready for E2.5 to flip the switch the moment scope is wired.



## Why this slice exists

PR #452 (E1) wired `signal_extraction` into the host's
`execution_services_provider` for `/content-ops/execute`.
PR #453 landed the host-side LLM + Skill adapters that the
remaining 5 generators all need. This slice closes the
smallest of those 5 -- `landing_page` -- which has the lightest
dependency footprint:

- `LandingPageRepository` (Postgres-backed via the extracted
  package's `PostgresLandingPageRepository`).
- `LLMClient` (host adapter from PR #453).
- `SkillStore` (host adapter from PR #453).
- **No** `IntelligenceRepository` (the other 4 LLM-needing
  generators all need this).

Wiring `landing_page` proves the full LLM + Skill + Postgres
chain works end-to-end against the host's infrastructure
without first needing an `IntelligenceRepository` factory.
After this lands, the bundle advertises 2 of 6 outputs as
`configured_outputs`; the remaining 4 (`campaign`,
`blog_post`, `report`, `sales_brief`) follow in E3+ once an
`IntelligenceRepository` host factory ships.

## Scope (this PR)

The bundle factory and one regression test. Frontend + API
layer untouched.

### Files touched

1. `atlas_brain/_content_ops_services.py` -- extend the
   bundle factory:
   - Add a host-side helper that builds a
     `LandingPageGenerationService` from
     `PostgresLandingPageRepository(pool=get_db_pool())`,
     `build_content_ops_llm_client()`, and
     `build_content_ops_skill_store()`. If the LLM client is
     `None` (no active model), skip the wiring and leave the
     bundle's `landing_page` slot `None` -- the executor's
     per-step dispatcher then surfaces `service_not_configured`
     to the UI's Execute enable-state.
   - Update `build_content_ops_execution_services()` to call
     the helper alongside the existing `signal_extraction`
     wiring.
   - Take an `llm_factory` / `skills_factory` / `pool_factory`
     dependency-injection set so tests can stub each piece
     without triggering host-side imports.
   - ~50 LOC delta.

2. `tests/test_atlas_content_ops_execution_services.py` --
   extend the existing test:
   - `test_landing_page_runs_through_host_bundle_when_dependencies_present`
     -- pass stubbed LLM client, skill store, and pool; assert
     the bundle has `landing_page` populated and a `/execute`
     run with `outputs=["landing_page"]` reaches the service
     layer (mocked stub returns a fake draft).
   - `test_landing_page_skipped_when_no_active_llm` --
     `build_content_ops_llm_client` returns `None`; bundle's
     `landing_page` slot stays `None`; configured_outputs =
     `('signal_extraction',)`.
   - Update existing
     `test_unwired_outputs_still_return_service_not_configured`
     -- pick a different unwired output (e.g. `report`) since
     `landing_page` is no longer in the unwired set.
   - Update
     `test_bundle_only_advertises_wired_outputs` -- expected
     tuple is now `('landing_page', 'signal_extraction')` when
     LLM is present.
   - ~120 LOC delta.

3. `plans/PR-Content-Ops-Execution-Services-Wire-2.md`
   (this file).

### What's NOT in this slice

- **`campaign` / `blog_post` / `report` / `sales_brief`**
  service wiring. Each needs an `IntelligenceRepository` host
  factory; that's a separate slice (E3 or its own
  infrastructure-only PR).
- **Reasoning context provider injection.** PR #402 already
  wired the route-level seam; the bundle's
  `with_reasoning_context()` derivation handles per-request
  rebinding once the route resolves a provider. This slice
  doesn't touch reasoning.
- **`PostgresIntelligenceRepository` factory.** The host has
  `get_db_pool()`; the extracted package has
  `PostgresIntelligenceRepository(pool=...)`. Wiring it is
  trivial mechanically but `landing_page` doesn't need it,
  so out of scope.
- **End-to-end smoke test** posting to `/api/v1/content-ops/execute`.
  Today's tests stop at the bundle / service layer.

## Mechanism

The bundle factory grows a single helper:

```python
def _build_landing_page_service(
    *,
    llm: LLMClient | None,
    skills: SkillStore,
    pool: Any,
) -> LandingPageGenerationService | None:
    if llm is None:
        # No active LLM -- leave the slot None so the executor
        # surfaces service_not_configured.
        return None
    return LandingPageGenerationService(
        landing_pages=PostgresLandingPageRepository(pool=pool),
        llm=llm,
        skills=skills,
    )
```

`build_content_ops_execution_services()` calls it after the
`signal_extraction` slot is set, threading in the LLM /
skills / pool from the existing factories. When the LLM is
absent, `landing_page` falls through to the same
`service_not_configured` path other unwired outputs use.

The factory accepts optional `llm_factory`, `skills_factory`,
and `pool_factory` kwargs (default `None`). When present, the
factory uses them; when omitted, it imports the canonical
host singletons lazily. Same DI pattern as PR #453 -- tests
pass stubs without triggering the host's full init chain
(torch / asyncpg / etc.).

The host's `DatabasePool` already exposes `fetchval` /
`fetch` / `execute` (`atlas_brain/storage/database.py:114-138`),
the asyncpg-shaped methods that
`PostgresLandingPageRepository` calls. No further adapter
needed; the duck-typed pool is structurally compatible.

## Intentional

- **`landing_page` first among LLM-needing services.** It has
  no `IntelligenceRepository` dependency, so it lights up the
  full LLM + Skill + Postgres chain with the smallest possible
  footprint. Other generators (`campaign`, `blog_post`,
  `report`, `sales_brief`) all need `IntelligenceRepository`,
  so they cluster behind that follow-up.
- **Slot stays `None` when LLM is absent.** Same pattern as
  E1: skip wiring rather than wire a stub. The catalog
  endpoint surfaces "not configured" to the UI's Execute
  enable-state.
- **Dependency injection on three factories.** LLM, skills,
  and pool all become testable seams. Test harness doesn't
  need a real DB or an active LLM to verify the bundle's
  slot-population logic.
- **No module-level singleton for the
  `LandingPageGenerationService`.** Unlike
  `SignalExtractionService` (stateless), the landing-page
  service holds a Postgres repository whose pool can change
  across host restarts. Build-on-demand so the latest pool is
  always used.

## Deferred

- `PostgresIntelligenceRepository` host factory + plug into
  `campaign` / `blog_post` / `report` / `sales_brief`. E3+
  slices.
- Reasoning context provider host factory. The route-level
  seam exists (PR #402); when the host wants
  multi-pass-by-default, a follow-up slice plugs
  `MultiPassCampaignReasoningProvider` in.
- End-to-end smoke test that POSTs to
  `/api/v1/content-ops/execute` with
  `outputs=["landing_page"]` and asserts a real draft lands.
- Per-request scope plumbing (today's wiring uses an empty
  scope by default).

## Verification

- `pytest tests/test_atlas_content_ops_execution_services.py`
  -- updated tests pass.
- AST parse + ASCII checks on the modified module + test.
- Smoke: `python -c "from atlas_brain._content_ops_services
   import build_content_ops_execution_services; b =
   build_content_ops_execution_services(...stubs...);
   print(b.configured_outputs())"` returns
  `('landing_page', 'signal_extraction')` when LLM is wired,
  `('signal_extraction',)` when not.

## Estimated diff size

- `_content_ops_services.py`: ~70 LOC delta (was ~50; the DI
  kwargs add a bit).
- Test: ~150 LOC delta (3 new tests + 2 existing-test
  updates).
- Plan doc: ~190 LOC.

Total: ~410 LOC. Marginally over the 400 LOC PR target;
splitting the DI kwargs into a separate slice would leave
the bundle factory partially wired with no testable
short-circuit path. Indivisible.
