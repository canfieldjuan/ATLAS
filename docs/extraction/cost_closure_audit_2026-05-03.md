# Cost-Closure Boundary Audit

Date: 2026-05-03

## Executive Decision

The cost-closure pieces (`llm_exact_cache.py`, `provider_cost_sync.py`, migrations 251 + 258, plus four new code modules) land **inside the existing `extracted_llm_infrastructure/` scaffold**, not as a new `llm-spend-py` package.

The existing extraction already owns the LLM provider abstractions, the FTL tracer, the semantic cache, the Anthropic batch machinery, and the `llm_usage` schema (migrations 127, 252, 253, 257). Adding cache-key-hashed exact caching, provider invoice reconciliation, cache-savings telemetry, and runtime budget gating to the same product completes the "spend closure" loop end-to-end inside one sellable surface. Carving them into a second package would duplicate the standalone substrate, fragment the dashboard story, and force consumers to install two things that are operationally one product.

The wedge -- and the reason this is sellable -- is **invoice reconciliation against per-call usage logs, exposed alongside cache-hit dollar savings**. Every observability tool in the LLM space logs token usage; almost none reconcile that log against the provider's own billing API. Cost-closure is the differentiated content. The cache and the usage log are table stakes.

## Verified Current State

The `extracted_llm_infrastructure/` manifest contains 14 file mappings + 6 migration mappings (127, 130, 252, 253, 255, 257) across:

| Surface | Files |
|---|---|
| LLM providers | `services/llm/{anthropic,openrouter,ollama,vllm,groq,together,hybrid,cloud}.py` |
| Routing | `services/llm_router.py`, `pipelines/llm.py` |
| Tracing | `services/tracing.py` |
| Caching (semantic only) | `reasoning/semantic_cache.py` |
| Batching | `services/b2b/anthropic_batch.py`, `services/b2b/cache_strategy.py` |
| Schema | migrations 127 (llm_usage), 130 (semantic_cache), 252 (cache breakdown columns), 253 (vendor + run_id), 255 (anthropic batches), 257 (reasoning attribution) |

Phase: 2 (standalone toggle landed via `EXTRACTED_LLM_INFRA_STANDALONE=1`; Phase 3 decoupling pending).

What is **missing** for the cost-closure pitch to be operationally complete:

| Surface | Currently | Gap |
|---|---|---|
| Exact LLM response cache | `atlas_brain/services/b2b/llm_exact_cache.py` (378 LOC), table `b2b_llm_exact_cache` (mig 251, 18 LOC) | Not in manifest; competitive_intelligence holds a Phase 1 bridge stub at `services/b2b/llm_exact_cache.py` that re-exports atlas symbols |
| Provider billing fetch + reconcile | `atlas_brain/services/provider_cost_sync.py` (286 LOC), tables `llm_provider_usage_snapshots` + `llm_provider_daily_costs` (mig 258, 32 LOC) | Not in manifest |
| Cache-hit dollar savings telemetry | None -- hit counters live in memory inside `enrichment_row_runner.py` | Net-new code; no Atlas implementation to lift |
| Local-vs-invoiced drift reporting | Data exists (snapshots + daily_costs vs llm_usage rollups) | No code computes/reports drift |
| Hard-cap budget gating | None -- `enrichment_budget.py` extracts budget signals from review text, not a runtime gate | Net-new code |
| OpenAI provider billing fetcher | None -- `provider_cost_sync.py` covers OpenRouter + Anthropic only | Net-new code |

## Why Adding To Existing Extraction Beats New Package

Three concrete reasons settled this in the 2026-05-03 strategy thread:

1. **The product is one operational thing.** Customers who install "LLM cost intelligence" expect the cache, the usage log, and the reconciliation in one box. Two packages would force them to wire the same DB pool, the same migrations runner, the same provider credentials twice.
2. **Bug fixes flow once.** Provider APIs churn (OpenRouter changed credit endpoints twice in 2025; Anthropic added the admin API mid-year). One codebase = one patch. Two parallel codebases = forever-tax on every provider change.
3. **Atlas dogfoods the package.** Once the additions land, Atlas continues running on the extracted surface via Phase 3 decoupling. That's the marketing story: "we use this in production processing N million LLM calls/month." Splitting weakens that.

## Files To Lift From Atlas

| Source | Target in extraction | LOC | Verb |
|---|---|---|---|
| `atlas_brain/services/b2b/llm_exact_cache.py` | `extracted_llm_infrastructure/services/b2b/llm_exact_cache.py` | 378 | Add (manifest mapping) |
| `atlas_brain/services/provider_cost_sync.py` | `extracted_llm_infrastructure/services/provider_cost_sync.py` | 286 | Add (manifest mapping) |
| `atlas_brain/storage/migrations/251_b2b_llm_exact_cache.sql` | `extracted_llm_infrastructure/storage/migrations/251_b2b_llm_exact_cache.sql` | 18 | Add (manifest mapping); rename target to `251_llm_exact_cache.sql` deferred to Phase 3 to avoid breaking existing Atlas references in this PR |
| `atlas_brain/storage/migrations/258_provider_cost_reconciliation.sql` | `extracted_llm_infrastructure/storage/migrations/258_provider_cost_reconciliation.sql` | 32 | Add (manifest mapping) |

Total lift: ~714 LOC across 4 files. All four are byte-for-byte copies under the existing Phase 1 scaffold contract -- they continue importing from `atlas_brain` until Phase 3.

## Bridge Reconciliation

`extracted_competitive_intelligence/services/b2b/llm_exact_cache.py` exists as a Phase 1 bridge stub that re-exports `atlas_brain.services.b2b.llm_exact_cache` programmatically. It is not in the competitive-intelligence manifest's `mappings` or `owned` lists.

Once the cache lands in `extracted_llm_infrastructure`, the bridge stub keeps working unchanged because Atlas still owns the source-of-truth path. A later cross-product migration (deferred, not in scope for cost-closure) will rewire the bridge to point at `extracted_llm_infrastructure.services.b2b.llm_exact_cache` instead. That migration is symmetrical to PR #80's wedge-registry compat-wrapper pattern.

## New Code Required (Not Liftable From Atlas)

These four modules do not exist in `atlas_brain/`. They are net-new and define the differentiated wedge.

### `services/cost/cache_savings.py` (PR-A3)

Persists cache-hit "saved spend" rows so `$ saved by cache last month` is queryable, not just a memory counter.

Public API:

```python
async def record_cache_hit(
    pool,
    *,
    cache_key: str,
    namespace: str,
    provider: str,
    model: str,
    would_have_been_input_tokens: int,
    would_have_been_output_tokens: int,
    would_have_been_cost_usd: Decimal,
    attribution: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None: ...

class CacheSavingsRollup(TypedDict):
    total_saved_usd: Decimal
    total_saved_input_tokens: int
    total_saved_output_tokens: int
    hit_count: int
    by_namespace: Mapping[str, Decimal]
    by_attribution_dim: Mapping[str, Mapping[str, Decimal]]

async def daily_cache_savings(
    pool, *, date_range: tuple[date, date], attribution_key: str | None = None,
) -> CacheSavingsRollup: ...
```

New migration owns `llm_cache_savings` table (one row per hit):

```
cache_key, namespace, provider, model,
saved_input_tokens, saved_output_tokens, saved_cost_usd,
attribution JSONB, metadata JSONB,
hit_at TIMESTAMPTZ
```

### `services/cost/drift.py` (PR-A4)

Computes local-usage-sum vs provider-billed-amount per (provider, model, day, attribution).

```python
@dataclass(frozen=True)
class DriftRow:
    provider: str
    model: str
    cost_date: date
    local_usd: Decimal
    invoiced_usd: Decimal
    delta_usd: Decimal
    delta_pct: float
    explained_by: list[str]   # heuristic chips: "retry-counted-twice", "cached-call-billed-anyway", ...

async def compute_drift(
    pool, *, provider: str, date_range: tuple[date, date],
) -> list[DriftRow]: ...
```

### `services/cost/budget.py` (PR-A4)

Runtime budget gate. Returns an allow/deny decision before LLM calls.

```python
@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str | None        # e.g. "daily_cap_exceeded", "attribution_cap_exceeded:customer_id=acme"
    consumed_usd: Decimal
    cap_usd: Decimal

class BudgetGate:
    def __init__(
        self, pool, *,
        daily_cap_usd: Decimal | None = None,
        per_attribution_caps: Mapping[str, Mapping[str, Decimal]] | None = None,
    ): ...

    async def check_before_call(
        self, *,
        estimated_cost_usd: Decimal,
        attribution: Mapping[str, str] | None = None,
    ) -> BudgetDecision: ...
```

### `services/llm/openai.py` (PR-A4)

OpenAI Costs API integration. Mirrors the `services/llm/anthropic.py` provider-billing surface. No corresponding Atlas implementation; commercially required because the package targets OpenAI users in v1.

## Schema Strategy

The existing `llm_usage` schema includes structured columns (`vendor_name`, `run_id`, `source_name`, `event_type`, `entity_type`, `entity_id`) added by Atlas migrations 253 + 257. These columns are domain-shaped (B2B-flavored names) and not generalizable as-is for downstream package consumers.

Decision: **leave the columns in the schema unchanged for cost-closure scope.** The cost-closure additions write to the existing structure; column generalization (renaming to opaque attribution dims, or moving to a `attribution JSONB` column with PostgreSQL generated columns derived from it) is **deferred to Phase 3**.

Rationale: the rename touches every call site in Atlas's enrichment + campaign + briefing tasks. That is a separate refactoring slice with its own risk profile. Cost-closure can ship today by writing to the existing columns; the abstraction can tighten later without re-doing the cost-closure work.

The `llm_cache_savings` table introduced in PR-A3 ships with `attribution JSONB` from day one, since it's net-new -- establishing the pattern for future tables without retrofitting.

## Public API Additions To `extracted_llm_infrastructure`

After all four follow-up PRs land, the package exposes these surfaces that today only Atlas consumes:

```python
# Existing in extraction (already exposed):
from extracted_llm_infrastructure.services.tracing import record_llm_call_span
from extracted_llm_infrastructure.services.llm import resolve_llm
from extracted_llm_infrastructure.reasoning.semantic_cache import SemanticCache

# New after PR-A1:
from extracted_llm_infrastructure.services.b2b.llm_exact_cache import (
    lookup_cached_text, store_cached_text,
    compute_cache_key, build_request_envelope,
)

# New after PR-A2:
from extracted_llm_infrastructure.services.provider_cost_sync import (
    sync_provider_costs,
)

# New after PR-A3:
from extracted_llm_infrastructure.services.cost.cache_savings import (
    record_cache_hit, daily_cache_savings,
)

# New after PR-A4:
from extracted_llm_infrastructure.services.cost.drift import compute_drift
from extracted_llm_infrastructure.services.cost.budget import BudgetGate
from extracted_llm_infrastructure.services.llm.openai import OpenAIProviderClient
```

Internal modules (`_normalize_*`, `_resolve_pool`, `_safe_float`, etc.) remain underscore-prefixed and out of the public surface.

## Cross-Product Dependency Implications

Per `extracted/_shared/docs/cross_product_dependency_graph.md`:

- Competitive Intelligence already declares dependency on LLM Infrastructure. Cost-closure additions inherit cleanly.
- Content Pipeline already declares dependency on LLM Infrastructure. Same.
- Quality Gate (planned) and Intent Router (planned) also point at LLM Infrastructure.

No new cross-product dependencies are introduced. The bridge stub in competitive-intelligence (`services/b2b/llm_exact_cache.py`) keeps working through Phase 1; the future migration to the extracted path is symmetrical to PR #80's wedge-registry pattern and lives under SYS-CI work, not this slice.

## Atlas Migration Required (Lightweight)

This audit's follow-up PRs do **not** touch Atlas -- Phase 1 byte-for-byte scaffolding only. Atlas continues to import from its own `atlas_brain.services.*` paths. Phase 3 decoupling (rewriting Atlas call sites to import from the extracted package) is a separate sequence outside cost-closure scope.

The only Atlas-side coordination required for cost-closure: the existing call sites enumerated below remain in Atlas's tree and continue to work because the source files are untouched.

| Atlas call site | What it imports today | Action in cost-closure |
|---|---|---|
| `atlas_brain/config.py` | `b2b_llm_exact_cache_enabled` setting | none |
| `atlas_brain/autonomous/scheduler.py` | schedules `provider_cost_sync` task | none |
| `atlas_brain/autonomous/tasks/provider_cost_sync.py` | thin wrapper | none |
| `atlas_brain/autonomous/tasks/{b2b_battle_cards,b2b_campaign_generation,b2b_churn_reports,b2b_enrichment_repair,b2b_product_profiles,b2b_tenant_report}.py` | call `lookup_cached_text` / `store_cached_text` | none |

## Follow-Up PR Sequence

### PR-A0: Cost-Closure Boundary Audit

This document.

Acceptance criteria:

- file lift list verified against current Atlas tree (paths + LOC)
- new-code modules classified with public-API stubs
- schema strategy decision recorded
- cross-product dependency implications recorded
- bridge-stub reconciliation recorded
- follow-up PR sequence is explicit with acceptance criteria each

### PR-A1: Add `llm_exact_cache` To LLM-Infrastructure Manifest

Scope:

- add manifest mapping for `services/b2b/llm_exact_cache.py` (378 LOC) and migration 251
- run `extracted/_shared/scripts/sync_extracted.sh extracted_llm_infrastructure`
- update `extracted_llm_infrastructure/README.md` "What's in scope" table to add the new entries
- update `extracted_llm_infrastructure/STATUS.md` if it tracks file count
- import-debt allowlist update if new relative imports surface

Acceptance criteria:

- `bash extracted/_shared/scripts/validate_extracted.sh extracted_llm_infrastructure` passes
- `bash extracted/_shared/scripts/check_ascii_python.sh extracted_llm_infrastructure` passes
- `python extracted/_shared/scripts/check_extracted_imports.py extracted_llm_infrastructure` passes (atlas-fallback mode acceptable for Phase 1)
- README scope table includes the two new entries

### PR-A2: Add `provider_cost_sync` To LLM-Infrastructure Manifest

Scope:

- add manifest mapping for `services/provider_cost_sync.py` (286 LOC) and migration 258
- sync + validate + ASCII check
- README scope table update

Acceptance criteria:

- same validation suite as PR-A1
- README scope table includes the two new entries
- migration 258 lands alongside the file (provider_cost_sync.py imports both tables it manages)

### PR-A3: Cache-Savings Persistence (NEW CODE)

Scope:

- new file `extracted_llm_infrastructure/services/cost/cache_savings.py` with the public API stubbed in this audit
- new migration owning `llm_cache_savings` table (in extraction's `storage/migrations/` only -- not back-ported to Atlas)
- new tests `tests/test_extracted_llm_infrastructure_cache_savings.py` covering record + rollup + attribution roll-up
- add to manifest `owned` list (since this is owned, not synced from Atlas)

Acceptance criteria:

- standalone toggle still passes (`EXTRACTED_LLM_INFRA_STANDALONE=1` smoke test runs)
- record + rollup tests cover empty range, single hit, multiple hits across attribution dims, decimal precision
- README scope table updates

### PR-A4: Drift Report + Budget Gate + OpenAI Provider (NEW CODE)

May split if too large. Likely sub-PRs:

- PR-A4a: `services/cost/drift.py` + tests + README update
- PR-A4b: `services/cost/budget.py` + tests + README update
- PR-A4c: `services/llm/openai.py` + tests + README update

Acceptance criteria (each):

- module is owned (not synced) -- added to manifest `owned` list
- standalone toggle smoke passes
- public API matches the stub in this audit
- tests cover at minimum: happy path, edge case, contract type assertions

### PR-A5 (DEFERRED, OUT OF COST-CLOSURE SCOPE)

Cross-product migration: rewire `extracted_competitive_intelligence/services/b2b/llm_exact_cache.py` bridge stub to import from `extracted_llm_infrastructure.services.b2b.llm_exact_cache` once the standalone toggle is fully Phase 3. Symmetrical to PR #80's wedge-registry compat-wrapper migration. Defer until both products are deeper into Phase 3.

## Immediate Next Code Slice

Start with **PR-A1** (manifest add for `llm_exact_cache.py` + migration 251). It is the smallest, lowest-risk slice and unblocks PR-A2 and PR-A3 in parallel.

Scope of the immediate next slice:

- one manifest mapping addition
- one README table row
- one sync run, one validate run, one ASCII check
- no code changes beyond what `extracted/_shared/scripts/sync_extracted.sh` produces
