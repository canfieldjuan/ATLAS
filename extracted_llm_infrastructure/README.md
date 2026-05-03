# extracted_llm_infrastructure

Phase 1 scaffold for extracting the **LLM Infrastructure & Cost Optimization** subsystem from `atlas_brain/` into a standalone, sellable package.

## What this is

A byte-for-byte snapshot of the LLM-infrastructure surface inside `atlas_brain/`, copied here so Phase 2 work (decoupling from Atlas globals) can iterate on the scaffold without touching production code. The contents are not yet runnable outside Atlas — every module still imports from `atlas_brain.config`, `atlas_brain.services.*`, etc.

The scaffold is **purely additive**. Atlas continues to import from its own paths; this directory is parallel infrastructure.

The pattern mirrors the content-pipeline scaffold under
`extracted_content_pipeline/`.

## What's in scope (Phase 1)

| File | Source | Purpose |
|---|---|---|
| `services/b2b/anthropic_batch.py` | `atlas_brain/services/b2b/anthropic_batch.py` | Anthropic Message Batches with cost tracking, dedup, replay (~1,450 LOC) |
| `services/b2b/cache_strategy.py` | `atlas_brain/services/b2b/cache_strategy.py` | Per-stage cache-mode registry (15 stages: exact / semantic / evidence_hash) |
| `pipelines/llm.py` | `atlas_brain/pipelines/llm.py` | LLM resolution by workload, FTL trace emission, JSON parsing, OpenRouter routing |
| `reasoning/semantic_cache.py` | `atlas_brain/reasoning/semantic_cache.py` | Postgres-backed semantic memory with exponential confidence decay |
| `services/llm_router.py` | `atlas_brain/services/llm_router.py` | Workflow-based singleton fallback (cloud / draft / triage / reasoning) |
| `services/llm/{anthropic,openrouter,ollama,vllm,groq,together,hybrid,cloud}.py` | `atlas_brain/services/llm/*.py` | Eight LLM provider implementations |
| `services/tracing.py` | `atlas_brain/services/tracing.py` | FTL tracer client (token counts, cost telemetry, hierarchical spans) |
| `services/b2b/llm_exact_cache.py` | `atlas_brain/services/b2b/llm_exact_cache.py` | Hash-keyed exact-match LLM response cache (~378 LOC); namespace + request envelope -> SHA -> response_text + usage_json. Cost-closure foundation. |
| `services/cost/cache_savings.py` (OWNED, PR-A3) | (new code) | Cache-savings persistence: one row per cache hit + ``daily_cache_savings`` rollup with per-namespace and per-attribution-dim breakdowns. Closes the "cache hits in memory only" telemetry gap. |
| `storage/migrations/127_llm_usage.sql` | mig 127 | Initial llm_usage table |
| `storage/migrations/130_reasoning_semantic_cache.sql` | mig 130 | Semantic cache + metacognition tables |
| `storage/migrations/251_b2b_llm_exact_cache.sql` | mig 251 | b2b_llm_exact_cache table (cost-closure: cached LLM responses + usage_json) |
| `storage/migrations/259_llm_cache_savings.sql` (OWNED, PR-A3) | (new schema) | llm_cache_savings table (cost-closure: per-hit savings rows for "$ saved by cache" rollups) |
| `storage/migrations/252_llm_usage_cache_breakdown.sql` | mig 252 | Cache + queue token breakdown |
| `storage/migrations/253_llm_usage_vendor_and_run_id.sql` | mig 253 | Vendor + run_id columns |
| `storage/migrations/255_anthropic_message_batches.sql` | mig 255 | Anthropic batch + items tables |
| `storage/migrations/257_llm_usage_reasoning_attribution.sql` | mig 257 | Reasoning attribution column |

## What's out of scope (Phase 3)

Phase 2 (standalone substrate) has landed. Remaining work:

- DB pool / Tracer / LLM `Protocol`-based DI seams across the scaffolded provider modules (Phase 3)
- Replacing `isinstance(AnthropicLLM)` checks throughout `services/b2b/anthropic_batch.py` and `services/llm_router.py` (Phase 3)
- Extracting the private `_convert_messages` from `AnthropicLLM` so batch code does not call a private method (Phase 3)
- Decoupling `SemanticCache` from `asyncpg.Pool` (Phase 3)
- Moving `evidence_hash` computation to a single owner (Phase 3)
- `atlas_brain/api/admin_costs.py` extraction (deferred — admin UI, not core infra)

## Standalone toggle (Phase 2)

Set `EXTRACTED_LLM_INFRA_STANDALONE=1` and the package's substrate
(settings, base class, protocols, registry, db pool) loads from the
local `_standalone/` subpackage instead of delegating to atlas_brain.
The provider modules (`services/llm/*.py`, `services/b2b/anthropic_batch.py`,
etc.) still import from atlas_brain in this scaffold — Phase 3 closes that
loop.

```bash
EXTRACTED_LLM_INFRA_STANDALONE=1 python -c "
from extracted_llm_infrastructure.config import settings
print(settings.llm.anthropic_model)            # claude-haiku-4-5
print(settings.ftl_tracing.pricing.cost_usd(   # 0.001125
    'anthropic', 'claude-haiku-4-5', 1000, 500
))
"
```

The standalone copies live under `extracted_llm_infrastructure/_standalone/`:

| File | Replaces |
|---|---|
| `_standalone/config.py` | atlas_brain.config (slim — only LLM-infra fields) |
| `_standalone/protocols.py` | atlas_brain.services.protocols (verbatim) |
| `_standalone/base.py` | atlas_brain.services.base (torch-free) |
| `_standalone/registry.py` | atlas_brain.services.registry (verbatim) |
| `_standalone/database.py` | atlas_brain.storage.database (slim asyncpg wrapper) |

Env vars match atlas_brain (`ATLAS_LLM_*`, `ATLAS_B2B_CHURN_*`,
`ATLAS_DB_*`) so a single .env file works in both modes.

## Sync workflow

The scaffold must stay byte-equal with the `atlas_brain/` source. Two scripts
maintain that invariant while preserving the stable per-product entry points:

```bash
# Re-copy from atlas_brain into the scaffold (idempotent; safe to re-run)
bash scripts/sync_extracted_llm_infrastructure.sh

# Verify zero drift; exits non-zero if anything differs
bash scripts/validate_extracted_llm_infrastructure.sh
```

When you change a source file under `atlas_brain/`, run the sync afterward and commit the scaffold update in the same PR. The CI workflow `.github/workflows/extracted_llm_infrastructure_checks.yml` enforces zero-drift on every PR that touches the scaffold.

These per-product scripts are thin wrappers over the shared extraction tooling
in `extracted/_shared/scripts/`.

## Local checks

```bash
bash scripts/run_extracted_llm_infrastructure_checks.sh
```

Runs five checks in sequence:

1. `validate_extracted_llm_infrastructure.sh` — byte-diff scaffold vs source
2. `check_ascii_python_llm_infrastructure.sh` — every scaffolded `.py` is ASCII-only
3. `check_extracted_llm_infrastructure_imports.py` — relative imports either resolve inside the scaffold or are listed in `import_debt_allowlist.txt`
4. `smoke_extracted_llm_infrastructure_imports.py` — every public module imports without raising
5. `smoke_extracted_llm_infrastructure_standalone.py` — standalone substrate
   imports use `extracted_llm_infrastructure._standalone`

## Import debt

`import_debt_allowlist.txt` lists every relative-import target in the scaffold that the resolver cannot find at the literal scaffold path. Each entry resolves at runtime via `atlas_brain` (the scaffold and atlas_brain mirror the same package layout, so `from ..config import settings` in a copied module reaches `atlas_brain.config` when the scaffold is imported alongside Atlas).

Phase 2 shrinks this list to zero by either (a) copying the dependency module into the scaffold, or (b) introducing a `Protocol`-based DI seam.

## Why a separate scaffold from `extracted_content_pipeline/`?

The content-pipeline scaffold includes `pipelines/llm.py` and
`services/b2b/anthropic_batch.py` as snapshotted siblings for transition
traceability.

The LLM-infrastructure scaffold lives separately because the LLM-infra subsystem
is a **distinct sellable product** (cost optimization for teams running
Claude/GPT at scale), not a content-pipeline implementation detail. The content
pipeline now points its LLM-facing bridges at `extracted_llm_infrastructure/`;
future scope trimming can remove duplicated transitional copies.

## Status

See `STATUS.md` for the per-file extraction state and remaining Phase 2 / Phase 3 work.
