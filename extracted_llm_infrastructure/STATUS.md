# extracted_llm_infrastructure — STATUS

## Phase 1 — Scaffold creation ✅

| Step | Status |
|---|---|
| Manifest of source → scaffold mappings | ✅ done |
| Verbatim byte-snapshot of 16 Python files | ✅ done (added `services/b2b/llm_exact_cache.py` in PR-A1, `services/provider_cost_sync.py` in PR-A2) |
| Verbatim byte-snapshot of 8 migration SQL files | ✅ done (added migration 251 in PR-A1, 258 in PR-A2) |
| Owned (not synced) Python files: 5 (PR-A3 + PR-A4a + PR-A4b + PR-A4c) | ✅ done (`services/cost/__init__.py`, `services/cost/cache_savings.py`, `services/cost/openai_billing.py`, `services/cost/drift.py`, `services/cost/budget.py`) |
| Owned (not synced) migration SQL files: 1 (PR-A3) | ✅ done (`storage/migrations/259_llm_cache_savings.sql`) |
| Package `__init__.py` files at every level | ✅ done |
| Sync + validate scripts | ✅ done via shared tooling wrappers |
| ASCII / smoke-import / import-debt checks | ✅ done via shared tooling wrappers |
| Driver script `run_extracted_llm_infrastructure_checks.sh` | ✅ done |
| GitHub Actions workflow | ✅ done |
| README + this STATUS file | ✅ done |
| `import_debt_allowlist.txt` documenting atlas_brain dependencies | ✅ done |

## Phase 2 — Standalone substrate ✅ (landed)

Goal: the package's substrate (settings, base class, protocols, registry, db pool) loads its own implementation instead of delegating to atlas_brain when `EXTRACTED_LLM_INFRA_STANDALONE=1` is set.

| Task | Status |
|---|---|
| Carve `LLMInfraSettings` (LLMSubConfig + B2BChurnSubConfig + ReasoningSubConfig + FTLTracingSubConfig + ModelPricingConfig) | ✅ `_standalone/config.py` |
| Standalone Message / ModelInfo / InferenceMetrics / LLMService Protocol | ✅ `_standalone/protocols.py` |
| Torch-free `BaseModelService` ABC | ✅ `_standalone/base.py` |
| `ServiceRegistry` + `llm_registry` singleton + `@register_llm` | ✅ `_standalone/registry.py` |
| Slim `DatabasePool` wrapper around asyncpg | ✅ `_standalone/database.py` |
| Bridge stubs gate on `EXTRACTED_LLM_INFRA_STANDALONE=1` | ✅ five bridges updated |
| Standalone smoke script + CI integration | ✅ `scripts/smoke_extracted_llm_infrastructure_standalone.py` |
| README documents the toggle and env-var layout | ✅ |

**Empirical result**: the standalone substrate is sufficient to unblock the
import contract for all 14 provider modules. They consume the substrate
transitively through the bridge stubs, so when
`EXTRACTED_LLM_INFRA_STANDALONE=1` is set, every provider sees the local
`_standalone/*` copies of `BaseModelService`, `LLMService` Protocol, `Message`,
`ModelInfo`, `ServiceRegistry`, `llm_registry`, `settings`, and `DatabasePool`.

The standalone smoke (`scripts/smoke_extracted_llm_infrastructure_standalone.py`) verifies this end-to-end: it sets the env var, imports every provider, and asserts (via `__module__` walk on `AnthropicLLM.__mro__`) that providers transitively consume the standalone substrate rather than silently falling back to atlas_brain.

## Phase 3 — Runtime decoupling 🟡 (in progress)

Import contract is closed; the remaining work is **runtime** behavior when functions execute, not when modules load:

| Task | Source file referenced | Status |
|---|---|---|
| `Protocol`-based DI for LLM instances; replace `isinstance(AnthropicLLM)` checks | `services/b2b/anthropic_batch.py:550, 740, 1034` (3 internal guards) + 6 dispatch sites in autonomous tasks (`b2b_blog_post_generation.py`, `b2b_product_profiles.py`, `b2b_enrichment_repair.py`, `enrichment_row_runner.py`) | ✅ PR-A5d (added `runtime_checkable` `AnthropicBatchableLLM` Protocol with `name`/`model`/`_async_client` surface; swapped 9 isinstance sites; private function type hints in `anthropic_batch.py` switched to the Protocol; companion `getattr(_async_client, None)` check preserved at the 3 internal guard sites) |
| Extract `_convert_messages` from `AnthropicLLM` so batch code does not call a private method | `services/llm/anthropic.py:103+`, called from `services/b2b/anthropic_batch.py:409` | ✅ PR-A5a (module-level `convert_messages`; method preserved as backwards-compat alias; batch caller imports the public function) |
| Decouple `SemanticCache` from Postgres (asyncpg.Record assumptions) | `reasoning/semantic_cache.py:70-339` | ✅ PR-A5b (added `SemanticCachePool` Protocol; `__init__` typed against it; `_row_to_entry` annotated as `Mapping[str, Any]`; SQL stays Postgres-specific) |
| Move `evidence_hash` computation to a single owner | currently split between `reasoning/semantic_cache.py:47` and B2B callers | ✅ PR-A5c (`compute_cross_vendor_evidence_hash` in `_b2b_cross_vendor_synthesis.py` is now a re-export of `compute_evidence_hash` from `semantic_cache.py`; both import paths preserved; one canonical implementation) |
| Open-source-grade README + LICENSE + pyproject.toml | scaffold root | 🔲 |
| Publishable PyPI package | scaffold root | 🔲 |

## Per-file extraction state

| Scaffold file | Phase 1 (snapshot) | Phase 2 (substrate) | Phase 3 (decoupled) |
|---|---|---|---|
| `_standalone/config.py` (new) | n/a | ✅ standalone settings | 🔲 |
| `_standalone/protocols.py` (new) | n/a | ✅ standalone protocols | 🔲 |
| `_standalone/base.py` (new) | n/a | ✅ torch-free BaseModelService | 🔲 |
| `_standalone/registry.py` (new) | n/a | ✅ standalone ServiceRegistry | 🔲 |
| `_standalone/database.py` (new) | n/a | ✅ slim DatabasePool | 🔲 |
| `config.py` (bridge) | n/a | ✅ env-gated dispatch | n/a |
| `services/base.py` (bridge) | n/a | ✅ env-gated dispatch | n/a |
| `services/protocols.py` (bridge) | n/a | ✅ env-gated dispatch | n/a |
| `services/registry.py` (bridge) | n/a | ✅ env-gated dispatch | n/a |
| `storage/database.py` (bridge) | n/a | ✅ env-gated dispatch | n/a |
| `services/b2b/anthropic_batch.py` | ✅ | ✅ (imports cleanly; consumes standalone substrate transitively) | 🔲 |
| `services/b2b/cache_strategy.py` | ✅ | ✅ (pure data; no atlas imports) | n/a |
| `services/b2b/llm_exact_cache.py` (PR-A1) | ✅ | ✅ (lazy `from ...config import settings` + `from ...storage.database import get_db_pool` route via env-gated bridges; new `from ...skills import get_skill_registry` routes via PR-A1.5 skills bridge with explicit standalone-mode error; `B2BChurnSubConfig.llm_exact_cache_enabled` flag added in PR-A1.5) | 🔲 (Phase 3: substrate skills layer or replace skill helpers with Protocol-based DI) |
| `skills/__init__.py` (bridge, PR-A1.5) | n/a | ✅ env-gated dispatch; raises NotImplementedError in standalone mode | 🔲 |
| `pipelines/llm.py` | ✅ | ✅ (lazy `from ..config import settings` routes to standalone) | 🔲 |
| `reasoning/semantic_cache.py` | ✅ | ✅ (pool injected by caller; standalone DatabasePool compatible) | 🔲 |
| `services/llm_router.py` | ✅ | ✅ (consumes standalone settings + registry) | 🔲 |
| `services/llm/anthropic.py` | ✅ | ✅ (transitive substrate verified by smoke check) | 🔲 |
| `services/llm/openrouter.py` | ✅ | ✅ | 🔲 |
| `services/llm/ollama.py` | ✅ | ✅ | 🔲 |
| `services/llm/vllm.py` | ✅ | ✅ | 🔲 |
| `services/llm/groq.py` | ✅ | ✅ | 🔲 |
| `services/llm/together.py` | ✅ | ✅ | 🔲 |
| `services/llm/hybrid.py` | ✅ | ✅ | 🔲 |
| `services/llm/cloud.py` | ✅ | ✅ | 🔲 |
| `services/tracing.py` | ✅ | ✅ | 🔲 |
| `services/provider_cost_sync.py` (PR-A2) | ✅ | 🔲 (Phase 2 follow-up: standalone substrate for provider settings + db pool wrapper for snapshot/daily-cost upserts; lift uses default Atlas mode for now) | 🔲 |
| `services/cost/__init__.py` (OWNED, PR-A3) | n/a | ✅ (no atlas imports; owned by extraction) | n/a |
| `services/cost/cache_savings.py` (OWNED, PR-A3) | n/a | ✅ (asyncpg-pool-shaped; runs standalone with the local DatabasePool) | n/a |
| `services/cost/openai_billing.py` (OWNED, PR-A4c) | n/a | ✅ (httpx + asyncpg-pool-shaped; settings.provider_cost lookup with env fallback) | 🔲 (Phase 3: unify with provider_cost_sync via ProviderBillingPort Protocol) |
| `services/cost/drift.py` (OWNED, PR-A4a) | n/a | ✅ (asyncpg-pool-shaped; pure SQL + dataclass output, no atlas imports) | n/a |
| `services/cost/budget.py` (OWNED, PR-A4b) | n/a | ✅ (asyncpg-pool-shaped; reads llm_usage; fail-open on bad input) | n/a |
| `storage/migrations/127_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/130_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/251_*.sql` (PR-A1) | ✅ | n/a | n/a |
| `storage/migrations/252_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/253_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/255_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/257_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/258_*.sql` (PR-A2) | ✅ | n/a | n/a |
| `storage/migrations/259_*.sql` (OWNED, PR-A3) | n/a | n/a | n/a |
