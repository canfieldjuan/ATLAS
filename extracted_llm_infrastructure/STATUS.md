# extracted_llm_infrastructure — STATUS

## Phase 1 — Scaffold creation ✅

| Step | Status |
|---|---|
| Manifest of source → scaffold mappings | ✅ done |
| Verbatim byte-snapshot of 14 Python files | ✅ done |
| Verbatim byte-snapshot of 6 migration SQL files | ✅ done |
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

## Phase 3 — Runtime decoupling 🔲 (later PRs)

Import contract is closed; the remaining work is **runtime** behavior when functions execute, not when modules load:

| Task | Source file referenced |
|---|---|
| `Protocol`-based DI for LLM instances; replace `isinstance(AnthropicLLM)` checks | `services/b2b/anthropic_batch.py:550, 740, 1034, 1144` and `services/llm_router.py:205-247` |
| Extract `_convert_messages` from `AnthropicLLM` so batch code does not call a private method | `services/llm/anthropic.py:103+`, called from `services/b2b/anthropic_batch.py:409` |
| Decouple `SemanticCache` from Postgres (asyncpg.Record assumptions) | `reasoning/semantic_cache.py:70-339` |
| Move `evidence_hash` computation to a single owner | currently split between `reasoning/semantic_cache.py:47` and B2B callers |
| Open-source-grade README + LICENSE + pyproject.toml | scaffold root |
| Publishable PyPI package | scaffold root |

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
| `storage/migrations/127_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/130_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/252_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/253_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/255_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/257_*.sql` | ✅ | n/a | n/a |
