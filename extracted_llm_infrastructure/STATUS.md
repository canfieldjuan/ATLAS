# extracted_llm_infrastructure — STATUS

## Phase 1 — Scaffold creation ✅

| Step | Status |
|---|---|
| Manifest of source → scaffold mappings | ✅ done |
| Verbatim byte-snapshot of 14 Python files | ✅ done |
| Verbatim byte-snapshot of 6 migration SQL files | ✅ done |
| Package `__init__.py` files at every level | ✅ done |
| Sync + validate scripts | ✅ done |
| ASCII / smoke-import / import-debt checks | ✅ done |
| Driver script `run_extracted_llm_infrastructure_checks.sh` | ✅ done |
| GitHub Actions workflow | ✅ done |
| README + this STATUS file | ✅ done |
| `import_debt_allowlist.txt` documenting atlas_brain dependencies | ✅ done |

## Phase 2 — Standalone toggle 🔲 (separate PR)

Goal: every scaffolded module is importable and runnable without `atlas_brain` on `sys.path`, gated by `EXTRACTED_LLM_INFRA_STANDALONE=1`.

| Task | Notes |
|---|---|
| Carve a slim `LLMInfraConfig` Pydantic class out of `atlas_brain/config.py` | Mix-in fields from `LLMConfig`, `B2BChurnConfig.openrouter_*`, `B2BChurnConfig.anthropic_batch_*`, `FTLTracingConfig.*`, `ModelPricingConfig.*` |
| Local DB pool abstraction | `LLMInfraStorage` Protocol + asyncpg adapter for standalone, atlas adapter for delegate |
| No-op tracer fallback | `LLMInfraTracer` Protocol with `start_span`/`end_span`; standalone defaults to no-op |
| Standalone migration runner | Apply the six SQL files under `storage/migrations/` to a fresh Postgres |
| Per-file extraction state column (below) | Track which files are still "delegate-only" vs "standalone-ready" |

## Phase 3 — Decoupling 🔲 (later PRs)

| Task | Source file referenced |
|---|---|
| `Protocol`-based DI for LLM instances; replace `isinstance(AnthropicLLM)` checks | `services/b2b/anthropic_batch.py:550, 740, 1034, 1144` and `services/llm_router.py:205-247` |
| Extract `_convert_messages` from `AnthropicLLM` so batch code does not call a private method | `services/llm/anthropic.py:103+`, called from `services/b2b/anthropic_batch.py:409` |
| Decouple `SemanticCache` from Postgres (asyncpg.Record assumptions) | `reasoning/semantic_cache.py:70-339` |
| Move `evidence_hash` computation to a single owner | currently split between `reasoning/semantic_cache.py:47` and B2B callers |
| Open-source-grade README + LICENSE + pyproject.toml | scaffold root |
| Publishable PyPI package | scaffold root |

## Per-file extraction state

| Scaffold file | Phase 1 (snapshot) | Phase 2 (standalone-ready) | Phase 3 (decoupled) |
|---|---|---|---|
| `services/b2b/anthropic_batch.py` | ✅ | 🔲 | 🔲 |
| `services/b2b/cache_strategy.py` | ✅ | 🔲 (pure data; trivial) | 🔲 |
| `pipelines/llm.py` | ✅ | 🔲 | 🔲 |
| `reasoning/semantic_cache.py` | ✅ | 🔲 | 🔲 |
| `services/llm_router.py` | ✅ | 🔲 | 🔲 |
| `services/llm/anthropic.py` | ✅ | 🔲 | 🔲 |
| `services/llm/openrouter.py` | ✅ | 🔲 | 🔲 |
| `services/llm/ollama.py` | ✅ | 🔲 | 🔲 |
| `services/llm/vllm.py` | ✅ | 🔲 | 🔲 |
| `services/llm/groq.py` | ✅ | 🔲 | 🔲 |
| `services/llm/together.py` | ✅ | 🔲 | 🔲 |
| `services/llm/hybrid.py` | ✅ | 🔲 | 🔲 |
| `services/llm/cloud.py` | ✅ | 🔲 | 🔲 |
| `services/tracing.py` | ✅ | 🔲 | 🔲 |
| `storage/migrations/127_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/130_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/252_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/253_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/255_*.sql` | ✅ | n/a | n/a |
| `storage/migrations/257_*.sql` | ✅ | n/a | n/a |
