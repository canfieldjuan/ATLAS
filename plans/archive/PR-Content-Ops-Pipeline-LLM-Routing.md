# PR: Content Ops Pipeline LLM Routing

## Why this slice exists

AI Content Ops generation services are wired to the Atlas DB pool and packaged
skills, but the LLM factory only checks `llm_registry.get_active()`. Other Atlas
generation paths resolve provider-backed Claude/OpenRouter models through
`atlas_brain.pipelines.llm.get_pipeline_llm(...)`, so Content Ops can look
unwired even when the configured OpenRouter Claude route is available through
pipeline routing.

This slice fixes the source wiring so Content Ops uses the same provider
resolution path as the rest of Atlas generation instead of requiring a
pre-activated global registry slot.

## Scope (this PR)

Ownership lane: content-ops/pipeline-llm-routing

1. Make `build_content_ops_llm_client()` resolve a pipeline-routed OpenRouter
   LLM before falling back to the active registry slot.
2. Preserve the existing test injection seam so unit tests do not import the
   heavy provider stack.
3. Keep DB, skill, route, and generation-service wiring unchanged.
4. Add regression coverage proving pipeline-routed LLMs wire Content Ops even
   when no active registry LLM exists.
5. Enroll the host Content Ops wiring tests in extracted-pipeline CI so the
   routing path has a pull-request regression guard.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Pipeline-LLM-Routing.md` | Plan doc for this source wiring fix. |
| `atlas_brain/_content_ops_infrastructure.py` | Resolve the Content Ops LLM through pipeline routing before registry fallback. |
| `tests/test_atlas_content_ops_infrastructure.py` | Pin pipeline resolver behavior and registry fallback. |
| `scripts/run_extracted_pipeline_checks.sh` | Run the host Content Ops LLM and execution-service wiring tests in extracted-pipeline CI. |
| `.github/workflows/extracted_pipeline_checks.yml` | Trigger extracted-pipeline CI when the host Content Ops LLM wiring or its tests change. |

## Mechanism

Production `build_content_ops_llm_client()` imports `get_pipeline_llm` lazily
and calls it with the OpenRouter-only workload:

```python
get_pipeline_llm(
    workload="openrouter",
    try_openrouter=True,
    auto_activate_ollama=False,
)
```

If that returns an LLM, Content Ops wraps it in the existing `_HostLLMClient`.
If the pipeline route is unavailable, the factory falls back to
`llm_registry.get_active()` to preserve the previous behavior for deployments
that intentionally activate a global provider.

## Intentional

- No database changes. DB-backed Content Ops services already use
  `get_db_pool()` and the Postgres repositories.
- No route changes. `/api/v1/content-ops/execute` already receives
  `build_content_ops_execution_services(enable_db_services=True)`.
- No live provider call in tests. The resolver is dependency-injected in unit
  tests so CI does not require OpenRouter credentials.
- No Ollama auto-activation for this Content Ops route. If OpenRouter is not
  configured, the old active-registry fallback remains the only fallback.

## Deferred

- Parked hardening: none. Root `HARDENING.md` has no matching items.
- A live end-to-end generation run with real OpenRouter credentials remains an
  operator acceptance check after this wiring lands.

## Verification

- `pytest tests/test_atlas_content_ops_infrastructure.py tests/test_atlas_content_ops_execution_services.py -q` -> 26 passed.
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1807 passed, 1 skipped.
- Programmatic wiring diagnostic with a fake pipeline-routed LLM -> Content Ops
  bundle advertised `landing_page` and all LLM-backed outputs.
- `git diff --check` -> passed.
- Local PR review wrapper -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| LLM factory | ~35 |
| Tests | ~50 |
| CI enrollment | ~10 |
| **Total** | **~170** |
