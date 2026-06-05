# PR-LLM-Registry-Service-Export

## Why this slice exists

Live provider generation for imported review-source rows fails before saving
provider-backed drafts because two adapter seams are incomplete:

1. `extracted_llm_infrastructure.pipelines.llm` imports `llm_registry` from
   `extracted_llm_infrastructure.services`, while the package namespace does
   not re-export the registry bridge.
2. After the registry seam is fixed, the OpenRouter backend receives extracted
   `LLMMessage` objects that lack optional host-backend chat fields such as
   `tool_calls`.

Both failures happen before useful generation output. This slice exposes the
existing registry bridge and normalizes extracted chat messages into the
duck-typed shape expected by live LLM backends.

## Scope (this PR)

1. Re-export `llm_registry` and `register_llm` from
   `extracted_llm_infrastructure.services`.
2. Normalize `PipelineLLMClient` chat messages before provider `.chat()` calls.
3. Add a regression test that the standalone pipeline resolver can traverse the
   package-level registry import without Atlas.
4. Add a regression test that chat-path messages include optional tool fields.

### Files touched

- `extracted_llm_infrastructure/services/__init__.py`
- `extracted_content_pipeline/campaign_llm_client.py`
- `tests/test_extracted_llm_infrastructure_registry_export.py`
- `tests/test_extracted_campaign_llm_client.py`
- `plans/PR-LLM-Registry-Service-Export.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

`services.registry` already selects the Atlas-backed registry in repo mode and
the local `_standalone.registry` implementation when
`EXTRACTED_LLM_INFRA_STANDALONE=1`. The package `__init__` now imports and
re-exports those two public names so existing package-level imports work:

```python
from .registry import llm_registry, register_llm
```

The regression test runs the `local_fast` resolver path in standalone mode with
Ollama auto-activation disabled. That path exists only to return the active
registry value, so the test catches the prior import failure without making any
network or provider call.

`PipelineLLMClient` now wraps extracted `LLMMessage` values in
`SimpleNamespace(role, content, tool_calls, tool_call_id)` before calling
provider `.chat()`. This mirrors the Atlas host bridge and keeps generated
prompt fallback behavior unchanged.

## Intentional

- This PR does not change provider routing or activation behavior. It only
  exposes the registry bridge at the namespace the resolver already uses.
- This PR does not add tool-call support to extracted campaign generation. It
  only supplies `None` defaults for optional fields that existing backends read.
- The test disables local model activation so it remains a pure import/runtime
  seam check.

## Deferred

- None. After merge, rerun live provider generation over the imported
  review-source rows.

## Verification

To run before push:

```bash
pytest tests/test_extracted_llm_infrastructure_registry_export.py
pytest tests/test_extracted_campaign_llm_client.py tests/test_extracted_llm_infrastructure_registry_export.py
python -m py_compile extracted_llm_infrastructure/services/__init__.py extracted_content_pipeline/campaign_llm_client.py tests/test_extracted_llm_infrastructure_registry_export.py tests/test_extracted_campaign_llm_client.py
bash scripts/local_pr_review.sh
```

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Namespace export | ~10 |
| Chat message normalization | ~15 |
| Regression tests | ~75 |
| Plan + coordination | ~110 |
| **Total** | **~210** |
