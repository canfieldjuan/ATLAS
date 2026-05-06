# Reasoning Provider Port Migration Guide

This guide captures the current migration slice from direct file-provider loading to the provider-port loader wrapper.

## Scope of this migration slice

In scope:
- Keep existing behavior.
- Move host entrypoints to a port-compatible loader name.
- Keep `FileCampaignReasoningContextProvider` as the reference file adapter.

Out of scope:
- Rewriting campaign generation internals.
- Changing `CampaignGenerationService` behavior.
- Altering reasoning payload shape.

## Old -> new mapping

| Previous usage | Current usage | Notes |
|---|---|---|
| `load_campaign_reasoning_context_provider(path)` | `load_reasoning_provider_port(path)` | New wrapper is provider-port aligned. |
| Direct mention of file adapter in host scripts | Port-compatible loader in host scripts | File adapter still used under the hood. |
| Provider accepted as `CampaignReasoningContextProvider` only | Provider accepted as `CampaignReasoningContextProvider | CampaignReasoningProviderPort` | Additive typing; behavior unchanged. |

## Current implementation state

- Loader wrapper added in `extracted_content_pipeline/campaign_reasoning_data.py`.
- Port protocol added in `extracted_content_pipeline/services/reasoning_provider_port.py`.
- Host script entrypoints switched:
  - `scripts/run_extracted_campaign_generation_example.py`
  - `scripts/run_extracted_campaign_generation_postgres.py`
- Generation entrypoints widened to accept both protocol types:
  - `extracted_content_pipeline/campaign_example.py`
  - `extracted_content_pipeline/campaign_postgres_generation.py`

## Host upgrade checklist

1. If host code calls `load_campaign_reasoning_context_provider(...)` directly for CLI wiring, switch to `load_reasoning_provider_port(...)`.
2. Keep reasoning JSON shape unchanged.
3. No changes required to campaign generation invocation payloads.
4. Re-run host smoke flow for:
   - example runner with `--reasoning-context`
   - postgres runner with `--reasoning-context`

## Compatibility guarantees in this slice

1. Existing reasoning JSON files continue to work.
2. Existing file-backed provider behavior is unchanged.
3. Existing campaign prompt metadata keys are unchanged.
4. Existing callers passing a `CampaignReasoningContextProvider` instance continue to work.

## Verification commands

```bash
python -m py_compile \
  extracted_content_pipeline/campaign_reasoning_data.py \
  extracted_content_pipeline/services/reasoning_provider_port.py \
  scripts/run_extracted_campaign_generation_example.py \
  scripts/run_extracted_campaign_generation_postgres.py
```
