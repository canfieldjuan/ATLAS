# PR-Content-Ops-Consumed-Reasoning-Service-Parity

## Why this slice exists

PR #451 added the execution response contract for
`reasoning.consumed_contexts`, but the generated-asset services still
mostly return only `reasoning_contexts_used`. That lets the control
surface say "reasoning was used" without exposing the bounded context
payload a frontend drawer can inspect.

This slice closes that service-level adoption gap for the five
LLM-backed AI Content Ops outputs.

## Scope (this PR)

1. Add optional `consumed_reasoning_contexts` result fields to campaign,
   blog post, report, landing page, and sales brief generation results.
2. Populate the field from the already prompt-visible
   `campaign_reasoning_context` payload only when a draft is generated.
3. Keep existing `reasoning_contexts_used`, prompt construction,
   persistence metadata, provider lookup, and public method signatures
   unchanged.
4. Tighten the offline execution smoke so `--with-reasoning` validates
   `reasoning.consumed_contexts`, not just the count.

### Files touched

- `extracted_content_pipeline/services/campaign_reasoning_context.py`
- `extracted_content_pipeline/campaign_generation.py`
- `extracted_content_pipeline/blog_generation.py`
- `extracted_content_pipeline/report_generation.py`
- `extracted_content_pipeline/landing_page_generation.py`
- `extracted_content_pipeline/sales_brief_generation.py`
- `scripts/smoke_extracted_content_ops_execution.py`
- `tests/test_extracted_campaign_generation.py`
- `tests/test_extracted_blog_generation.py`
- `tests/test_extracted_report_generation.py`
- `tests/test_extracted_landing_page_generation.py`
- `tests/test_extracted_sales_brief_generation.py`
- `tests/test_extracted_content_ops_execution_smoke.py`
- `docs/frontend/content_ops_frontend_contract.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

`consumed_campaign_reasoning_contexts(payload)` extracts only the
prompt-visible `campaign_reasoning_context` sibling (or the same key
nested under `reasoning_context`), normalizes it through the existing
bounded `CampaignReasoningContext` contract, and returns a tuple of
drawer-ready dictionaries.

Each generator appends those dictionaries at the same point it already
increments `reasoning_contexts_used`: after parse and quality gates
pass, immediately before the draft is accepted. Failed, skipped, or
blocked rows do not contribute consumed context.

## Intentional

- No raw provider rows are exposed. The result uses the same bounded
  prompt payload the LLM saw.
- No behavior changes when reasoning is absent; `as_dict()` omits
  `consumed_reasoning_contexts` when empty.
- Campaign follow-up channels may report the same context more than
  once because the existing `reasoning_contexts_used` count is per
  generated draft/channel, not per unique target.
- No changes to provider resolution or DB/file-backed host wiring.

## Deferred

- Frontend Reasoning Context Drawer rendering.
- Richer attribution display for consumed contexts.
- Deduped consumed-context summaries if the UI later wants
  target-level grouping instead of per-generated-asset grouping.

## Verification

- `pytest tests/test_extracted_campaign_generation.py tests/test_extracted_blog_generation.py tests/test_extracted_report_generation.py tests/test_extracted_landing_page_generation.py tests/test_extracted_sales_brief_generation.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_ops_execution_smoke.py` -> 176 passed
- `python -m py_compile extracted_content_pipeline/campaign_generation.py extracted_content_pipeline/blog_generation.py extracted_content_pipeline/report_generation.py extracted_content_pipeline/landing_page_generation.py extracted_content_pipeline/sales_brief_generation.py extracted_content_pipeline/services/campaign_reasoning_context.py scripts/smoke_extracted_content_ops_execution.py` -> passed
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1433 passed, 1 existing torch/pynvml warning
- `git diff --check` -> passed
- ASCII byte check on edited Python files -> passed

## Estimated diff size

17 files, roughly +250 / -25. Under the 400 LOC soft review budget.
