# PR-Content-Ops-Consumed-Reasoning-Campaign-Blog

## Why this slice exists

PR #451 added the execution response contract for
`reasoning.consumed_contexts`, but the generated-asset services still
mostly return only `reasoning_contexts_used`. The first attempt to close
all services in one PR crossed the review-size gate, so this smaller
slice ships the shared helper and two generator adoptions first.

## Scope (this PR)

1. Add a shared helper that extracts bounded prompt-visible consumed
   reasoning payloads from an enriched generation payload.
2. Adopt that helper in campaign generation.
3. Adopt that helper in blog post generation.
4. Keep `reasoning_contexts_used`, prompt construction, persistence
   metadata, provider lookup, and public method signatures unchanged.

### Files touched

- `extracted_content_pipeline/services/campaign_reasoning_context.py`
- `extracted_content_pipeline/campaign_generation.py`
- `extracted_content_pipeline/blog_generation.py`
- `tests/test_extracted_campaign_generation.py`
- `tests/test_extracted_blog_generation.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Consumed-Reasoning-Campaign-Blog.md`

## Mechanism

`consumed_campaign_reasoning_contexts(payload)` extracts only the
prompt-visible `campaign_reasoning_context` sibling, or the same key
nested under `reasoning_context`. It normalizes through the existing
bounded `CampaignReasoningContext` contract and returns a tuple of
drawer-ready dictionaries.

Campaign and blog generation append those dictionaries at the same
point they already increment `reasoning_contexts_used`: after parse and
quality gates pass, immediately before a draft is accepted.

## Intentional

- No raw provider rows are exposed.
- No behavior changes when reasoning is absent; `as_dict()` omits
  `consumed_reasoning_contexts` when empty.
- Campaign follow-up channels may report the same context more than
  once because existing usage is per generated draft/channel.
- Report, landing page, sales brief, smoke, and docs move to the next
  PR to keep this under the review-size gate.

## Deferred

- Report, landing page, and sales brief service adoption.
- Offline execution smoke validation for `reasoning.consumed_contexts`.
- Frontend Reasoning Context Drawer rendering.

## Verification

- `pytest tests/test_extracted_campaign_generation.py tests/test_extracted_blog_generation.py` -> 59 passed
- `python -m py_compile extracted_content_pipeline/services/campaign_reasoning_context.py extracted_content_pipeline/campaign_generation.py extracted_content_pipeline/blog_generation.py` -> passed
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1432 passed, 1 existing torch/pynvml warning
- `git diff --check` -> passed
- ASCII byte check on edited Python files -> passed

## Estimated diff size

7 files, roughly +145 / -7. Under the 400 LOC soft review budget.
