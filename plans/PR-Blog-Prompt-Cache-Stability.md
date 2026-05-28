# PR-Blog-Prompt-Cache-Stability

## Why this slice exists

The live support-ticket SaaS demo retry proved provider prompt caching is
hitting (`cached_tokens=9814`, `cache_write_tokens=17434`), but the blog prompt
builder still allows stored skill templates to place per-run `{topic}` and
`{blueprint_json}` values in the system prompt. That can silently collapse the
cache for any template that uses those placeholders in the static prompt block.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Production hardening

1. Keep the blog generation system prompt static by replacing dynamic
   placeholders with stable handoff text.
2. Put the actual blueprint JSON and operator topic in the user message for
   initial generation and quality repair.
3. Update tests so support-ticket descriptive contracts, stale-contract cleanup,
   per-call topic handling, and repair prompts prove the dynamic values stay out
   of the system prompt.

### Files touched

- `extracted_content_pipeline/blog_generation.py` - move dynamic blog prompt values to the user message.
- `tests/test_extracted_blog_generation.py` - update and add prompt-shape regression coverage.
- `tests/test_extracted_content_ops_live_execute_harness.py` - keep execute harness assertions aligned with the static system prompt contract.
- `plans/PR-Blog-Prompt-Cache-Stability.md` - plan for this slice.

## Mechanism

The blog prompt helper now treats the skill prompt as the cacheable system
instruction block. It replaces `{blueprint_json}` and `{topic}` with stable
phrases that point to the user message, then builds a user prompt that always
contains the serialized blueprint JSON and, when supplied, the operator topic.
The support-ticket descriptive addendum remains attached to the user prompt so
the large static denylist stays outside the per-run blueprint JSON while still
travelling with the generation request.

## Intentional

- The stored markdown prompt still contains `{topic}` for operator-readable
  placement, but the runtime no longer substitutes the per-run topic into the
  system prompt.
- This does not change the LLM provider, cache-control machinery, or cost
  storage schema. The live result already proves cache metrics are being
  surfaced in generation output; this slice hardens the prompt shape that makes
  those cache hits repeatable.

## Deferred

- `HARDENING.md` still carries `LLM usage storage schema mismatch hides per-run cost telemetry`; that is a separate local schema/cost-surfacing slice, not required to keep the provider prompt cache stable.
- Parked hardening: none

## Verification

- pytest tests/test_extracted_blog_generation.py -q -> 71 passed.
- pytest tests/test_extracted_content_ops_live_execute_harness.py::test_support_ticket_provider_feeds_real_blog_post_generation -q -> 1 passed.
- pytest tests/test_extracted_blog_generation.py tests/test_extracted_content_ops_live_execute_harness.py::test_support_ticket_provider_feeds_real_blog_post_generation -q -> 72 passed.
- python -m py_compile extracted_content_pipeline/blog_generation.py tests/test_extracted_blog_generation.py tests/test_extracted_content_ops_live_execute_harness.py -> passed.
- bash scripts/validate_extracted_content_pipeline.sh -> passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -> passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -> passed.
- bash scripts/check_ascii_python.sh -> passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-blog-prompt-cache-stability-body.md -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `extracted_content_pipeline/blog_generation.py` | ~26 |
| `tests/test_extracted_blog_generation.py` | ~69 |
| `tests/test_extracted_content_ops_live_execute_harness.py` | ~6 |
| `plans/PR-Blog-Prompt-Cache-Stability.md` | ~74 |
| Total | ~175 |
