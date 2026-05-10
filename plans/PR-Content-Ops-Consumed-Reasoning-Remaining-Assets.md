# PR-Content-Ops-Consumed-Reasoning-Remaining-Assets

## Why this slice exists

PR #467 shipped the shared consumed-reasoning helper and adopted it in
campaign and blog generation. The first full-service attempt was too large
for the review-size gate, so this follow-up closes the remaining generated
asset services without touching the already-merged campaign/blog changes.

## Scope (this PR)

1. Adopt consumed reasoning payloads in report generation.
2. Adopt consumed reasoning payloads in landing page generation.
3. Adopt consumed reasoning payloads in sales brief generation.
4. Tighten the offline Content Ops execution smoke so `--with-reasoning`
   validates both `reasoning_contexts_used` and the bounded consumed payloads.
5. Update status/frontend docs to say all LLM-backed generated assets now
   expose consumed reasoning payloads when reasoning reaches the prompt.

### Files touched

- `extracted_content_pipeline/report_generation.py`
- `extracted_content_pipeline/landing_page_generation.py`
- `extracted_content_pipeline/sales_brief_generation.py`
- `scripts/smoke_extracted_content_ops_execution.py`
- `tests/test_extracted_report_generation.py`
- `tests/test_extracted_landing_page_generation.py`
- `tests/test_extracted_sales_brief_generation.py`
- `tests/test_extracted_content_ops_execution_smoke.py`
- `docs/frontend/content_ops_frontend_contract.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The three remaining generators use
`consumed_campaign_reasoning_contexts(payload)` at the same point they
already increment `reasoning_contexts_used`: after a prompt-visible
reasoning context helped produce an accepted draft.

The result `as_dict()` methods keep the existing response shape when no
reasoning was consumed. They only include `consumed_reasoning_contexts`
when at least one bounded payload exists.

The execution smoke uses deterministic fake services and now requires
`reasoning.consumed_contexts` to exist whenever a step reports a positive
reasoning usage count.

## Intentional

- No raw provider rows are exposed.
- No public method signatures change.
- `reasoning_contexts_used` semantics stay unchanged.
- Campaign/blog code remains untouched in this split because PR #467 already
  owns that adoption.
- The Reasoning Context Drawer UI remains a frontend follow-up; this PR only
  guarantees the backend contract is populated.

## Verification

- Focused generator/smoke tests.
- `python -m py_compile` on edited Python files.
- Full extracted pipeline check.
- `git diff --check`.
- ASCII byte check on edited Python files.

## Estimated diff size

12 files, roughly +145 / -20. Under the 400 LOC soft review budget.
