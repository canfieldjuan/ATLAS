# Support Ticket Blog Generated Content Gate

## Why this slice exists

PR #958 proved the support-ticket generated-content evaluator with a live
Haiku run, and it correctly caught a blog draft that invented unsupported
future-impact percentages. That evaluator still only runs from smoke/export
tooling, so the product path can save a support-ticket-backed blog draft that
the evaluator would reject.

This slice promotes the evaluator into the blog save-time quality path for
support-ticket blueprints. It keeps the FAQ boundary intact: blog generation
consumes normalized support-ticket evidence from the input package and
generation `data_context`, not the FAQ generator's current output shape.

The diff is over the normal budget because the existing evaluator moves from a
script-owned implementation into an importable package module. Most of that
diff is relocation, not new behavior.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Vertical slice

1. Move the generated-content evaluator behind an importable
   `extracted_content_pipeline` module while keeping the script CLI as a thin
   wrapper.
2. Thread support-ticket input-package context through the blog execute path
   and run the evaluator inside `BlogPostGenerationService._quality_check`
   only when the merged generation context is support-ticket-backed.
3. Feed evaluator failures into the existing blog quality repair loop and
   block saving if the repaired draft still fails.
4. Add focused tests for support-ticket blocking, repair feedback, persistent
   repair failure, and the non-support-ticket no-op path.

### Files touched

- `plans/PR-Support-Ticket-Blog-Generated-Content-Gate.md`
- `extracted_content_pipeline/blog_generation.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/support_ticket_generated_content_eval.py`
- `scripts/evaluate_support_ticket_generated_content.py`
- `tests/test_extracted_content_ops_live_execute_harness.py`
- `tests/test_extracted_blog_generation.py`
- `tests/test_support_ticket_provider_landing_blog_execute.py`

## Mechanism

The evaluator remains deterministic and pure. The script entry point imports
the package module instead of owning the implementation.

The content-ops executor derives trusted blog `data_context` from the
support-ticket input package when `filters.topic_type` is the canonical
`content_ops_support_ticket_faq` discriminator. It passes that context into
`BlogPostGenerationService.generate`, which merges it onto the blueprint before
the prompt and quality gate run.

Blog generation then builds an in-memory blog export row from the parsed draft
plus the merged trusted/generated `data_context`, then calls
`evaluate_support_ticket_generated_content(..., output="blog_post")`.

Only support-ticket contexts trigger the extra check. Runtime support-ticket
context comes from the canonical input-package topic type; the lower-level
service predicate still recognizes provider/source, period, count, and cluster
markers so stored support-ticket blueprints remain gated too. Evaluator errors
become quality blockers with a stable
`support_ticket_generated_content:` prefix, so the existing repair prompt can
tell the model exactly what to fix.

The repair loop and save semantics stay unchanged: if all gates pass after the
configured repair budget, the draft saves; otherwise the blueprint is skipped
with `reason: quality_blocked`.

## Intentional

- No FAQ generator changes. This slice does not consume generated FAQ article
  output, FAQ Markdown, or FAQ answer schemas.
- No live LLM run in this PR. PR #958 already recorded the live Haiku failure;
  this slice wires the deterministic gate into the product path and proves it
  with service-level plus execute-route tests.
- Landing-page generation is left unchanged because its support-ticket path
  already passed the live generated-content evaluation in #958. This slice is
  specifically for the blog save-time gap exposed by the failed live blog run.
- Existing blog SEO/AEO/GEO quality checks remain load-bearing. The
  support-ticket evaluator composes on top of them rather than replacing them.
- Cross-layer caller hints were inspected. The `BlogPostGenerationService`
  constructor call sites keep the same signature, the new `generate`
  `data_context` kwarg is optional, and the quality-check symbol references in
  other generators are same-name methods, not callers of this blog
  implementation. The new `_positive_int` helper shares a name with an
  unrelated script-local helper only.

## Deferred

- Future PR: rerun the live support-ticket blog smoke with Haiku after this
  gate lands, using the same evaluator as the product save path.
- Future PR: add a versioned FAQ evidence projection only when the FAQ session
  needs blog/landing injection. Blog/landing generators should consume a stable
  evidence projection such as `faq_evidence_context` or
  `faq_answer_snippets`, not the mutable FAQ output shape.
- Parked hardening: none added by this slice.

## Verification

- Targeted blog-generation pytest over `tests/test_extracted_blog_generation.py`
  - 55 passed.
- Targeted evaluator pytest over
  `tests/test_evaluate_support_ticket_generated_content.py`
  - 18 passed.
- Py compile for `extracted_content_pipeline/blog_generation.py`,
  `extracted_content_pipeline/support_ticket_generated_content_eval.py`,
  `scripts/evaluate_support_ticket_generated_content.py`,
  `tests/test_extracted_blog_generation.py`, and
  `tests/test_evaluate_support_ticket_generated_content.py`
  - Passed.
- Combined pytest over `tests/test_smoke_content_ops_live_generation.py`,
  `tests/test_evaluate_support_ticket_generated_content.py`,
  `tests/test_extracted_blog_generation.py`, and
  `tests/test_extracted_content_ops_live_execute_harness.py`
  - 115 passed.
- Combined pytest over those tests plus
  `tests/test_extracted_content_ops_execution.py` and
  `tests/test_support_ticket_provider_landing_blog_execute.py`
  after the CI failure fix
  - 176 passed.
- Execute harness pytest over
  `tests/test_extracted_content_ops_live_execute_harness.py`
  - 8 passed.
- Focused execute-route proof over
  `tests/test_extracted_content_ops_live_execute_harness.py::test_support_ticket_provider_feeds_real_blog_post_generation`
  and
  `tests/test_extracted_content_ops_live_execute_harness.py::test_support_ticket_provider_triggers_blog_generated_content_gate`
  - 2 passed.
- Focused trusted-context regression over
  `tests/test_extracted_blog_generation.py::test_generate_keeps_trusted_support_ticket_context_over_model_context`
  plus support-ticket block/repair tests
  - 3 passed.
- Extracted content pipeline validation script at
  `scripts/validate_extracted_content_pipeline.sh`
  - Passed.
- Atlas reasoning import audit at
  `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py`
  - Passed.
- Standalone extracted audit at `scripts/audit_extracted_standalone.py`
  - Passed.
- ASCII Python check at `scripts/check_ascii_python.sh`
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~100 |
| Evaluator module/wrapper relocation | ~1,120 |
| Blog quality/execute wiring | ~160 |
| Tests | ~390 |
| **Total** | **~1,770** |
