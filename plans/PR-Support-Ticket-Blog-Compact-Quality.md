# Support-Ticket Blog Compact Quality

## Why this slice exists

PR #1087's live 36-row SaaS demo retry reached the observed support-ticket blog
shell but failed before save on `content_too_short:1111_words_need_1500`. The
failure is not that the draft is empty or underdeveloped; the observed shell
produced the intended compact 1,111-word no-outcome support-ticket diagnostic,
while the generic long-form blog policy still required 1,500 words.

The same failed candidate also introduced fixed future tracking intervals
("30, 60, and 90 days") even though the upload was undated. That is a
source-contract gap: the support-ticket contract says to compare future tickets
but does not explicitly forbid fixed day windows when `has_dated_window=false`.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider/blog-quality

Slice phase: Production hardening

1. Extend the compact no-outcome support-ticket blog quality policy to cover
   the 36-row SaaS demo shape.
2. Keep larger support-ticket uploads on the default long-form blog policy.
3. Tighten the no-outcome support-ticket blog contract so undated uploads do
   not invite fixed future measurement windows.
4. Add generated-content evaluator backstop coverage for "next 30/60/90 days"
   style unsupported windows on undated uploaded-ticket sources.
5. Add focused tests for the compact-policy threshold, contract guidance, and
   evaluator negative fixture.
6. Address review follow-up by keeping the canonical and extracted skill
   prompts aligned with the 50-row compact threshold and allowing exact
   source-backed customer wording that contains a future interval.

### Files touched

- `extracted_content_pipeline/blog_generation.py` - compact support-ticket blog
  threshold and no-window measurement guidance.
- `atlas_brain/skills/digest/blog_post_generation.md` - canonical skill prompt
  compact-threshold language.
- `extracted_content_pipeline/skills/digest/blog_post_generation.md` - synced
  extracted skill prompt compact-threshold language.
- `extracted_content_pipeline/support_ticket_generated_content_eval.py` -
  undated upload timeframe backstop for future day windows, with source-backed
  customer wording carve-out.
- `tests/test_atlas_content_ops_infrastructure.py` - prompt threshold contract
  test.
- `tests/test_extracted_blog_generation.py` - compact policy and contract tests.
- `tests/test_evaluate_support_ticket_generated_content.py` - evaluator
  negative and source-backed fixtures for fixed future intervals.
- `tests/test_smoke_content_ops_live_generation.py` - integration expectation
  for the seeded support-ticket blog blueprint contract.
- `plans/PR-Support-Ticket-Blog-Compact-Quality.md` - this plan.

## Mechanism

The blog generator already selects a compact `QualityPolicy` for small
support-ticket uploads with no measured outcomes and no resolution evidence.
This slice adjusts that threshold to include the 36-row SaaS demo while leaving
larger uploads on the generic policy.

The support-ticket descriptive contract also gains explicit measurement
guidance that tells the model not to add fixed day/week/month checkpoints unless
the source context includes a dated window. The evaluator backstop catches the
known bad shape if it still appears in generated text, while using the same
source-backed span logic as cadence checks so quoted customer questions are not
treated as generated measurement claims.

## Intentional

- This does not disable blog quality gates or bypass repair. It routes the
  existing compact support-ticket shape to the already-supported compact
  quality policy.
- This does not run another live retry in the same PR. The next validation
  slice should rerun the 36-row SaaS demo path after this source fix lands.
- The evaluator backstop is not the primary fix; the prompt/data contract is
  tightened so generation has the correct evidence boundary up front.
- The root `HARDENING.md` LLM usage telemetry issue remains parked because it
  does not affect this quality-policy or truthfulness fix.
- The ownership lane is narrowed to the blog-quality sublane because #1092 is
  a separate robust-testing gate on discoverability false-greens. Both touch
  the evaluator, so the drift audit should still warn on file overlap, but the
  slices are not claiming the same work lane.

## Deferred

- Live 36-row SaaS demo blog retry after this source fix lands.
- Full deterministic rendering that bypasses free-form blog body generation
  remains deferred from PR #1086.
- Parked hardening considered but left parked: `LLM usage storage schema
  mismatch hides per-run cost telemetry`; it affects observability, not this
  source contract.

## Verification

- Command: python -m pytest tests/test_extracted_blog_generation.py::test_demo_sized_support_ticket_blog_context_uses_compact_quality_policy tests/test_extracted_blog_generation.py::test_support_ticket_descriptive_blog_contract_requires_no_outcome_or_resolution_evidence tests/test_evaluate_support_ticket_generated_content.py::test_blog_export_fails_unsupported_uploaded_ticket_future_interval -q
  - Passed, 3 tests.
- Command: python -m pytest tests/test_atlas_content_ops_infrastructure.py::test_blog_generation_prompt_trims_small_support_ticket_uploads tests/test_extracted_blog_generation.py::test_demo_sized_support_ticket_blog_context_uses_compact_quality_policy tests/test_extracted_blog_generation.py::test_support_ticket_blog_context_uses_compact_policy_when_included_count_is_small tests/test_evaluate_support_ticket_generated_content.py::test_blog_export_fails_unsupported_uploaded_ticket_future_interval tests/test_evaluate_support_ticket_generated_content.py::test_blog_export_allows_source_backed_uploaded_ticket_future_interval tests/test_evaluate_support_ticket_generated_content.py::test_blog_export_blocks_timeframe_outside_source_quote_in_same_sentence -q
  - Passed, 6 tests.
- Command: python -m pytest tests/test_extracted_blog_generation.py tests/test_evaluate_support_ticket_generated_content.py -q
  - Passed, 123 tests.
- Command: python -m py_compile extracted_content_pipeline/support_ticket_generated_content_eval.py tests/test_evaluate_support_ticket_generated_content.py tests/test_extracted_blog_generation.py tests/test_atlas_content_ops_infrastructure.py && python -m pytest tests/test_extracted_blog_generation.py tests/test_evaluate_support_ticket_generated_content.py tests/test_atlas_content_ops_infrastructure.py::test_blog_generation_prompt_trims_small_support_ticket_uploads -q
  - Passed, 127 tests.
- Command: python -m pytest tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_uses_csv_counts tests/test_extracted_blog_generation.py tests/test_evaluate_support_ticket_generated_content.py -q
  - Passed, 124 tests after updating the integration expectation.
- Command: python -m py_compile extracted_content_pipeline/blog_generation.py extracted_content_pipeline/support_ticket_generated_content_eval.py tests/test_extracted_blog_generation.py tests/test_evaluate_support_ticket_generated_content.py
  - Passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh && python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline && python scripts/audit_extracted_standalone.py --fail-on-debt && bash scripts/check_ascii_python.sh && bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - Passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - First run failed on the stale
    `tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_uses_csv_counts`
    measurement-guidance expectation. After updating that integration test,
    rerun passed with 2,615 passed, 8 skipped, 1 warning.
  - Review-fix rerun passed with 2,618 passed, 8 skipped, 1 warning.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/support-ticket-blog-compact-quality-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Blog contract and policy | ~15 |
| Skill prompt threshold sync | ~2 |
| Evaluator backstop | ~45 |
| Tests | ~145 |
| Plan doc | ~110 |
| Total | ~315 |
