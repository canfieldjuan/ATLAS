# PR: Support Ticket Provider Real Landing Blog Generation

## Why this slice exists

The support-ticket provider has proof for FAQ execute and a pending proof that
provider-built inputs reach landing/blog dispatchers. The next validation gap is
actual generator execution: provider-built support-ticket inputs should drive
the real `LandingPageGenerationService` and `BlogPostGenerationService` to save
drafts when hosts inject deterministic LLM and repository ports.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Add a deterministic real-service execute test for support-ticket provider
   inputs feeding `landing_page`.
2. Add a deterministic real-service execute test for support-ticket provider
   inputs feeding `blog_post`.
3. Reuse an existing enrolled test file and add it to the extracted-checks path
   trigger so future test-only edits self-trigger CI.

### Files touched

- `tests/test_extracted_content_ops_live_execute_harness.py`
- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Support-Ticket-Provider-Real-Landing-Blog-Generation.md`

## Mechanism

The tests build a Content Ops control-surface router with the Atlas
support-ticket input provider, actual `LandingPageGenerationService` or
`BlogPostGenerationService`, deterministic fake LLM responses, and in-memory
repositories. They post the packaged support-ticket CSV rows to `/content-ops/execute`.

The landing-page test asserts a draft is saved and the saved draft carries FAQ
Report campaign fields, support-ticket CTA/meta, and support-ticket reference
IDs.

The blog-post test seeds a matching support-ticket blueprint, executes
`blog_post`, and asserts a blog draft is saved with the support-ticket topic and
metadata from the real generator path.

The workflow update adds the existing harness file to both `pull_request` and
`push` path filters. The test was already enrolled in
`scripts/run_extracted_pipeline_checks.sh`; this closes the trigger gap called
out in review.

## Intentional

- No production code changes. This is validation only.
- No live LLM or database calls; fake LLM and memory repositories keep the run
  deterministic and cheap.
- No FAQ generator changes. FAQ output remains owned by the FAQ lane.
- This does not touch `tests/test_atlas_content_ops_input_provider.py` or the
  new #919 test file while those PRs are active.

## Deferred

- Live DB/LLM landing/blog generation from support-ticket fixtures can follow
  once deterministic service execution is locked.
- Full uploaded CSV -> persisted import -> provider -> landing/blog generation
  remains blocked on the file-ingestion/import lookup lane.
- Parked hardening: none. `HARDENING.md` was scanned; the existing FAQ scale
  and file-ingestion concurrency entries are outside this support-ticket
  landing/blog validation slice.

## Verification

- Focused pytest for `tests/test_extracted_content_ops_live_execute_harness.py`
  - 7 passed.
- Py compile for `tests/test_extracted_content_ops_live_execute_harness.py` -
  passed.
- Git whitespace check - passed.
- Local PR review wrapper - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Workflow | ~5 |
| Plan | ~85 |
| Tests | ~280 |
| **Total** | **~370** |
