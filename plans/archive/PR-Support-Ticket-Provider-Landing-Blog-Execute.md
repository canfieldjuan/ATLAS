# PR: Support Ticket Provider Landing Blog Execute

## Why this slice exists

The support-ticket input provider now has execute-route proof for FAQ Markdown,
but the original ingestion-provider goal also includes landing-page and blog
generation. Preview and plan already show that support-ticket rows expand into
`faq_markdown`, `landing_page`, and `blog_post`; this slice proves the
provider-built package also reaches the landing-page and blog dispatchers with
the expected FAQ Report context.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Add a focused execute-route test for support-ticket input provider output
   feeding `landing_page`.
2. Add a focused execute-route test for support-ticket input provider output
   feeding `blog_post`.
3. Keep services offline and deterministic. This validates handoff and
   dispatcher wiring without DB, LLM, or FAQ-session ownership.
4. Enroll the new test file in extracted pipeline CI so the proof runs on pull
   requests and pushes.

### Files touched

- `tests/test_support_ticket_provider_landing_blog_execute.py`
- `.github/workflows/extracted_pipeline_checks.yml`
- `scripts/run_extracted_pipeline_checks.sh`
- `plans/PR-Support-Ticket-Provider-Landing-Blog-Execute.md`

## Mechanism

The tests build the extracted Content Ops control-surface router with the Atlas
support-ticket input provider and fake generation services. Each test sends the
packaged support-ticket CSV rows as `inputs.source_material` and explicitly
requests one output.

For `landing_page`, the fake service captures the `MarketingCampaign` object so
the test can assert FAQ Report offer, audience, CTA, source period,
SEO/GEO/AEO fields, and extracted customer questions reach
`campaign.context`.

For `blog_post`, the fake service captures dispatcher kwargs so the test can
assert the provider-built topic and topic-type filter reach the blog generator
path.

The workflow path filters and extracted pipeline check script include the new
test file so this validation gates in CI instead of relying only on local
execution.

## Intentional

- No production code changes. This is validation only.
- No live LLM or database calls. Generator content quality remains covered by
  the existing live-generation and smoke scripts.
- No FAQ Markdown behavior changes. FAQ generation remains owned by the FAQ
  lane.
- This avoids `tests/test_atlas_content_ops_input_provider.py` because PR #916
  currently touches that file.
- The CI enrollment is included in this PR because a new test file would
  otherwise run locally but not in extracted pipeline CI.

## Deferred

- Full uploaded CSV -> persisted import -> provider -> landing/blog execution
  waits for the file-ingestion/import lookup lane.
- Live DB/LLM landing/blog generation from support-ticket fixtures can follow
  after this deterministic handoff is locked.
- Parked hardening: none.

## Verification

- Python compile check for
  `tests/test_support_ticket_provider_landing_blog_execute.py` - passed.
- Pytest focused landing/blog provider execute suite:
  `tests/test_support_ticket_provider_landing_blog_execute.py` - 2 passed.
- Pytest support-ticket provider/package suite:
  `tests/test_support_ticket_provider_landing_blog_execute.py`
  `tests/test_extracted_support_ticket_input_provider.py`
  `tests/test_extracted_support_ticket_input_package.py` - 28 passed.
- `git diff --check` - passed.
- Local PR review - pending after CI enrollment update.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~95 |
| New tests | ~200 |
| CI enrollment | ~5 |
| **Total** | **~300** |
