# PR: Support-Ticket SaaS Demo Blog Accepted Fixture

## Why this slice exists

The 36-row SaaS demo support-ticket blog path has been blocked by two source
issues: unsupported outcome phrasing and then `geo_entity_clarity_missing`
repair guidance that did not address vague H2 headings. The descriptive
contract and GEO repair guidance have now landed. This slice reruns the real
Haiku-backed Content Ops blog path and records whether the representative SaaS
demo CSV can finally produce a saved, evaluator-passing blog draft.

This is the live proof point deferred by
`PR-Support-Ticket-Blog-GEO-Clarity-Repair`.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Functional validation

1. Run one live Haiku blog-post smoke against the 36-row SaaS demo support-ticket
   CSV with saved draft export and generated-content evaluation enabled.
2. Do not commit a fixture unless the saved draft passes support-ticket
   truthfulness evaluation and exported SEO/AEO plus GEO readiness.
3. Update the SaaS demo generated-content validation report with the exact run
   result.
4. If the run still fails or exposes a validator mismatch, do not chase another
   source fix in this slice; record the blocker and park the next source issue.

### Files touched

- `plans/PR-Support-Ticket-SaaS-Demo-Blog-Accepted-Fixture.md` - Plan doc for the live validation slice.
- `docs/extraction/validation/support_ticket_saas_demo_generated_content_acceptance_2026-05-28.md` - SaaS demo acceptance status update.
- `HARDENING.md` - Park the validator mismatch exposed by the live retry.

## Mechanism

The smoke command uses the existing live harness:

python scripts/smoke_content_ops_live_generation.py --output blog_post
--support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv
--evaluate-generated-content --export-saved-draft <tmp export path>

The command loads the Atlas env files plus the existing Haiku override env file,
packages the CSV through the support-ticket input provider, seeds the blog
blueprint, routes the LLM through the configured provider, saves the draft, and
runs deterministic generated-content evaluation against the saved export.

Acceptance requires all three signals to be true:

- the draft saves through the real Content Ops execution path;
- deterministic support-ticket generated-content evaluation passes;
- exported SEO/AEO and GEO readiness are both `ready`.

## Intentional

- This uses Haiku for validation spend, not Sonnet.
- This does not alter prompts, detectors, repair guidance, or readiness
  validators. If live generation exposes a source issue, this slice records it
  instead of patching another symptom inline.
- This does not touch FAQ article generation or FAQ report ownership.

## Deferred

- Future PR: align save-time blog GEO citable-section validation with export
  readiness before accepting the SaaS demo blog fixture.
- Future PR: add a scripted regression gate once the accepted SaaS demo blog
  fixture exists and the current run proves the fixture shape.
- Parked hardening:
  - Blog save-time GEO gate and export readiness disagree on citable sections.

## Verification

- Live Haiku blog smoke against the 36-row SaaS demo CSV - saved draft
  `4e0a7748-4247-4e34-b20f-81b5f19e8c01`; generated-content evaluation passed;
  exported SEO/AEO readiness was ready; exported GEO readiness was
  `needs_review` because `citable_section_structure` was missing.
- Direct support-ticket generated-content evaluator CLI against the saved draft
  export - passed.
- Diagnostic quality-pack replay against the saved draft - save-time quality
  passed with only `methodology_disclaimer_missing_self_selected` warning,
  confirming the mismatch is between save-time and export GEO citable checks.
- Local PR review with the support-ticket SaaS demo blog fixture PR body - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| Validation report update | ~100 |
| HARDENING entry | ~9 |
| **Total** | **~202** |
