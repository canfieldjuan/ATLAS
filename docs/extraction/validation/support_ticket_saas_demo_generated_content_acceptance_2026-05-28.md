# Support-Ticket SaaS Demo Generated Content Acceptance - 2026-05-28

## Scope

This validation runs the support-ticket landing/blog generation path against the
repo's 36-row SaaS demo CSV:

`extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`

The CSV has 36 synthetic support-ticket rows, 35 direct customer questions, and
7 observed clusters. It is broader than the packaged 4-row support-ticket CSV
used by the current acceptance matrix.

## Result

| Output | Status | Artifact | Notes |
|---|---|---|---|
| Landing page | Accepted | `docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_landing_page.json` | Live generation saved a draft, export context matched the provider package, SEO/AEO and GEO readiness were ready, and the support-ticket generated-content evaluator passed. |
| Blog post | Accepted | `docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_blog_post.json` | After the descriptive contract, GEO repair guidance, save-time/export citable-section alignment, citable repair contract, and false-green outcome detector fixes landed, the final live retry saved a draft with generated-content evaluation passed, SEO/AEO ready, GEO ready, and manual truthfulness review accepted. |

Both the landing page and blog post paths are accepted for this representative
CSV shape.

## What Changed During Validation

The first blog run blocked on a cadence false positive: the generated article
quoted a source question that contained "weekly", and the evaluator treated the
word as an invented recurring cadence. The fix was to allow cadence language
only when the sentence includes exact cadence-bearing source wording from the
support-ticket context.

Subsequent runs exposed unsupported outcome claims in several forms:

- a support ticket could have been prevented by an FAQ entry
- a FAQ could address incoming ticket volume
- support ticket volume or repeat-question volume would be reduced
- questions would stop appearing
- the team would handle future instances more efficiently
- customers would find answers faster or resolve without follow-up
- fixed 2-4 week measurement windows or resolution-time improvements were implied

Those forms now have focused negative fixtures in
`tests/test_evaluate_support_ticket_generated_content.py`, and the blog prompt
now names those forbidden claims explicitly.

## Acceptance Notes

The landing page stayed within the current support-ticket evidence contract:

- used uploaded-ticket framing instead of a dated window
- surfaced 36 source rows, 36 included rows, and 35 question-like rows
- mentioned observed ticket clusters and customer wording
- avoided unsupported outcome, cadence, and resolution-step claims

The blog generator no longer fails on the original unsupported outcome language
for this CSV shape, and save-time validation now blocks drafts that would miss
the exported GEO citable-section readiness bar. After the citable repair
contract and false-green outcome detector fixes landed, the final live retry
produced a draft that passed generated-content evaluation, SEO/AEO readiness,
GEO readiness, and manual truthfulness review.

## Follow-up Retry After GEO Repair Guidance

After `PR-Support-Ticket-Blog-GEO-Clarity-Repair` landed, the live Haiku retry
used the same 36-row SaaS demo CSV and the same generated-content evaluator.

Result: not accepted yet.

The run did make progress:

- saved blog draft id: `4e0a7748-4247-4e34-b20f-81b5f19e8c01`
- generated-content evaluation: passed
- SEO/AEO readiness: ready
- GEO readiness: `needs_review`
- missing GEO check: `citable_section_structure`

The exported H2 headings were specific and no vague `## Summary` heading
remained:

- `What do repeat support tickets reveal?`
- `Which FAQ gaps should a small team fix first?`
- `How should old tickets become review-ready FAQ shells?`
- `What to measure after publishing`
- `Building a sustainable FAQ process`
- `Next steps for your team`

Diagnostic replay showed the save-time quality pack passed the same draft with
only a `methodology_disclaimer_missing_self_selected` warning. The source issue
is that save-time GEO citable-section validation and export GEO readiness are
not using equivalent self-contained-section rules.

## Follow-up Retry After Citable-Section Alignment

After `PR-Blog-GEO-Citable-Readiness-Alignment` landed, the live Haiku retry
used the same 36-row SaaS demo CSV and generated-content evaluation settings.

Result: not accepted yet.

The run failed before save:

- blueprint id: `1be74115-82b2-4901-9d94-e592438f920c`
- saved draft ids: none
- generated-content evaluation: not run because no draft was saved
- failed blocker: `geo_citable_section_structure_missing`
- repair attempts used: 2
- failed candidate title: `Support-Ticket Questions Customers Keep Asking: What 36 Uploaded Tickets Reveal`
- failed candidate target keyword: `support ticket FAQ gaps`
- failed candidate word count: 2303

This is progress in the gate behavior: the draft no longer saves and then fails
export readiness. The remaining issue is that the repair loop needs more direct
instructions for the stricter citable-section requirement.

## Follow-up Retry After False-Green Outcome Fixes

After `PR-Support-Ticket-Blog-False-Green-Outcome-Claims` landed, the live
Haiku retry used the same 36-row SaaS demo CSV and generated-content evaluation
settings.

Result: accepted.

The run saved a blog draft that passed all acceptance gates:

- saved blog draft id: `4ff4f0fe-274e-46ad-9af2-e47475dd749b`
- generated-content evaluation: passed
- SEO/AEO readiness: ready
- GEO readiness: ready
- generation model: `anthropic/claude-haiku-4-5`
- generation quality repair attempts: 2
- manual truthfulness scan: accepted; no unsupported ticket-reduction,
  resolution-time, churn, retention, ROI, support-capacity, or concrete
  resolution-step claims found
- accepted fixture:
  `docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_blog_post.json`

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output landing_page --account-id acct_support_ticket_saas_demo_acceptance_20260528_landing --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_generated_content_acceptance_20260528/landing-page-draft.json --output-result tmp/support_ticket_saas_demo_generated_content_acceptance_20260528/landing-page-result.json --evaluate-generated-content --json
  - Passed.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post ... --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --evaluate-generated-content --json
  - Multiple runs saved drafts after repair, but manual audit found unsupported outcome language. The final observed bad artifact is committed as a known-bad fixture and now fails the evaluator.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output landing_page docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_landing_page.json --pretty
  - Passed.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/known_bad_saas_demo_blog_post.json --pretty
  - Failed as expected on `support_ticket_outcome_claims_grounded`.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_after_geo --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_after_geo/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_after_geo/blog-post-result.json --evaluate-generated-content --json
  - Saved draft `4e0a7748-4247-4e34-b20f-81b5f19e8c01`; generated-content evaluation passed; SEO/AEO ready; GEO `needs_review` on missing `citable_section_structure`.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_saas_demo_blog_acceptance_20260528_after_geo/blog-post-draft.json --pretty
  - Passed.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_after_citable --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_after_citable/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_after_citable/blog-post-result.json --evaluate-generated-content --json
  - Failed before save on `geo_citable_section_structure_missing` after two repair attempts; no saved draft export was produced.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_final_retry --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_final_retry/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_final_retry/blog-post-result.json --evaluate-generated-content --json
  - Saved draft `4ff4f0fe-274e-46ad-9af2-e47475dd749b`; generated-content evaluation passed; SEO/AEO ready; GEO ready.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_blog_post.json --pretty
  - Passed.
- Command: python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q
  - Passed, 47 tests.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed.
- Command: bash scripts/check_ascii_python.sh
  - Passed.
- Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - Passed.
