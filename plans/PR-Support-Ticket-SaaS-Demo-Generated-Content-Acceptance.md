# PR: Support-Ticket SaaS Demo Generated Content Acceptance

## Why this slice exists

The current support-ticket generated-content acceptance matrix proves the
landing-page and blog-post path against the packaged small CSV scenario. The
next gap is broader functional validation against the repo's 36-row SaaS demo
ticket CSV, which has multiple clusters and more realistic repeated customer
questions.

This slice keeps the work in validation mode. It should not tune prompts or
change generation logic unless the broader run exposes a data-truthfulness
failure that must be fixed before the result can be called accepted.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Functional validation

1. Run live Content Ops generation through the support-ticket input provider
   using `extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`.
2. Use the Haiku test routing env so the run does not spend on Sonnet.
3. Generate and export one landing-page draft and one blog-post draft.
4. Run the deterministic support-ticket generated-content evaluator against
   each exported draft.
5. Record a concise validation report with command outcomes, artifact shape,
   evaluator status, and any acceptance notes.
6. Commit a minimized accepted landing-page fixture and a minimized known-bad
   blog fixture so the result can be rechecked from a fresh checkout.
7. Tighten the support-ticket blog prompt and generated-content evaluator for
   the unsupported outcome phrasings found during the broader run.
8. Keep source-backed cadence allowances scoped to the exact source phrase so a
   generated cadence claim elsewhere in the same sentence still blocks.

### Files touched

- `plans/PR-Support-Ticket-SaaS-Demo-Generated-Content-Acceptance.md` - Plan doc for this broader validation slice.
- `docs/extraction/validation/support_ticket_saas_demo_generated_content_acceptance_2026-05-28.md` - Validation report for the SaaS demo support-ticket generated outputs.
- `docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_landing_page.json` - Minimized accepted landing-page export fixture.
- `docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/known_bad_saas_demo_blog_post.json` - Minimized known-bad blog export fixture from the final observed SaaS demo run.
- `atlas_brain/skills/digest/blog_post_generation.md` - Source blog prompt tightened for support-ticket no-outcome language.
- `extracted_content_pipeline/skills/digest/blog_post_generation.md` - Synced extracted blog prompt.
- `extracted_content_pipeline/support_ticket_generated_content_eval.py` - Support-ticket generated-content evaluator catches source-backed cadence and broader unsupported outcome claims.
- `tests/test_evaluate_support_ticket_generated_content.py` - Focused positive/negative fixtures for the new detector branches.
- `HARDENING.md` - Parked remaining support-ticket blog contract hardening and cost telemetry schema mismatch.

## Mechanism

The live run uses the existing smoke harness, real DB wiring, real support-ticket
input provider packaging, and pipeline-routed LLM calls:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output <landing_page|blog_post> \
  --account-id <validation account> \
  --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft <tmp export path> \
  --output-result <tmp result path> \
  --evaluate-generated-content \
  --json
```

The report records both the smoke result and the deterministic evaluator result.
The committed fixtures keep generated copy, source context, readiness summaries,
and the fields needed for future evaluator re-runs. They omit account-specific
ids, token usage, and local run metadata.

The landing fixture must pass the evaluator. The blog fixture is deliberately
known-bad: the final observed SaaS demo blog run still contained unsupported
benefit/outcome language, and this PR makes that artifact fail the evaluator
instead of false-passing.

## Intentional

- This is a broader acceptance run, not a new generator prompt slice.
- This uses Haiku for the validation call because Sonnet is too expensive for
  routine test generation.
- This stays out of FAQ generation and FAQ article output ownership. The CSV is
  used only as support-ticket source context for landing/blog outputs.
- This commits minimized synthetic fixtures instead of raw DB exports.
- This does not claim the SaaS demo blog path is accepted. The broader blog run
  exposed a deeper contract problem that needs the next hardening slice.

## Deferred

- Future PR: add a scripted regression gate once we have enough accepted
  representative fixture shapes to justify CI coverage.
- Future PR: run the same acceptance shape against a real customer export once a
  sanitized dataset is approved.
- Future PR: make the no-outcome descriptive support-ticket blog mode
  structural so the 36-row SaaS demo blog can produce an accepted fixture.
- Parked hardening:
  - Support-ticket blog generation needs contract-level descriptive mode before
    SaaS demo acceptance.
  - LLM usage storage schema mismatch hides per-run cost telemetry.

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output landing_page --account-id acct_support_ticket_saas_demo_acceptance_20260528_landing --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_generated_content_acceptance_20260528/landing-page-draft.json --output-result tmp/support_ticket_saas_demo_generated_content_acceptance_20260528/landing-page-result.json --evaluate-generated-content --json
  - Passed.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post ... --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --evaluate-generated-content --json
  - Multiple Haiku runs saved drafts after repair, but manual audit found
    unsupported outcome language. The final observed bad artifact is committed
    as a known-bad fixture and now fails the evaluator.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output landing_page docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/current_saas_demo_landing_page.json --pretty
  - Passed.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post docs/extraction/validation/fixtures/support_ticket_saas_demo_generated_content_acceptance_2026-05-28/known_bad_saas_demo_blog_post.json --pretty
  - Failed as expected on `support_ticket_outcome_claims_grounded`.
- Command: python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q
  - Passed, 46 tests.
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
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-saas-demo-generated-content-acceptance-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~141 |
| Validation doc | ~82 |
| Fixture exports | ~438 |
| Prompt/evaluator/tests | ~354 |
| Hardening notes | ~20 |
| **Total** | **~1037** |

This is intentionally over the normal diff budget because durable validation
fixtures are part of the acceptance evidence, and the broader run exposed
detector gaps that must be fixed inline to avoid false acceptance. The large
portion is minimized JSON generated from synthetic support-ticket artifacts.
