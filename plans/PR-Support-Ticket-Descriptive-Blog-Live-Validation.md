# PR: Support-Ticket Descriptive Blog Live Validation

## Why this slice exists

PR #983 changed the support-ticket blog contract so no-outcome/no-resolution
ticket uploads can produce a descriptive article instead of either saving
unsupported claims or failing closed. Unit and service tests proved the new
contract deterministically, but the live route still needs to prove the real
Haiku generation path can produce and save a grounded support-ticket blog draft
from the packaged CSV.

This slice is the live validation follow-up named in #983. It exercises the
host DB pool, support-ticket input provider, packaged skills, pipeline-routed
LLM, save path, export path, and generated-content evaluator together.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider-descriptive-blog-live-validation
Slice phase: Functional validation

1. Run a live Claude Haiku blog-post smoke with the packaged support-ticket CSV.
2. Export the saved draft if one is produced.
3. Run the deterministic support-ticket generated-content evaluation against
   the saved export.
4. Manually inspect the saved draft or blocked candidate for the contract
   classes this lane protects:
   - unsupported support-volume, churn, retention, time-savings, or speed claims
   - unsupported date-window or cadence claims
   - concrete answer steps without resolution evidence
   - copied guardrail language instead of reader-facing descriptive prose
5. Record commands, artifact paths, result, and follow-up in a validation doc.
6. If live output exposes a data-truthfulness blocker, fix it in this PR. If it
   exposes only non-blocking product polish, park it in the appropriate
   hardening file.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Support-Ticket-Descriptive-Blog-Live-Validation.md` | Plan doc for this live validation slice. |
| `docs/extraction/validation/support_ticket_descriptive_blog_live_validation_2026-05-26.md` | Live Haiku validation record. |
| `HARDENING.md` | Park non-blocking product polish found during live validation. |

## Mechanism

Use the existing live smoke harness with the Haiku override:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_descriptive_blog_live_validation_20260526_blog \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_descriptive_blog_live_validation_20260526/blog-post-draft.json \
  --output-result tmp/support_ticket_descriptive_blog_live_validation_20260526/blog-post-result.json \
  --evaluate-generated-content \
  --json
```

The saved JSON artifacts stay under `tmp/`; the committed validation doc records
their paths and summarizes the result. The slice succeeds if the live path
either saves a draft that passes the generated-content evaluator or blocks a
misleading candidate before save.

## Intentional

- This is validation, not another prompt or evaluator redesign unless live
  output proves a truthfulness blocker.
- The run uses Haiku because it is the cheaper and stricter stress case for
  this lane.
- This does not add FAQ Article output or customer-language keyword promotion;
  that future product slice was logged in #983.

## Deferred

- Parked hardening: `Support-ticket descriptive blog output is long and
  repetitive on tiny uploads`.
- Broader support-ticket acceptance testing across many real customer CSV
  shapes remains a later robust-testing slice.
- Customer-language keyword promotion and standalone FAQ Article output remain
  a future product slice owned outside this validation PR.

## Verification

Planned:

- Live Haiku blog-post smoke with support-ticket CSV and generated-content
  evaluation - passed; saved draft id
  `3e9de393-2eb6-4afd-b2a0-62d77a11dd87`.
- Manual copy audit of the saved draft - passed for source-truthfulness;
  parked long/repetitive article shape as polish.
- `python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_descriptive_blog_live_validation_20260526/blog-post-draft.json --pretty`
  - passed.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q`
  - `42 passed`.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py tests/test_extracted_blog_generation.py tests/test_support_ticket_provider_landing_blog_execute.py tests/test_extracted_content_ops_live_execute_harness.py -q`
  - `118 passed`.
- `bash scripts/local_pr_review.sh --current-pr-body-file <PR body file>`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~115 |
| Validation doc | ~150 |
| Hardening note | ~10 |
| **Total** | **~275** |
