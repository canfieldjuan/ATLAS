# Support-Ticket Descriptive Blog Live Validation - 2026-05-26

## Scope

This validation reran the live Content Ops support-ticket blog-post path after
PR #983 changed the no-outcome/no-resolution blog contract from promissory to
descriptive.

The goal was to prove the real route can now save a grounded support-ticket
blog draft from uploaded-ticket context instead of failing closed or saving
unsupported benefit claims.

Environment:

- Repo: `/home/juan-canfield/Desktop/Atlas-support-ticket-provider`
- Env files:
  - `/home/juan-canfield/Desktop/Atlas/.env`
  - `/home/juan-canfield/Desktop/Atlas/.env.local`
  - `tmp/support_ticket_live_haiku_eval_20260525/haiku.env`
- Artifact directory:
  - `tmp/support_ticket_descriptive_blog_live_validation_20260526`
- Model override:
  - Claude Haiku family through OpenRouter

Source input:

- `extracted_content_pipeline/examples/support_ticket_sources.csv`
- 4 uploaded support-ticket rows
- 2 direct customer questions
- 2 ticket clusters:
  - `email and profile updates` - 2 tickets
  - `reporting friction` - 2 tickets
- No dated window
- No measured outcomes
- No verified resolution evidence

## Command

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

Explicit evaluator rerun:

```bash
python scripts/evaluate_support_ticket_generated_content.py \
  --output blog_post \
  tmp/support_ticket_descriptive_blog_live_validation_20260526/blog-post-draft.json \
  --pretty
```

## Result

Live Haiku blog generation passed and saved one draft.

- Saved draft id: `3e9de393-2eb6-4afd-b2a0-62d77a11dd87`
- Draft title:
  `Support-Ticket Questions Customers Keep Asking: What Your Uploaded Tickets Reveal`
- SEO/AEO readiness: ready
- GEO readiness: ready
- Generated-content evaluation: passed
- Source count visibility: passed with visible counts `4`, `4`, and `2`
- Matched source signals:
  - `email and profile updates`
  - `reporting friction`
  - `How do I change my login email?`
  - `How do we export campaign attribution data before renewal?`

Generation usage:

- Input tokens: 23,153
- Output tokens: 10,315
- Cache write tokens: 18,684
- Cached tokens: 4,460
- Billable input tokens: 9

## Manual Audit

The saved draft is source-truthful for the support-ticket evidence contract:

- It uses the uploaded-ticket source period and does not invent a calendar
  window or recurring cadence.
- It cites the real uploaded counts: 4 rows, 2 clusters, and 2 tickets per
  cluster.
- It preserves customer wording from the source rows.
- It does not make percentage, ROI, churn, retention, ticket-volume, capacity,
  or time-savings claims.
- It keeps verified-resolution content as a review-needed placeholder:
  `Draft answer - support team should add the verified resolution before publishing.`
- It does not include concrete UI paths or exact product steps when
  `support_ticket_resolution_evidence_present` is false.

The draft does contain operational review advice such as verifying feature
availability, UI accuracy, permissions, completeness, and clarity before
publishing. That is acceptable because it is guidance for the support team to
verify answers, not a claim that the uploaded tickets contain verified product
resolution steps.

## Parked Polish

The output is safe but long and repetitive for a 4-row upload. It repeats some
cluster explanations and used more than 10k output tokens. This is not a
truthfulness blocker for this validation, but it should be tightened before the
product experience is polished.

Parked in `HARDENING.md`:

- `Support-ticket descriptive blog output is long and repetitive on tiny uploads`

## Verification

- Live Haiku blog-post smoke with support-ticket CSV and generated-content
  evaluation - passed.
- Explicit generated-content evaluator rerun - passed.
- Manual copy audit for unsupported outcomes, unsupported timeframes/cadence,
  and concrete answer steps without resolution evidence - passed.

