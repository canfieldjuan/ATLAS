# Support-Ticket Source Context Live Validation - 2026-05-24

## Scope

This validation proves the support-ticket input provider can feed live
landing-page and blog-post generation, persist drafts, and export the exact
saved draft rows for inspection after PR #930.

Ownership lane: `content-ops/support-ticket-input-provider`

## Environment

- Repo: `canfieldjuan/ATLAS`
- Branch: `claude/support-ticket-source-context-live-validation`
- Model override: `anthropic/claude-haiku-4-5`
- Source CSV:
  `extracted_content_pipeline/examples/support_ticket_sources.csv`
- Account ids:
  - Landing: `acct_support_ticket_source_context_20260524`
  - Blog: `acct_support_ticket_source_context_20260524b`

The Atlas `.env` and `.env.local` files supplied DB and OpenRouter credentials.
The Haiku override env file was loaded last.

## Commands

Landing page:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --account-id acct_support_ticket_source_context_20260524 \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file /tmp/atlas-support-ticket-source-context-validation/haiku.env \
  --export-saved-draft /tmp/atlas-support-ticket-source-context-validation/landing-page-draft.json \
  --json
```

Blog post:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_source_context_20260524b \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file /tmp/atlas-support-ticket-source-context-validation/haiku.env \
  --export-saved-draft /tmp/atlas-support-ticket-source-context-validation/blog-post-draft-neutral-period.json \
  --json
```

## Landing Result

- Smoke status: passed
- Saved draft id: `d181fc92-1711-40dd-98e0-e79bcdb1c304`
- Export row count: 1
- Quality repair attempts: 1
- Last repair result: passed

The exported landing-page draft includes `metadata.source_context` with:

- `source_row_count`: 4
- `included_ticket_row_count`: 4
- `skipped_ticket_row_count`: 0
- `truncated_ticket_row_count`: 0
- `question_like_ticket_count`: 2
- `top_ticket_clusters`:
  - `email and profile updates`: 2
  - `reporting friction`: 2
- Four `customer_wording_examples`, including login-email, account-email,
  campaign-attribution, and reporting-dashboard wording.

Visible/exported landing copy contains the real source terms:

- `email and profile updates`
- `reporting friction`
- `login email`
- `account email`
- `campaign attribution`
- `reporting dashboard`

## Blog Result

- Smoke status: passed
- Seeded blueprint id: `a7f2421b-f2ff-40ae-82a4-d0b1435ea44f`
- Saved draft id: `90cf80e1-baad-478a-8b25-98394c509279`
- Export row count: 1
- SEO/AEO readiness: ready
- GEO readiness: ready

The exported blog draft carries:

- `data_context.source_row_count`: 4
- `data_context.question_like_ticket_count`: 2
- `data_context.top_clusters`:
  - `email and profile updates`: 2
  - `reporting friction`: 2
- `data_context.source_period`: `Uploaded support tickets`
- `data_context.review_period`: `uploaded tickets`

Because the packaged CSV does not include parseable ticket dates, the final
blog export does not claim `last 90 days`.

## Notes

The live validation exposed and fixed a smoke-helper truthfulness gap: the
support-ticket blog blueprint used to claim a 90-day source window even when
the CSV rows did not carry dates. The helper now uses `Last 90 days of support
tickets` only when every included row has a parseable date. Undated CSVs use
neutral uploaded-ticket wording.

No FAQ generator or file-ingestion behavior was changed in this slice.
