# Support-Ticket Blog Small Upload Live Validation - 2026-05-26

## Scope

This validation followed PR #987, which tightened the support-ticket blog prompt
so tiny no-outcome/no-resolution uploads should produce compact descriptive
drafts instead of long, repetitive articles.

The validation used the same packaged 4-row support-ticket CSV that previously
produced a truth-safe but long draft.

Environment:

- Repo: `/home/juan-canfield/Desktop/Atlas-support-ticket-provider`
- Env files:
  - `/home/juan-canfield/Desktop/Atlas/.env`
  - `/home/juan-canfield/Desktop/Atlas/.env.local`
  - `tmp/support_ticket_live_haiku_eval_20260525/haiku.env`
- Model override: Claude Haiku family through OpenRouter
- Source CSV: `extracted_content_pipeline/examples/support_ticket_sources.csv`

Source input:

- 4 uploaded support-ticket rows
- 2 direct customer questions
- 2 ticket clusters:
  - `email and profile updates` - 2 tickets
  - `reporting friction` - 2 tickets
- No dated window
- No measured outcomes
- No verified resolution evidence

## Commands

Initial post-#987 validation run:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_blog_small_upload_live_validation_20260526 \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_blog_small_upload_live_validation_20260526/blog-post-draft.json \
  --output-result tmp/support_ticket_blog_small_upload_live_validation_20260526/blog-post-result.json \
  --evaluate-generated-content \
  --json
```

Final validation run after the prompt and compact quality-policy fix:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_blog_small_upload_live_validation_20260526_policy \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_blog_small_upload_live_validation_20260526_policy/blog-post-draft.json \
  --output-result tmp/support_ticket_blog_small_upload_live_validation_20260526_policy/blog-post-result.json \
  --evaluate-generated-content \
  --json
```

Explicit evaluator rerun:

```bash
python scripts/evaluate_support_ticket_generated_content.py \
  --output blog_post \
  tmp/support_ticket_blog_small_upload_live_validation_20260526_policy/blog-post-draft.json \
  --pretty
```

## Intermediate Findings

The first post-#987 live run saved draft
`f3e51234-004e-40dd-bd01-b5516e1e0bb0`, but it did not fully validate the
compact product goal:

- 1,709 words
- 5 H2 sections
- 3 H3 subsections
- 5,805 output tokens
- SEO/AEO readiness: ready
- GEO readiness: needs_review
- Missing GEO check: `citable_section_structure`
- Generated-content evaluation: passed

This proved the prompt trim helped but was still too loose. The prompt allowed
process/scaling sections and the blog quality gate still used the default
1500-word floor, which fought the 700-1100 word compact prompt target.

A second live run after prompt tightening blocked correctly before save:

- Reason: `quality_blocked`
- Blockers:
  - `content_too_short:1496_words_need_1500`
  - unsupported support-ticket outcome claim:
    `customers find answers without opening a support ticket`

That block identified the source integration gap: the blog quality gate needed
to receive the same compact word-count policy as the support-ticket prompt for
small no-outcome/no-resolution uploads.

## Final Result

The final Haiku run passed and saved one draft.

- Saved draft id: `4dc73f34-3bfa-45f1-91ab-afe19a9df339`
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

- Input tokens: 13,703
- Output tokens: 3,993
- Cache write tokens: 13,697
- Cached tokens: 0
- Billable input tokens: 6

Shape comparison:

| Run | Saved | Words | H2 | H3 | Output tokens | SEO/AEO | GEO |
|---|---:|---:|---:|---:|---:|---|---|
| Pre-#987 baseline | yes | 1,903 | 9 | 0 | 10,315 | ready | ready |
| Post-#987 prompt-only | yes | 1,709 | 5 | 3 | 5,805 | ready | needs_review |
| Final compact policy | yes | 1,095 | 4 | 0 | 3,993 | ready | ready |

## Manual Audit

The final saved draft is source-truthful and matches the compact support-ticket
brief shape:

- It uses the uploaded-ticket source period and does not invent a calendar
  window or recurring cadence.
- It cites the real uploaded counts: 4 rows, 2 clusters, and 2 tickets per
  cluster.
- It preserves customer wording from the source rows.
- It does not make percentage, ROI, churn, retention, ticket-volume, capacity,
  or time-savings claims.
- It keeps FAQ answers in review-needed form because no verified resolution
  evidence is present.
- It does not include concrete UI paths, menu names, or exact product steps.
- It has 4 H2 sections and no H3 subsections.
- It is within the compact 700-1100 word target at 1,095 words.
- Blog SEO/AEO and GEO readiness are both ready.

## Verification

- Initial live Haiku validation after #987 - passed truthfulness, but missed
  compact/GEO shape.
- Prompt and compact quality-policy fix - implemented after the first two live
  findings.
- Final live Haiku validation - passed and saved draft
  `4dc73f34-3bfa-45f1-91ab-afe19a9df339`.
- Explicit generated-content evaluator rerun - passed.
- Manual copy audit - passed.
