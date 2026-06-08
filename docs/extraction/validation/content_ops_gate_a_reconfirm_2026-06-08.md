# Gate A Live Convergence Proof: Post-Fix Reconfirm

Date: 2026-06-08

Issue: #1357 / #1376 follow-up

Lane: `content-ops/gate-a-output-quality`

Verdict: **Not self-certified**. The harness completed with `ok=true`, but the
reviewer owns the product-quality judgment against the exported drafts.

Review-response note: the first #1383 proof run exposed that the seeded
support-ticket blog source summary still contained debug generation wording
and that the blog debug-source detector missed the exact live phrase. The next
review showed that quote-specific detection was not enough: passing blog
variants could still narrate uploaded/source mechanics. This refresh makes the
support-ticket prose instruction unconditional, removes upload/source-row copy
from the live seed, broadens the detector to the source-mechanics class, and
reruns the same selected outputs.

Raw artifacts:

- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/execution-result.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/review-results.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/summary.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-landing_page.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-blog_post.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_reconfirm_20260608/export-sales_brief.json`

## Run

Command:

```bash
EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false \
python scripts/smoke_content_ops_gate_a_live_quality.py \
  --account-id 2b2b950d-f64b-4852-bc30-f92a34cdf169 \
  --user-id 11111111-1111-4111-8111-111111111111 \
  --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --output-dir tmp/content_ops_gate_a_reconfirm_20260608 \
  --outputs landing_page,blog_post,sales_brief \
  --variant-count 3 \
  --quality-repair-attempts 1 \
  --max-cost-usd 20.00 \
  --json
```

Runtime route:

- Real local Postgres: existing Atlas DB pool from `.env` / `.env.local`.
- Real model route: `anthropic/claude-sonnet-4-5`.
- Local Ollama fallback disabled:
  `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false`.
- Source material: `support_ticket_saas_demo_sources.csv`, 36 support-ticket rows.
- Outputs requested: `landing_page`, `blog_post`, `sales_brief`.
- Excluded output: `email_campaign`.
- Variant request: `variant_count=3`.
- Budget cap: `--max-cost-usd 20.00`; not triggered.

## Structural Result

The harness completed execution, review, and exact export with no missing
review ids. `summary.json` reports `ok=true`.

| Output | Saved ids | Exported rows | Review status |
|---|---:|---:|---|
| landing_page | 3 | 3 | approved, no missing ids |
| blog_post | 3 | 3 | approved, no missing ids |
| sales_brief | 3 | 3 | approved, no missing ids |

No selected-output variants were blocked in the final run. `summary.json`
records 3 generated variants for each selected output with empty per-variant
error lists.

The exported blog samples were checked for source-mechanics/debug-generation
phrases including `uploaded tickets`, `uploaded ticket set`, `source rows`,
`usable rows`, `included rows`, `rows were included`, `we ingested`,
`you sent`, `support ticket export`, and `source mechanics`; none were present
in public blog content or the full blog export JSON.

## Samples

Landing-page exports:

| Variant | Draft id | Title | Hero headline | Brand voice |
|---|---|---|---|---|
| pain_led | `d8b20a0d-44c5-42ee-ae7a-b29251426db8` | "Support Ticket FAQ Gap Audit - Turn Repeat Tickets Into Approved Answers" | "Your repeat tickets show the FAQ gaps" | passed |
| outcome_led | `067c7531-b8c7-4b88-acaf-4f7643cd43f6` | "Cut Repeat Support Tickets with FAQ Gap Audit" | "Turn Repeat Support Tickets into Approved FAQ Answers" | passed |
| social_proof | `104b8793-9015-4aac-abab-41917dd38309` | "Support Ticket FAQ Gap Audit - Turn Repeat Questions Into Approved Answers" | "Your repeat tickets already show the FAQ gaps" | passed |

Blog exports:

| Variant | Draft id | Title | Opening heading | Brand voice |
|---|---|---|---|---|
| pain_led | `0501c17f-90c9-4318-9d15-e3253160a4e5` | "Support Ticket FAQ Gaps: What Repeat Tickets Reveal Before Renewal" | "What repeat support questions show" | passed |
| outcome_led | `2f9ec4e3-475a-4046-9e01-182757d1de20` | "Support Ticket FAQ Gaps: What Your Repeat Questions Reveal Before Renewal" | "What repeat support questions show" | passed |
| social_proof | `44d56e20-d9c8-4626-8e39-e894cb8fd085` | "Support Ticket FAQ Gaps: What 36 Repeat Tickets Reveal Before Renewal" | "What repeat support questions show" | passed |

Sales-brief exports:

| Variant | Draft id | Headline | Brief type | Brand voice |
|---|---|---|---|---|
| pain_led | `3fc2cbe9-eca0-42ee-8804-2b63c9af9057` | "Your RevOps lead hit a wall exporting attribution data right before a board meeting. That's your renewal conversation starter." | renewal | passed |
| outcome_led | `490247f3-19bf-4c0e-b532-e78c369cf989` | "Your RevOps lead needs attribution exports before the board meeting. Renewal window is your leverage to fix the blocker." | renewal | passed |
| social_proof | `2689157c-2473-44e9-bc75-3f786b105aae` | "RevOps lead blocked on attribution exports days before board meeting. Support ticket shows renewal-risk friction." | renewal | passed |

## Reviewer Notes

This run is meant to confirm the post-fix convergence signal against the #1360
failure baseline, not to certify final product quality. Review the exported
drafts directly for:

- publishable openings with no debug-source narration;
- variant distinction in title, body, and section order;
- grounded counts only;
- second-person voice;
- whether the 3/3 exported samples meet the GOOD bar.

The artifact set is complete enough for a reviewer to judge the real samples
without rerunning the live generation.
