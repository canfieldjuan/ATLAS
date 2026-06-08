# Gate A Live Convergence Proof: Post-Fix Reconfirm

Date: 2026-06-08

Issue: #1357 / #1376 follow-up

Lane: `content-ops/gate-a-output-quality`

Verdict: **Not self-certified**. The harness completed with `ok=true`, but the
reviewer owns the product-quality judgment against the exported drafts.

Review-response note: the first #1383 proof run exposed that the seeded
support-ticket blog source summary still contained debug generation wording
and that the blog debug-source detector missed the exact live phrase. This
refresh includes the detector/source-summary hardening and reruns the same
selected outputs. The rerun blocked a bad blog candidate on
`support_ticket_generated_content:debug_source_narration` and exported only the
saved drafts listed below.

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
| blog_post | 2 | 2 | approved, no missing ids |
| sales_brief | 3 | 3 | approved, no missing ids |

Blocked variants recorded by the harness:

| Output | Variant | Blocker |
|---|---|---|
| blog_post | pain_led | `geo_citable_section_structure_missing`, `support_ticket_generated_content:debug_source_narration` |

The blocked variants are preserved in `summary.json` under
`variant_summary`. The blocked blog candidate includes the exact debug-source
shape found in review ("In the uploaded tickets, 35 of 36 rows...") and was not
saved or exported. The exported blog samples were also checked for the known
debug-generation phrases `rows were included for generation`,
`uploaded ticket CSV can produce`, and `Your uploaded tickets contain`; none
were present.

## Samples

Landing-page exports:

| Variant | Draft id | Title | Hero headline | Brand voice |
|---|---|---|---|---|
| pain_led | `49e324c1-f331-404d-bfa4-dcbcb5387027` | "Stop Answering the Same Support Tickets Every Week" | "Your team answers the same questions every week" | passed |
| outcome_led | `2b09400f-a8b9-4787-86dd-ca655686d246` | "Cut Repeat Support Tickets with Review-Ready FAQ Answers" | "Turn repeat support tickets into approved FAQ answers" | passed |
| social_proof | `a1181e5e-91e3-49d0-9ad1-1c5494783330` | "35 Support Tickets Showed the Same 9 FAQ Gaps - Here's Your Audit" | "Your repeat tickets already told you what's missing" | passed |

Blog exports:

| Variant | Draft id | Title | Opening heading | Brand voice |
|---|---|---|---|---|
| outcome_led | `2f9ec4e3-475a-4046-9e01-182757d1de20` | "Support Ticket FAQ Gaps: What Your Repeat Tickets Reveal Before Renewal" | "What your repeat support questions show" | passed |
| social_proof | `44d56e20-d9c8-4626-8e39-e894cb8fd085` | "Support Ticket FAQ Gaps: What Your Repeat Tickets Reveal Before Renewal" | "What repeat support questions show" | passed |

Sales-brief exports:

| Variant | Draft id | Headline | Brief type | Brand voice |
|---|---|---|---|---|
| pain_led | `33044118-629f-491d-ada6-0aa3fc838222` | "Your RevOps lead hit a reporting wall before their board meeting. Export friction at renewal time is churn friction." | renewal | passed |
| outcome_led | `e71d3a76-e32a-4891-8879-e6b99d9e58af` | "Your RevOps lead needs attribution exports before the board meeting. Support ticket shows renewal-stage reporting gap." | renewal | passed |
| social_proof | `8744dc21-df4e-424e-9a9d-e5cada717886` | "RevOps lead needs attribution exports before board meeting. Support ticket signals reporting gap risk at renewal." | renewal | passed |

## Reviewer Notes

This run is meant to confirm the post-fix convergence signal against the #1360
failure baseline, not to certify final product quality. Review the exported
drafts directly for:

- publishable openings with no debug-source narration;
- variant distinction in title, body, and section order;
- grounded counts only;
- second-person voice;
- whether the one blocked blog variant should keep Gate A from passing.

The artifact set is complete enough for a reviewer to judge the real samples
without rerunning the live generation.
