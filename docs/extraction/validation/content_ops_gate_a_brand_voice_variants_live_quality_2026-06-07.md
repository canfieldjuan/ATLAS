# Gate A Live Output-Quality Proof: Brand Voice + Variations

Date: 2026-06-07

Issue: #1357

Lane: `content-ops/live-output-quality-proof`

Verdict: **FAIL for product acceptance**.

The live execution/review/export path completed, but the generated output did
not clear the Gate A bar. Blog variants collapsed to one persisted draft, the
pain-led blog variant was blocked by quality gates, and two exported assets
missed the requested second-person brand voice.

Raw artifacts:

- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/execution-result.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/review-results.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/summary.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/export-landing_page.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/export-blog_post.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_brand_voice_variants_2026-06-07/export-sales_brief.json`

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
  --output-dir tmp/content_ops_gate_a_brand_voice_variants_20260607 \
  --variant-count 3 \
  --quality-repair-attempts 1 \
  --max-cost-usd 20.00 \
  --json
```

Runtime route:

- Real local Postgres: the existing Atlas DB pool from `.env` / `.env.local`.
- Real model route: `anthropic/claude-sonnet-4-5`.
- Local Ollama fallback disabled by `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false`.
- Source material: `support_ticket_saas_demo_sources.csv`, 36 support-ticket rows.
- Brand profile: inline `Sharp Support Operator`, second-person, concise, banned terms enabled.
- Outputs requested in one execution request: `landing_page`, `blog_post`, `sales_brief`.
- Variant request: `variant_count=3`.

## Technical Result

The harness completed execution, review, and exact export:

| Output | Saved ids | Exported rows | Review status |
|---|---:|---:|---|
| landing_page | 3 | 3 | approved, no missing ids |
| blog_post | 2 entries, 1 unique id | 1 | approved, duplicate id |
| sales_brief | 3 | 3 | approved, no missing ids |

The summary file now reports `ok=false` because the stricter harness treats the
blog duplicate-id case as a structural variant-persistence failure:
`blog_post variant persistence collapsed: 2 successful variant(s), 2 saved id
entries, 1 unique saved id(s), 1 exported row(s)`.

Budget behavior:

- Correct final plan estimated `12.60` USD in the execution preview.
- The final `--max-cost-usd 20.00` cap did not trigger.
- An earlier low-cap run with `--max-cost-usd 3.00` blocked before generation
  with the expected cost warning; that run was not used as the final proof.

All-variants-fail / no-output behavior:

- Before the blog blueprint was seeded, blog variants returned zero saved ids
  with the warning `No blog variants generated; all requested variants were blocked or skipped.`
- In the final run, the pain-led blog candidate was blocked by quality gates
  for `aeo_structure_missing`, `geo_answer_first_sections_missing`, and
  `geo_citable_section_structure_missing`.

## Samples

Landing page variants were the least broken output, but not a product pass. The
three exported hero headlines were distinct and all passed the brand-voice
audit:

| Variant | Draft id | Hero headline |
|---|---|---|
| Pain-led | `a04230f4-54b1-4929-aa3f-98c763844335` | "Your team keeps answering the same questions" |
| Outcome-led | `f5746d35-502d-44ac-8c92-737fb4ab53ec` | "Turn repeat support tickets into approved FAQ answers" |
| Social-proof-led | `998f8272-f077-4e82-97f0-afbb27512f1d` | "Turn Repeat Support Tickets Into Approved FAQ Answers" |

Blog post exported one approved row:

- Draft id: `9c2cdf6c-9fbf-42db-8af8-6a59e850cf16`
- Title: `Support-Ticket Questions Customers Keep Asking`
- Voice audit: failed with `preferred_pov_second_person_not_detected`
- Grounding excerpt: "The uploaded CSV contains 36 support-ticket rows... 35 rows include direct customer questions... 9 repeated topics..."

The two successful blog variants both returned the same saved id. The persisted
draft metadata kept the social-proof variant angle, so the outcome-led result
did not survive as a distinct review/export item. The approved prose is also not
publishable: it opens with internal source narration rather than reader-facing
marketing copy.

Sales brief variants exported three rows:

| Variant | Draft id | Headline | Voice audit |
|---|---|---|---|
| Pain-led | `9d41714b-d3d7-4951-b3c2-7a4f0a9b6a8d` | "RevOps lead needs attribution exports before board meeting. Current tooling blocks critical reporting workflow." | passed |
| Outcome-led | `7843150e-413c-48f7-b06f-fed8f58db90a` | "RevOps lead needs attribution exports before board meeting. Current tooling can't deliver. Renewal window opens next quarter." | passed |
| Social-proof-led | `6451290c-ffd6-4019-bd22-b55626b21002` | "RevOps lead blocked on attribution export ahead of board meeting. Support ticket shows reporting gap under time pressure." | failed second-person POV |

Sales briefs were grounded to `saas-demo-001` and did not fabricate the named
account/contact. However, all three stored `brief_type=pre_call` even though
the request supplied `inputs.brief_type=renewal`.

## Human Review

Grounding held only on a tidy fixture. Landing pages and blog content used the
36 ticket rows, the 35 question-like rows, and the nine observed clusters, but
the fixture is clean and evenly bucketed: nine clusters with four tickets each.
That proves the live path, not grounding resilience on noisy real tickets.
Sales briefs stayed on the first ticket (`saas-demo-001`) and its
account/contact fields.

Brand voice was only partially applied. Landing pages matched the profile well:
plain, short, second-person, and no banned terms. The blog row and one sales
brief failed the automated second-person audit.

Variants were not reliable enough. Sales-brief variants were meaningfully
different. Landing-page variants mostly changed the hero while sharing the same
title and much of the same body, including repeated support-evidence phrasing.
Blog variants did not pass: one blocked, and two successful variants collapsed
into the same persisted row.

Overall quality verdict: **FAIL**. The system can generate, review, and export
live assets, but Gate A should not proceed as "validated" until blog variants
persist as distinct review/export items and brand voice misses are repaired or
surfaced as blocking output-quality failures.

## Parked Follow-Ups

Added to `HARDENING.md`:

- Blog post variants collapse to one persisted draft id.
- Brand-voice second-person guidance is not consistently honored.
- Sales brief live generation drifts from requested renewal brief type.
- Landing-page variants pass audits but are not meaningfully distinct.
- Blog output uses debug-style source narration instead of publishable prose.
- Gate A needs a messy-ticket grounding rerun.
