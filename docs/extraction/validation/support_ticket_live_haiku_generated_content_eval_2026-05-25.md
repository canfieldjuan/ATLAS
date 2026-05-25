# Support-Ticket Live Haiku Generated Content Eval - 2026-05-25

## Scope

This validation proves the live Content Ops smoke can run support-ticket-backed
landing-page and blog-post generation with:

- the support-ticket CSV input provider
- saved-draft export
- deterministic generated-content evaluation
- OpenRouter routed to the Claude Haiku family for lower-cost testing

Ownership lane: `content-ops/support-ticket-input-provider`

## Environment

- Repo: `canfieldjuan/ATLAS`
- Branch: `claude/pr-support-ticket-live-haiku-eval`
- Source CSV:
  `extracted_content_pipeline/examples/support_ticket_sources.csv`
- Env files:
  - `/home/juan-canfield/Desktop/Atlas/.env`
  - `/home/juan-canfield/Desktop/Atlas/.env.local`
  - `tmp/support_ticket_live_haiku_eval_20260525/haiku.env`
- Haiku override:
  - `ATLAS_LLM_OPENROUTER_REASONING_MODEL=anthropic/claude-haiku-4-5`
  - `ATLAS_LLM__OPENROUTER_REASONING_MODEL=anthropic/claude-haiku-4-5`

The raw smoke outputs and saved-draft exports were written to ignored local
`tmp/support_ticket_live_haiku_eval_20260525/`.

## Commands

Landing page:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --account-id acct_support_ticket_live_haiku_eval_20260525_landing \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_live_haiku_eval_20260525/landing-page-draft.json \
  --output-result tmp/support_ticket_live_haiku_eval_20260525/landing-page-result.json \
  --evaluate-generated-content \
  --json
```

Blog post, after prompt/evaluator tightening:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_live_haiku_eval_20260525_blog_fixed4 \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_live_haiku_eval_20260525/blog-post-draft-fixed4.json \
  --output-result tmp/support_ticket_live_haiku_eval_20260525/blog-post-result-fixed4.json \
  --evaluate-generated-content \
  --json
```

## Landing Result

- Smoke status: passed
- Saved draft id: `cad10d59-d62a-4c04-a111-7036cfd73e0b`
- Generated: 1
- Skipped: 0
- SEO/AEO readiness: ready
- GEO readiness: ready
- Generated-content evaluation: passed
- Evaluator errors: none

The landing export carried source context for all 4 uploaded support-ticket
rows and surfaced both observed clusters:

- `email and profile updates` - 2 tickets
- `reporting friction` - 2 tickets

## Blog Result

The blog path generated and saved drafts, but the generated-content evaluator
correctly failed the latest run because the model invented unsupported future
impact percentages.

Latest run:

- Smoke status: failed by generated-content evaluation
- Saved draft id: `c0669850-b56a-48ac-9eff-7eb1a43582ea`
- Generated: 1
- Skipped: 0
- SEO/AEO readiness: ready
- GEO readiness in export: needs_review
- Generated-content evaluation: failed
- Evaluator error:
  `generated text contains percentage claims not backed by support-ticket source counts: 30-50%, 30-50%, 30-50%`

Earlier live blog runs exposed two additional issues that are now guarded:

- Unsupported uploaded-ticket timeframe language such as
  `Between May 2026 and the present`.
- Unsupported future impact percentages such as `20-40%`, `70%`, and `75%`.

## Code Changes From This Validation

The validation exposed gaps that could have produced a false green. This slice
closed those detection/source gaps:

- Undated support-ticket blog blueprints no longer include `report_date`.
- The blog-generation prompt now says uploaded-ticket inputs must use uploaded
  ticket language and must not invent calendar windows.
- The blog-generation prompt now tells the model not to invent predictive ROI,
  future ticket-reduction, time-savings, or customer-satisfaction numbers.
- The generated-content evaluator now fails unsupported calendar-window claims
  for undated uploaded tickets.
- The generated-content evaluator now fails unsupported percentage claims unless
  the percentage is directly source-backed by the support-ticket counts.
- The blog prompt now names the citable-section shape required by the GEO gate.

## Interpretation

The landing-page path is currently proven end to end with Haiku, saved-draft
export, readiness, and generated-content evaluation.

The blog-post path is not yet clean end to end. The smoke now catches the drift
instead of passing it silently, but Haiku still tends to invent future impact
percentages for support-ticket FAQ copy. That should be the next generator
slice: use the evaluator failure as repair feedback or tighten the
support-ticket blog prompt/input contract further until the saved draft passes
both export readiness and generated-content evaluation.

One additional integration gap was observed: the latest blog run saved a draft
whose exported `geo_readiness.status` was `needs_review`. A follow-up should
align blog save-time quality gating with the exported readiness checks, or make
the live smoke fail on exported readiness when quality gates are enabled.
