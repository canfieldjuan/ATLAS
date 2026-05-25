# Support-Ticket Live Blog Gate Validation - 2026-05-25

## Scope

This validation proves the merged support-ticket blog generated-content gate
works on the live Content Ops path:

- support-ticket CSV input provider
- seeded support-ticket blog blueprint
- pipeline-routed OpenRouter Haiku model
- saved blog draft persistence
- exact saved-draft export
- deterministic generated-content evaluation

Ownership lane: `content-ops/support-ticket-input-provider`

## Environment

- Repo: `canfieldjuan/ATLAS`
- Branch: `claude/pr-support-ticket-live-blog-gate-validation`
- Source CSV:
  `extracted_content_pipeline/examples/support_ticket_sources.csv`
- Env files:
  - `/home/juan-canfield/Desktop/Atlas/.env`
  - `/home/juan-canfield/Desktop/Atlas/.env.local`
  - `tmp/support_ticket_live_blog_gate_20260525/haiku.env`
- Haiku override:
  - `ATLAS_LLM_OPENROUTER_REASONING_MODEL=anthropic/claude-haiku-4-5`
  - `ATLAS_LLM__OPENROUTER_REASONING_MODEL=anthropic/claude-haiku-4-5`

The raw smoke outputs and saved-draft exports were written to ignored local
`tmp/support_ticket_live_blog_gate_20260525/`.

## Commands

Initial live blog smoke:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_live_blog_gate_20260525 \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_blog_gate_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_live_blog_gate_20260525/blog-post-draft.json \
  --output-result tmp/support_ticket_live_blog_gate_20260525/blog-post-result.json \
  --evaluate-generated-content \
  --json
```

Evaluator rerun after adding the cadence check:

```bash
python scripts/evaluate_support_ticket_generated_content.py \
  --output blog_post \
  tmp/support_ticket_live_blog_gate_20260525/blog-post-draft.json
```

Final live blog smoke after the cadence fix:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_live_blog_gate_20260525_cadence2 \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_blog_gate_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_live_blog_gate_20260525/blog-post-draft-cadence2.json \
  --output-result tmp/support_ticket_live_blog_gate_20260525/blog-post-result-cadence2.json \
  --evaluate-generated-content \
  --json
```

## Initial Result

- Smoke status under the old evaluator: passed
- Saved draft id: `a90aeb06-8b71-4a02-b31a-4c329503885f`
- Generated: 1
- Skipped: 0
- SEO/AEO readiness: ready
- GEO readiness: ready
- Generated-content evaluation under the old checker: passed

The saved draft carried the expected support-ticket context:

- `source_period`: `Uploaded support tickets`
- `source_row_count`: 4
- `included_ticket_row_count`: 4
- `question_like_ticket_count`: 2
- top clusters:
  - `email and profile updates`: 2
  - `reporting friction`: 2

However, the generated copy converted undated uploaded tickets into cadence
language, including `per week`. Because the CSV does not contain ticket dates
or a weekly observation window, that was a false green.

## Cadence Fix

The evaluator now adds an `uploaded_ticket_cadence_truthful` check for neutral
uploaded-ticket sources. It fails recurring cadence terms such as `per week`,
`weekly`, `per month`, and `monthly` when the source period is only
`Uploaded support tickets` / `uploaded tickets`.

Rerunning the updated evaluator against the initial saved draft failed as
expected:

- Evaluation status: failed
- Error:
  `generated text claims a recurring cadence for an undated uploaded-ticket source: per week`

The blog prompt was also tightened so uploaded-ticket posts do not invent
recurring cadence language unless that cadence exists in `data_context`.

## Final Result

- Smoke status: passed
- Saved draft id: `ea0a3333-e952-4cbd-9f9b-57c9825b3470`
- Generated: 1
- Skipped: 0
- SEO/AEO readiness: ready
- GEO readiness: ready
- Generated-content evaluation: passed
- Evaluator errors: none

Generated-content checks:

- support-ticket context present: passed
- generated text present: passed
- support-ticket framing present: passed
- no stale benchmark numbers: passed
- source count visible: passed
- uploaded-ticket timeframe truthful: passed
- uploaded-ticket cadence truthful: passed
- percentage claims source-backed: passed
- source signal visible: passed

The final saved draft preserved the source counts and customer language without
inventing a calendar window, rolling window, or recurring cadence.

## Interpretation

The support-ticket blog path is now validated end to end with Haiku after #961:
the real route packages support-ticket rows, injects trusted blog
`data_context`, generates through the live LLM route, persists a draft, exports
the exact saved row, and passes deterministic generated-content evaluation.

The validation also closed a false-green class: undated uploaded-ticket data can
no longer be translated into weekly/monthly cadence claims without being caught
by the evaluator.
