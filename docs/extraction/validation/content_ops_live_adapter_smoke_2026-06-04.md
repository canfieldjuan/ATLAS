# Content Ops Live Adapter Smoke - 2026-06-04

Ownership lane: content-ops/marketer-reviews-as-input-live-smoke-gate

## Environment

- Worktree: `/home/juan-canfield/Desktop/Atlas/worktrees/content-ops-live-adapter-smoke`
- Database: `localhost:5433/atlas` as user `atlas`
- Env files loaded: `/home/juan-canfield/Desktop/Atlas/.env`, `/home/juan-canfield/Desktop/Atlas/.env.local`
- Local model fallback: `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false`
- Configured model route: OpenRouter, `anthropic/claude-sonnet-4-5`
- Primary smoke account: `2b2b950d-f64b-4852-bc30-f92a34cdf169`
- Second tenant isolation account: `e955ae18-903f-4604-9112-40345a80efe1`

## Migrations

The configured Postgres initially had 8 pending Content Ops migrations:

- `067_b2b_campaigns_updated_at.sql`
- `328_ticket_faq_macro_writebacks.sql`
- `329_ticket_faq_macro_writebacks_pending.sql`
- `330_ticket_faq_macro_publish_attempts.sql`
- `331_social_posts.sql`
- `332_ad_copy_drafts.sql`
- `333_quote_card_drafts.sql`
- `334_stat_card_drafts.sql`

Applied with the packaged `apply_content_pipeline_migrations(pool)` runner.
Post-apply dry-run reported 0 pending migrations, 37 skipped already-applied
migrations, and versions 331-334 recorded in
`content_pipeline_schema_migrations`.

## Blog Post LLM Smoke

Command:

```bash
EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false \
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id 2b2b950d-f64b-4852-bc30-f92a34cdf169 \
  --user-id 11111111-1111-4111-8111-111111111111 \
  --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --export-saved-draft tmp/content_ops_live_adapter_smoke_20260604/blog-post-draft-uuid.json \
  --output-result tmp/content_ops_live_adapter_smoke_20260604/blog-post-result-uuid.json \
  --evaluate-generated-content \
  --json
```

Observed:

- Exit status: 0
- Saved blog draft id: `9c2cdf6c-9fbf-42db-8af8-6a59e850cf16`
- Seeded blueprint id: `0da54d85-706c-43e9-aa59-036b182b2d87`
- Generated title: `Support-Ticket Questions Customers Keep Asking: 36 Tickets, 9 Clusters, 6 Draft FAQ Shells`
- Draft status: `draft`
- Model recorded in draft metadata: `anthropic/claude-sonnet-4-5`
- Generated content evaluator: `ok=true`, `errors=[]`, 11/11 checks passed
- Draft metadata usage: 18,577 input tokens, 5,455 output tokens, 8,783 cache write tokens, 9,788 cached tokens, 6 billable input tokens

Durable `llm_usage` rows for the smoke account:

| Provider | Model | Endpoint | Input | Output | Total | Cost USD | Status |
|---|---|---|---:|---:|---:|---:|---|
| openrouter | anthropic/claude-sonnet-4-5 | `https://openrouter.ai/api/v1/chat/completions` | 7,710 | 2,771 | 10,481 | 0.053591 | completed |
| openrouter | anthropic/claude-sonnet-4-5 | `https://openrouter.ai/api/v1/chat/completions` | 10,867 | 2,684 | 13,551 | 0.064125 | completed |

Totals: 18,577 input tokens, 5,455 output tokens, 24,032 total tokens,
`0.117716` USD.

## Stat Card Smoke

Command:

```bash
EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false \
python scripts/smoke_content_ops_live_generation.py \
  --output stat_card \
  --account-id 2b2b950d-f64b-4852-bc30-f92a34cdf169 \
  --user-id 11111111-1111-4111-8111-111111111111 \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --export-saved-draft tmp/content_ops_live_adapter_smoke_20260604/stat-card-draft.json \
  --output-result tmp/content_ops_live_adapter_smoke_20260604/stat-card-result.json \
  --json
```

Observed:

- Exit status: 0
- Saved stat-card id: `91e74867-12e1-4113-9bd7-9472eab1aa84`
- Claim: `NPS score: 42`
- Metric label/display: `NPS score` / `42`
- Evidence: `NPS score dropped to 42 after renewal because customers could not find clear export and account-change answers.`
- Exact saved-draft export returned one row and filtered to that saved id.

## Review, Export, And Tenant Isolation

Direct SQL confirmed:

- Blog draft visible to account A: true
- Blog draft visible to account B: false
- Stat card visible to account A: true
- Stat card visible to account B: false
- Wrong-account stat-card status update command: `UPDATE 0`
- Final stat-card status: `approved`

The generated-assets FastAPI router was also exercised through ASGI with the
real asyncpg pool:

- Account B `POST /content-assets/stat_card/drafts/review` returned `updated=false`.
- Account A `POST /content-assets/stat_card/drafts/review` rejected the row, then approved it again with `updated=true`.
- Account A `GET /content-assets/stat_card/drafts?status=approved` returned the generated id.
- Account A JSON export returned the generated id.
- Account A HTML export returned `text/html; charset=utf-8`, filename `content_assets_stat_card.html`, and contained both `NPS score: 42` and the `visual-card stat-card` article.
- Account B approved list returned count 0 and did not include the generated id.

## Browser Rendering

The stat-card HTML export was rendered through headless Chromium with
Playwright and captured to:

- `tmp/content_ops_live_adapter_smoke_20260604/stat-card-visual.html`
- `tmp/content_ops_live_adapter_smoke_20260604/stat-card-visual.png`

Observed:

- PNG bytes: 44,166
- PNG header: `89504e470d0a1a0a`
- Rendered card count: 1
- Browser page title: `Stat Cards Visual Export`
- Visual inspection: card rendered coherently with metric `42`, claim
  `NPS score: 42`, supporting text, evidence, source, and tag visible.

## Local Checks

```bash
python -m pytest tests/test_smoke_content_ops_live_generation.py -q
python -m py_compile scripts/smoke_content_ops_live_generation.py tests/test_smoke_content_ops_live_generation.py
git diff --check
```

Results:

- `tests/test_smoke_content_ops_live_generation.py`: 38 passed
- `py_compile`: passed
- `git diff --check`: passed

## Notes

An initial blog-post smoke used a non-UUID account id and completed generation,
but trace persistence warned because `llm_usage.account_id` is UUID-typed. The
primary evidence above is the rerun with a real UUID account and clean durable
trace rows.

#1300 remains untouched and unmerged in this slice.
