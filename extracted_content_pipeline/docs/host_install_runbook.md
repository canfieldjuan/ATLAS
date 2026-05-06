# AI Content Ops Host Install Runbook

Date: 2026-05-04

This runbook is the standalone database-backed path for the extracted AI
Content Ops campaign product. It assumes a host application owns Postgres,
provider credentials, tenant/auth policy, and any reasoning producer. The
content package owns migrations, customer opportunity normalization, prompt
lookup, generation orchestration, and draft persistence.

## What This Installs

The flow below gives a host app a working campaign-generation loop:

1. Apply packaged SQL migrations.
2. Load customer opportunity data from JSON or CSV.
3. Optionally attach host-produced reasoning context.
4. Optionally override packaged prompt skills.
5. Generate campaign drafts through the DB-backed runner.
6. Inspect, approve, queue, and send persisted drafts in `b2b_campaigns`.

It does not install an API server, auth layer, dashboard, or hosted background
worker. Sending is available through the host-run CLI below; long-running
scheduling and hosted orchestration remain host-owned integration points.

## Prerequisites

- Python environment with this repo on `PYTHONPATH`.
- Postgres database reachable through `EXTRACTED_DATABASE_URL` or `DATABASE_URL`.
- `asyncpg` installed in the host environment for DB-backed commands.
- UUID generation support for `gen_random_uuid()`. On older Postgres versions,
  enable `pgcrypto` before running migrations:

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

- LLM provider configuration if using the real product adapter:
  `EXTRACTED_CAMPAIGN_LLM_*` environment variables.
- No reasoning runtime is required. Reasoning is optional host input through
  `CampaignReasoningContextProvider` or a JSON file.

## Step 1: Configure Database Access

Use a product-specific DSN when possible:

```bash
export EXTRACTED_DATABASE_URL="postgres://user:password@localhost:5432/content_ops"
```

If `EXTRACTED_DATABASE_URL` is not set, the migration, import, and generation
commands fall back to `DATABASE_URL`. Use `--database-url` when the host app
keeps AI Content Ops data in a separate database.

## Step 2: Apply Migrations

Preview pending migrations:

```bash
python scripts/run_extracted_content_pipeline_migrations.py --dry-run
```

Apply them:

```bash
python scripts/run_extracted_content_pipeline_migrations.py
```

The runner writes applied migration filenames and checksums to
`content_pipeline_schema_migrations`. Re-running the command skips files already
recorded in that table.

Use a custom metadata table only if the host has a naming convention:

```bash
python scripts/run_extracted_content_pipeline_migrations.py \
  --migration-table customer_content_pipeline_migrations
```

## Step 3: Validate Customer Data Offline

Before writing to Postgres, validate the same JSON or CSV file through the file
adapter:

```bash
python scripts/run_extracted_campaign_generation_example.py \
  customer_opportunities.csv \
  --format csv \
  --channels email_cold,email_followup
```

This proves the opportunity rows can normalize into prompt-ready fields without
requiring a database or provider credentials.

Minimum useful columns:

| Column | Purpose |
|---|---|
| `company` / `company_name` | Account or buyer name. |
| `vendor` / `vendor_name` | Incumbent or target vendor. |
| `email` / `contact_email` | Recipient email and stable target key. |
| `title` / `contact_title` | Recipient role context. |
| `pain_category` / `pain_points` | Prompt-visible customer pains. |
| `competitor` / `competitors` | Alternative vendor context. |
| `opportunity_score` | Ranking input. |
| `urgency_score` | Ranking input and default ordering. |
| `evidence` | Optional source snippets, facts, or proof rows. |

Unknown columns are preserved in draft metadata and prompt context through the
normalized opportunity payload.

## Step 4: Load Opportunities Into Postgres

Preview the import:

```bash
python scripts/load_extracted_campaign_opportunities.py \
  customer_opportunities.csv \
  --format csv \
  --account-id acct_123 \
  --dry-run
```

Write rows:

```bash
python scripts/load_extracted_campaign_opportunities.py \
  customer_opportunities.csv \
  --format csv \
  --account-id acct_123
```

Imports are append-only by default. For repeatable test loads, replace matching
target ids for the same account and target mode:

```bash
python scripts/load_extracted_campaign_opportunities.py \
  customer_opportunities.csv \
  --format csv \
  --account-id acct_123 \
  --replace-existing
```

`--replace-existing` only deletes rows with matching target ids inside the
selected `account_id` and `target_mode`; it does not truncate the table.

Amazon seller installs can refresh broad category intelligence from review data
before preparing opportunities:

```bash
python scripts/refresh_extracted_seller_category_intelligence.py \
  --category supplements \
  --min-reviews 50
```

Then prepare opportunities from active seller targets and cached category
intelligence snapshots:

```bash
python scripts/prepare_extracted_seller_campaign_opportunities.py \
  --account-id acct_123 \
  --category supplements \
  --replace-existing
```

The seller preparation command reads `seller_targets` and the latest
`category_intelligence_snapshots` row per category, then writes normalized
`campaign_opportunities` rows with `target_mode="amazon_seller"`.

## Step 5: Add Optional Reasoning Context

If a host already has account reasoning output, pass it as JSON:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --reasoning-context extracted_content_pipeline/examples/campaign_reasoning_context.json
```

The file-backed reasoning adapter matches rows by target id, company, email, or
vendor. The generator still works without this file, but output quality is lower
because prompts only see the opportunity row.

If a host does not already have reasoning JSON, it can configure
`SinglePassCampaignReasoningProvider` from
`extracted_content_pipeline.services.single_pass_reasoning_provider`. The
provider uses the same LLM and skill ports as campaign generation, calls the
packaged `digest/b2b_campaign_reasoning_context` prompt once per opportunity,
and returns the same normalized context shape. This improves specificity
without importing Atlas reasoning producers or long-running graph state.

The Postgres generation runner wires that provider directly when the product
LLM adapter is configured:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --single-pass-reasoning
```

Use `--reasoning-skill-name`, `--reasoning-max-tokens`, and
`--reasoning-temperature` to tune the provider, or `--skills-root` to override
the packaged reasoning prompt.

See `reasoning_handoff_contract.md` for the accepted shape and the no-direct-
import rule. AI Content Ops consumes compressed reasoning; it does not import a
reasoning engine.

## Step 6: Add Optional Prompt Overrides

Use packaged skills by default. To override copy strategy, provide a markdown
skill root:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --skills-root customer_skills
```

Files with the same skill name override bundled prompts. Missing prompts fall
back to `extracted_content_pipeline/skills/digest/*.md`.

## Step 7: Generate Drafts

Generate cold email drafts:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --limit 10
```

Generate cold email plus follow-up drafts:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --channels email_cold,email_followup \
  --limit 10
```

The runner reads active `campaign_opportunities`, builds drafts through
`CampaignGenerationService`, and persists them to `b2b_campaigns`.

For Amazon seller drafts, use the seller target mode after the preparation step:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --target-mode amazon_seller \
  --channels email_cold,email_followup \
  --limit 10
```

## Step 8: Verify Output

Inspect imported opportunities:

```sql
SELECT target_id, company_name, vendor_name, urgency_score, updated_at
FROM campaign_opportunities
WHERE account_id = 'acct_123'
ORDER BY urgency_score DESC NULLS LAST
LIMIT 20;
```

Inspect generated drafts:

```sql
SELECT id, company_name, vendor_name, status, created_at
FROM b2b_campaigns
WHERE metadata -> 'scope' ->> 'account_id' = 'acct_123'
ORDER BY created_at DESC
LIMIT 20;
```

Inspect draft metadata for source opportunity and reasoning context:

```sql
SELECT metadata
FROM b2b_campaigns
WHERE metadata -> 'scope' ->> 'account_id' = 'acct_123'
ORDER BY created_at DESC
LIMIT 1;
```

Or export generated drafts through the product CLI:

```bash
python scripts/export_extracted_campaign_drafts.py \
  --account-id acct_123 \
  --limit 20

python scripts/export_extracted_campaign_drafts.py \
  --account-id acct_123 \
  --format csv \
  --output campaign_drafts.csv
```

Approve or queue selected drafts after review:

```bash
python scripts/review_extracted_campaign_drafts.py \
  <campaign-id> \
  --account-id acct_123 \
  --status approved

python scripts/review_extracted_campaign_drafts.py \
  <campaign-id> \
  --account-id acct_123 \
  --status queued \
  --from-status draft,approved \
  --from-email audit@customer.com
```

Or mount the B2B draft API router in a host FastAPI app and inject tenant
scope/auth from the host application:

```python
from fastapi import Depends

from extracted_content_pipeline.api.b2b_campaigns import create_b2b_campaign_router


app.include_router(
    create_b2b_campaign_router(
        pool_provider=get_pool,
        scope_provider=current_tenant_scope,
        dependencies=[Depends(require_content_ops_user)],
    )
)
```

The router exposes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/b2b/campaigns/drafts` | List scoped drafts as JSON. |
| `GET` | `/b2b/campaigns/drafts/export` | Export scoped drafts as CSV or JSON. |
| `POST` | `/b2b/campaigns/drafts/review` | Approve, queue, cancel, or expire selected drafts. |

Amazon seller installs can mount the seller-specific router:

```python
from fastapi import Depends

from extracted_content_pipeline.api.seller_campaigns import create_seller_campaign_router


app.include_router(
    create_seller_campaign_router(
        pool_provider=get_pool,
        scope_provider=current_tenant_scope,
        dependencies=[Depends(require_content_ops_user)],
    )
)
```

It exposes seller target CRUD under `/seller/targets`, hosted refresh and
opportunity preparation triggers under `/seller/intelligence/refresh`,
`/seller/opportunities/prepare`, and
`/seller/operations/refresh-and-prepare`, plus seller draft list/export/review
routes under `/seller/campaigns/drafts`. Seller draft review is guarded to
`target_mode="amazon_seller"` so the route cannot update other campaign
products by id.

Example hosted operation payload:

```json
{
  "category": "supplements",
  "min_reviews": 50,
  "replace_existing": true
}
```

Send queued drafts through the configured provider:

```bash
export EXTRACTED_RESEND_API_KEY="re_..."
python scripts/send_extracted_campaigns.py \
  --provider resend \
  --default-from-email audit@customer.com \
  --limit 10
```

Ingest Resend webhook events after provider delivery, open, click, bounce,
complaint, or unsubscribe callbacks:

```bash
export EXTRACTED_RESEND_WEBHOOK_SECRET="whsec_..."
python scripts/ingest_extracted_campaign_webhook.py \
  --body-file resend-event.json \
  --headers-json resend-headers.json \
  --json
```

Or mount the router in a host FastAPI app and inject the host database and
secret providers:

```python
from extracted_content_pipeline.api.campaign_webhooks import (
    create_campaign_webhook_router,
)
from extracted_content_pipeline.campaign_send import verify_unsubscribe_token as verify_token


async def verify_unsubscribe_token(email: str, token: str) -> bool:
    return verify_token(
        email,
        token,
        get_unsubscribe_token_secret(),
    )


app.include_router(
    create_campaign_webhook_router(
        pool_provider=get_pool,
        signing_secret_provider=get_resend_webhook_secret,
        unsubscribe_token_verifier=verify_unsubscribe_token,
    )
)
```

The unsubscribe route accepts both browser `GET` links and RFC 8058 one-click
`POST` requests. Keep the default token requirement enabled and issue tokens in
the host send path so public URLs cannot unsubscribe arbitrary recipients. Set
the same secret on `CampaignSendConfig.unsubscribe_token_secret` or
`EXTRACTED_CAMPAIGN_UNSUBSCRIBE_TOKEN_SECRET` and in the verifier above.

Refresh analytics after sends or webhook ingestion:

```bash
python scripts/refresh_extracted_campaign_analytics.py --json
```

Progress due sequences and queue generated follow-up drafts:

```bash
python scripts/progress_extracted_campaign_sequences.py \
  --from-email audit@customer.com \
  --limit 10
```

Add `--visibility-jsonl /var/log/content-ops/campaign-events.jsonl` to the
generation, send, sequence progression, or analytics CLIs when the host wants a
file-backed operation audit trail without mounting the FastAPI operations
router. Inspect the same file with:

```bash
python scripts/read_extracted_campaign_visibility.py \
  /var/log/content-ops/campaign-events.jsonl \
  --json \
  --limit 20
```

Or mount the hosted operations router in a FastAPI app for admin-triggered
draft generation, send, sequence progression, and analytics refresh actions:

```python
from fastapi import Depends

from extracted_content_pipeline.api.campaign_operations import (
    CampaignOperationsApiConfig,
    create_campaign_operations_router,
)


app.include_router(
    create_campaign_operations_router(
        pool_provider=get_pool,
        sender_provider=get_campaign_sender,
        llm_provider=get_campaign_llm,
        skills_provider=get_campaign_skills,
        reasoning_context_provider=get_campaign_reasoning_context,
        config=CampaignOperationsApiConfig(
            send_default_from_email="audit@customer.com",
            sequence_from_email="audit@customer.com",
        ),
        dependencies=[Depends(require_content_ops_admin)],
    )
)
```

The hosted operations payloads are intentionally narrow. Callers may set
tenant scope, target/channel, filters, `limit`, and, for sequence progression,
`max_steps`; provider credentials, sender identity, unsubscribe policy, LLM
client, skill roots, and reasoning providers stay host-configured.

Inject `visibility_provider` when the host wants admin UI telemetry. The four
POST operation routes emit best-effort `campaign_operation_started`,
`campaign_operation_completed`, and `campaign_operation_failed` events through
the `VisibilitySink` port. Sink failures are logged and do not change operation
responses.

For a file-backed audit trail without writing a custom sink first:

```python
from extracted_content_pipeline.campaign_visibility import JsonlVisibilitySink

visibility = JsonlVisibilitySink("/var/log/content-ops/campaign-events.jsonl")

app.include_router(
    create_campaign_operations_router(
        pool_provider=get_pool,
        sender_provider=get_campaign_sender,
        visibility_provider=lambda: visibility,
    )
)
```

For lightweight hosted installs without a separate reasoning provider, set
`generation_single_pass_reasoning=True` on `CampaignOperationsApiConfig`. The
draft generation route then builds `SinglePassCampaignReasoningProvider` from
the injected LLM and skill providers before calling the Postgres generation
runner. Explicit `reasoning_context_provider` injection still takes precedence.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/campaigns/operations/status` | Report database availability, provider presence, feature readiness, and configured limits for admin dashboards. |
| `POST` | `/campaigns/operations/drafts/generate` | Generate and persist campaign drafts from `campaign_opportunities`. |
| `POST` | `/campaigns/operations/send/queued` | Send queued campaign rows through the injected sender. |
| `POST` | `/campaigns/operations/sequences/progress` | Generate and queue due follow-up sequence steps. |
| `POST` | `/campaigns/operations/analytics/refresh` | Refresh packaged campaign analytics aggregates. |

For B2B installs, mount this router beside `create_b2b_campaign_router` and run
the hosted admin sequence in order:

1. Check `/campaigns/operations/status` so the admin UI only enables ready
   actions.
2. Generate drafts from active opportunities with
   `/campaigns/operations/drafts/generate`.
3. Inspect generated rows with `/b2b/campaigns/drafts` or
   `/b2b/campaigns/drafts/export`.
4. Approve and queue selected rows with `/b2b/campaigns/drafts/review`.
5. Send queued rows with `/campaigns/operations/send/queued`.
6. Refresh reporting with `/campaigns/operations/analytics/refresh`.

Reject a draft without deleting it:

```bash
python scripts/review_extracted_campaign_drafts.py \
  <campaign-id> \
  --account-id acct_123 \
  --status cancelled \
  --reason "customer rejected"
```

## Retry And Rollback Notes

- Migrations are idempotent by filename. Re-running the migration command skips
  recorded files.
- Opportunity imports are append-only unless `--replace-existing` is set.
- Draft review updates require explicit campaign ids and default to rows still
  in `draft` status. Use `--from-status draft,approved` when moving an
  approved draft to the send queue.
- For test tenants, deleting rows by `account_id` is the cleanest reset:

```sql
DELETE FROM b2b_campaigns WHERE metadata -> 'scope' ->> 'account_id' = 'acct_123';
DELETE FROM campaign_opportunities WHERE account_id = 'acct_123';
```

- If generation produces no drafts, verify active opportunities exist for the
  same `account_id` and `target_mode`, then run with a low `--limit`.
- If generated drafts lack buyer specificity, add reasoning JSON or richer
  opportunity evidence before tuning prompts.

## Current Limits

- Dashboard auth and long-running hosted workers are not part of this install
  path yet. Host apps inject auth dependencies into the packaged routers.
- The runbook covers campaign opportunity generation, not blog generation or
  vendor briefing delivery.
- Long-running reasoning production remains host-owned. The product includes a
  single-pass opportunity-level provider for lightweight installs, but it does
  not compute graph state or multi-hop synthesis.
