# Extracted Content Pipeline (Staging Copy)

This directory is an additive extraction scaffold copied from `atlas_brain`.
It is intentionally kept side-by-side with Atlas so pipeline logic can be
carved out safely without removing or changing production code.

## Current contents

- `autonomous/tasks/`: copied task implementations
- `services/`: copied support shims and staged service dependencies
- `skills/digest/`: copied prompt skill contracts, including campaign and
  sequence prompts used by the standalone services
- `storage/migrations/`: copied persistence migrations
- `docs/`: extraction maps for productized pipeline slices

## Sync command

To refresh this scaffold from Atlas source of truth:

```bash
bash scripts/sync_extracted_content_pipeline.sh
```

This stable product entry point delegates to
`extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline`.

## Manifest

Mirror mappings are declared in `extracted_content_pipeline/manifest.json` so sync and validation use one source of truth.

## Scope

This scaffold preserves code exactly as copied so behavior and signatures remain
unchanged while extraction work continues.

The standalone audit now passes with zero runtime `atlas_brain` imports. The
scaffold is still a staging boundary until the minimal runtime adapters are
hardened into customer-grade ports and the copied helper surface is trimmed to
the sellable workflows.


## Validation command

```bash
bash scripts/validate_extracted_content_pipeline.sh
```

This stable product entry point delegates to
`extracted/_shared/scripts/validate_extracted.sh extracted_content_pipeline`.

## ASCII compliance check

```bash
bash scripts/check_ascii_python.sh
```

This stable product entry point delegates to
`extracted/_shared/scripts/check_ascii_python.sh extracted_content_pipeline`.

## Import debt check

```bash
python scripts/check_extracted_imports.py
```

This stable product entry point delegates to
`extracted/_shared/scripts/check_extracted_imports.py extracted_content_pipeline`.
Known unresolved relative imports are tracked in `extracted_content_pipeline/import_debt_allowlist.txt`.

## Standalone readiness audit

```bash
python scripts/audit_extracted_standalone.py
python scripts/audit_extracted_standalone.py --fail-on-debt
```

The first command reports Atlas runtime coupling. The second is the product gate
for keeping the extracted package free of runtime `atlas_brain` imports.

## One-shot checks

```bash
bash scripts/run_extracted_pipeline_checks.sh
```

The one-shot runner enforces the standalone readiness audit with
`--fail-on-debt`; any new runtime `atlas_brain` import fails CI.

## Compatibility shims

To keep copied task modules importable inside this repo, package-level bridge modules are provided under `extracted_content_pipeline/` (for example `config.py`, `storage/database.py`, `pipelines/llm.py`, and `services/*`). Runtime imports no longer delegate to `atlas_brain`; most adapters are intentionally minimal local implementations.

B2B helper siblings required by `b2b_blog_post_generation.py` are also copied into `extracted_content_pipeline/autonomous/tasks/`.

These minimal adapters are extraction scaffolding. They need hardening before
shipping in the customer product.

The email/campaign generation slice is mapped in `docs/email_campaign_generation_pipeline.md`, with standalone productization requirements in `docs/standalone_productization.md`.

Reasoning is a host/product boundary, not copied into AI Content Ops. The
campaign generator consumes already-compressed reasoning through
`CampaignReasoningContextProvider`; see
`docs/reasoning_handoff_contract.md` for the accepted context shape and the
no-direct-import rule.

`campaign_opportunities.py` defines the customer-data contract for campaign
generation. Hosts can pass loose opportunity rows from a CRM, warehouse, or
vendor-intelligence feed; the product normalizes them into stable prompt and
draft metadata fields while preserving custom columns.

`campaign_customer_data.py` adds JSON/CSV file adapters and a
`FileIntelligenceRepository` so hosts can run the generator directly from
customer exports before wiring a database integration.

## Campaign generation example

Run the standalone campaign generator against the included customer-data
payload:

```bash
python scripts/run_extracted_campaign_generation_example.py
```

Or run it against a customer JSON file and write drafts to disk:

```bash
python scripts/run_extracted_campaign_generation_example.py customer_payload.json --output campaign_drafts.json
```

CSV exports work too. The loader normalizes common CRM/warehouse columns such
as `company`, `vendor`, `email`, `title`, `pain_category`, `competitor`,
`opportunity_score`, and `urgency_score`, while preserving custom columns in
draft metadata:

```bash
python scripts/run_extracted_campaign_generation_example.py customer_opportunities.csv --format csv
```

Generate cold-email and follow-up drafts for each opportunity by passing
channels:

```bash
python scripts/run_extracted_campaign_generation_example.py --channels email_cold,email_followup
```

Pass host-provided reasoning context without installing a reasoning engine:

```bash
python scripts/run_extracted_campaign_generation_example.py \
  --reasoning-context extracted_content_pipeline/examples/campaign_reasoning_context.json
```

`campaign_reasoning_data.FileCampaignReasoningContextProvider` matches context
rows by target id, company, email, or vendor and feeds the normalized
`CampaignReasoningContextProvider` port documented in
`docs/reasoning_handoff_contract.md`.

For lightweight installs that do not already have reasoning JSON, use
`services.single_pass_reasoning_provider.SinglePassCampaignReasoningProvider`.
It calls the configured `LLMClient` once per opportunity with the packaged
`digest/b2b_campaign_reasoning_context` prompt and returns the same normalized
context shape. This is not a multi-hop graph reasoner; it is the small packaged
Tier 1 path for "source row in, reasoned draft out."

The example CLI can wire that provider directly when the product LLM adapter is
configured:

```bash
python scripts/run_extracted_campaign_generation_example.py \
  --llm pipeline \
  --single-pass-reasoning
```

Use host-provided prompt contracts by pointing at a markdown skill directory:

```bash
python scripts/run_extracted_campaign_generation_example.py \
  --skills-root customer_skills
```

Custom prompts with the same skill name override packaged prompts; missing
prompts fall back to the bundled `skills/digest/*.md` files.

The example uses in-memory product ports and an offline deterministic LLM stand
in, so it does not need Atlas, a database, or provider credentials. It proves
the customer-data path: JSON opportunities in, normalized campaign drafts out.

To run the same example through the product LLM adapter, configure the
`EXTRACTED_CAMPAIGN_LLM_*` environment variables and pass `--llm pipeline`:

```bash
python scripts/run_extracted_campaign_generation_example.py --llm pipeline
```

For database-backed runs, apply the product migrations, set
`EXTRACTED_DATABASE_URL`, and run the Postgres generation runner. It reads
`campaign_opportunities`, generates drafts, and persists them into
`b2b_campaigns`:

```bash
python scripts/run_extracted_content_pipeline_migrations.py --dry-run
python scripts/run_extracted_content_pipeline_migrations.py
```

The migration command reads `EXTRACTED_DATABASE_URL` first, then `DATABASE_URL`.
Pass `--database-url` explicitly when a host app keeps product data in a
separate database.

Load customer opportunities from JSON or CSV into the product table:

```bash
python scripts/load_extracted_campaign_opportunities.py customer_opportunities.csv --dry-run
python scripts/load_extracted_campaign_opportunities.py customer_opportunities.csv --account-id acct_123 --replace-existing
```

The loader uses the same normalization contract as the offline example. It is
append-only by default; `--replace-existing` deletes matching target ids for
the selected account and target mode before inserting the new rows.

For the full database-backed host install path, see
`docs/host_install_runbook.md`.

```bash
python scripts/run_extracted_campaign_generation_postgres.py --account-id acct_123 --limit 10
```

The Postgres runner accepts the same channel expansion:

```bash
python scripts/run_extracted_campaign_generation_postgres.py --account-id acct_123 --channels email_cold,email_followup
```

It also accepts the same host-provided reasoning JSON as the offline example:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --reasoning-context extracted_content_pipeline/examples/campaign_reasoning_context.json
```

Or generate lightweight reasoning context during the DB-backed run:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --single-pass-reasoning
```

Use `--skills-root customer_skills` on the Postgres runner for the same
host-prompt override behavior.

Export generated drafts for review without writing SQL:

```bash
python scripts/export_extracted_campaign_drafts.py --account-id acct_123 --limit 20
python scripts/export_extracted_campaign_drafts.py --account-id acct_123 --format csv --output campaign_drafts.csv
```

After review, update selected draft rows without writing SQL:

```bash
python scripts/review_extracted_campaign_drafts.py <campaign-id> --account-id acct_123 --status approved
python scripts/review_extracted_campaign_drafts.py <campaign-id> --account-id acct_123 --status queued --from-email audit@customer.com
python scripts/review_extracted_campaign_drafts.py <campaign-id> --account-id acct_123 --status cancelled --reason "customer rejected"
```

Hosts with FastAPI apps can mount the same draft review/export loop through a
router factory. The host injects its database pool, tenant scope, and auth
dependencies:

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

This adds JSON draft listing, CSV/JSON export, and approve/queue/cancel/expire
review routes without importing Atlas API globals.

Amazon seller installs can mount the seller-specific router. It adds seller
target CRUD, hosted category refresh and opportunity preparation triggers, plus
seller draft list/export/review routes locked to `target_mode="amazon_seller"`:

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

Hosts can call `POST /seller/intelligence/refresh`,
`POST /seller/opportunities/prepare`, or
`POST /seller/operations/refresh-and-prepare` from an admin UI or scheduler.
Draft generation still runs through the worker/CLI path so hosts can control
LLM provider policy and runtime separately.

Before generating seller drafts, prepare seller opportunities from active
seller targets and cached category intelligence:

```bash
python scripts/prepare_extracted_seller_campaign_opportunities.py \
  --account-id acct_123 \
  --category supplements \
  --replace-existing

python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --target-mode amazon_seller \
  --channels email_cold,email_followup
```

Send queued drafts through the configured provider:

```bash
export EXTRACTED_RESEND_API_KEY="re_..."
python scripts/send_extracted_campaigns.py \
  --provider resend \
  --default-from-email audit@customer.com \
  --limit 10
```

Ingest Resend webhook payloads into the campaign tables:

```bash
export EXTRACTED_RESEND_WEBHOOK_SECRET="whsec_..."
python scripts/ingest_extracted_campaign_webhook.py \
  --body-file resend-event.json \
  --headers-json resend-headers.json \
  --json
```

Hosts with FastAPI apps can mount the same policy through a router factory:

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

The unsubscribe route accepts both `GET` and RFC 8058 one-click `POST`
requests. By default, hosts must provide an unsubscribe-token verifier so a
public query string cannot suppress arbitrary recipient addresses. Use the same
secret for `CampaignSendConfig.unsubscribe_token_secret` or
`EXTRACTED_CAMPAIGN_UNSUBSCRIBE_TOKEN_SECRET` so generated unsubscribe links
carry tokens the router can verify.

Refresh campaign analytics after send or webhook updates:

```bash
python scripts/refresh_extracted_campaign_analytics.py --json
```

Progress due sequences and queue generated follow-up drafts:

```bash
python scripts/progress_extracted_campaign_sequences.py \
  --from-email audit@customer.com \
  --limit 10

python scripts/progress_extracted_campaign_sequences.py \
  --llm offline \
  --json
```

For non-FastAPI worker installs, the four operational CLIs can append the same
start/completed/failed telemetry to a JSONL audit trail:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --visibility-jsonl /var/log/content-ops/campaign-events.jsonl

python scripts/send_extracted_campaigns.py \
  --provider resend \
  --default-from-email audit@customer.com \
  --visibility-jsonl /var/log/content-ops/campaign-events.jsonl

python scripts/read_extracted_campaign_visibility.py \
  /var/log/content-ops/campaign-events.jsonl \
  --operation send_queued \
  --limit 10
```

Hosts with FastAPI apps can mount draft generation, send, sequence progression,
and analytics worker triggers through a hosted operations router. The host
injects its database pool, sender, optional LLM/skill/reasoning providers, and
auth dependencies; request payloads only control tenant scope, target/channel,
filters, and batch sizing.

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

If the host does not have a separate reasoning provider but does provide LLM and
skill providers, it can omit `reasoning_context_provider` and enable the
packaged single-pass provider:

```python
config=CampaignOperationsApiConfig(generation_single_pass_reasoning=True)
```

This adds `POST /campaigns/operations/drafts/generate`,
`POST /campaigns/operations/send/queued`,
`POST /campaigns/operations/sequences/progress`, and
`POST /campaigns/operations/analytics/refresh` without exposing provider
credentials, sender identity, unsubscribe policy, or LLM configuration through
HTTP payloads.

It also adds `GET /campaigns/operations/status` for admin dashboards. The
status route reports database availability, injected provider presence, feature
readiness, and configured limits without resolving sender/LLM/skill providers
or exposing secrets.

Hosts can inject a `visibility_provider` when mounting the router. The four
POST operation routes emit best-effort `campaign_operation_started`,
`campaign_operation_completed`, and `campaign_operation_failed` events through
the `VisibilitySink` port so dashboards can show worker activity without the
content product owning a dashboard store. Sink failures are logged and do not
change operation responses.

For local dashboards or host audit logs, wire a product-owned visibility sink:

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

Mount this router beside `create_b2b_campaign_router` to run the hosted B2B
flow without SQL in the admin UI:

1. `GET /campaigns/operations/status` lets the admin UI enable only ready
   operations.
2. `POST /campaigns/operations/drafts/generate` creates scoped draft rows from
   active `campaign_opportunities`.
3. `GET /b2b/campaigns/drafts` or `/drafts/export` lets operators inspect the
   generated drafts.
4. `POST /b2b/campaigns/drafts/review` moves selected drafts to `queued` after
   approval.
5. `POST /campaigns/operations/send/queued` sends approved queued drafts
   through the injected sender.
6. `POST /campaigns/operations/analytics/refresh` refreshes packaged funnel
   reporting after send/webhook activity.

## Import smoke test

```bash
python scripts/smoke_extracted_pipeline_imports.py
```

## Status tracker

Current extraction status is tracked in `extracted_content_pipeline/STATUS.md`.

## CI workflow

GitHub Actions workflow: `.github/workflows/extracted_pipeline_checks.yml` runs `bash scripts/run_extracted_pipeline_checks.sh` when extracted scaffold files change.

## File inventory

```bash
bash scripts/list_extracted_pipeline_files.sh
```

## LLM offline fallback

Set `EXTRACTED_PIPELINE_STANDALONE=1` to make the LLM bridge modules use their local no-op fallbacks instead of delegating to `extracted_llm_infrastructure`.

`campaign_llm_client.py` provides the product-owned `PipelineLLMClient`
adapter for campaign services. It satisfies the `campaign_ports.LLMClient`
port, resolves an LLM through the extracted LLM bridge when configured, and
normalizes `chat()` / `generate()` provider responses into `LLMResponse`.
`PipelineLLMClientConfig` and `create_pipeline_llm_client()` let a host wire
provider routing from explicit config, settings objects, or these environment
variables:

- `EXTRACTED_CAMPAIGN_LLM_WORKLOAD`
- `EXTRACTED_CAMPAIGN_LLM_PREFER_CLOUD`
- `EXTRACTED_CAMPAIGN_LLM_TRY_OPENROUTER`
- `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA`
- `EXTRACTED_CAMPAIGN_LLM_OPENROUTER_MODEL`

## Pipeline shims

`extracted_content_pipeline/pipelines/notify.py` provides a product-owned
notification dispatcher over the `VisibilitySink` port. It remains a safe no-op
when no sink is configured, and host apps can route emitted
`pipeline_notification` events to ntfy, Slack, dashboards, or audit streams.

Content-pipeline LLM bridge modules delegate to
`extracted_llm_infrastructure` instead of `atlas_brain`. That keeps the content
generation product boundary pointed at the extracted LLM/cost-optimization
product rather than at the monolith.

## Local utility shims

Several small utility shims provide product-owned local behavior by default so task imports do not require Atlas service modules:

- `config.py`: extracted settings from `settings.py`
- `pipelines/notify.py`: host-visible notification dispatcher backed by the
  `VisibilitySink` port
- `autonomous/tasks/_execution_progress.py`,
  `autonomous/tasks/_google_news.py`, `autonomous/tasks/_blog_ts.py`, and
  `autonomous/tasks/_blog_deploy.py`: product-owned utility helpers used by
  copied blog and campaign tasks
- `autonomous/tasks/_b2b_batch_utils.py`: product-owned Anthropic batch helper
  functions for metadata gates, request fingerprints, LLM slot resolution, and
  existing batch artifact reconciliation
- `autonomous/tasks/_blog_matching.py`: product-owned campaign-to-blog matcher
  with extracted base URL environment fallbacks
- `campaign_sequence_context.py` and
  `autonomous/tasks/_campaign_sequence_context.py`: product-owned sequence
  prompt/storage compaction helpers plus compatibility exports for copied tasks
- `autonomous/tasks/campaign_audit.py`: product-owned audit-log writer for
  campaign state changes
- `campaign_llm_client.py`: `PipelineLLMClient` adapter from the campaign
  `LLMClient` port to extracted LLM infrastructure services, with product-owned
  provider routing config
- `campaign_visibility.py`: reference in-memory and JSONL `VisibilitySink`
  adapters for hosted operations telemetry
- `storage/database.py` and `storage/models.py`: minimal `get_db_pool` and `ScheduledTask` fallbacks
- `campaign_postgres.py`: async Postgres adapters for intelligence,
  campaign, sequence, suppression, and audit ports, including the product-owned
  `campaign_opportunities` source table
- `campaign_postgres_generation.py`: product runner wiring
  `PostgresIntelligenceRepository`, `PostgresCampaignRepository`,
  `PipelineLLMClient`, and the local skill registry for DB-backed draft
  generation
- `campaign_postgres_export.py`: read-only draft export for host review flows
- `campaign_postgres_seller_targets.py`: seller target CRUD/list helpers for
  Amazon seller campaign installs
- `campaign_postgres_seller_opportunities.py`: prepares Amazon seller
  `campaign_opportunities` rows from seller targets and cached category
  intelligence snapshots
- `campaign_postgres_seller_category_intelligence.py`: refreshes broad Amazon
  seller category snapshots from review and product metadata tables
- `campaign_postgres_send.py`: DB-backed queued send runner that composes the
  campaign, suppression, audit, and sender ports for host worker CLIs
- `campaign_postgres_analytics.py`: DB-backed analytics refresh runner that
  composes campaign and audit ports for host worker CLIs
- `campaign_postgres_webhooks.py`: DB-backed webhook ingestion runner that
  composes campaign, suppression, audit, and Resend verification ports for
  host worker CLIs
- `api/campaign_webhooks.py`: optional FastAPI router factory for host-mounted
  campaign webhook and unsubscribe routes
- `api/campaign_operations.py`: optional FastAPI router factory for
  host-mounted draft generation, send, sequence progression, and analytics
  operation triggers with optional `VisibilitySink` telemetry
- `api/b2b_campaigns.py`: optional FastAPI router factory for host-mounted
  B2B draft list/export/review routes
- `api/seller_campaigns.py`: optional FastAPI router factory for host-mounted
  seller target management, category refresh, opportunity preparation, and
  seller draft review routes
- `campaign_postgres_sequence_progression.py`: DB-backed due-sequence worker
  that composes the sequence, audit, LLM, and skill ports for follow-up
  generation
- `campaign_postgres_import.py`: JSON/CSV customer opportunity import into the
  product `campaign_opportunities` table
- `storage/repositories/scheduled_task.py`: local execution metadata updater
- `skills/registry.py`: configurable markdown-backed skill registry
  implementing `.get()` and product `SkillStore.get_prompt()`, with optional
  host roots that override packaged prompt contracts
- `reasoning/archetypes.py`: product-owned deterministic churn-archetype scorer
  for extracted report builders
- `reasoning/temporal.py`: product-owned temporal analytics over vendor
  snapshot rows, including velocities, trends, category baselines, and
  anomaly serialization
- `reasoning/evidence_engine.py`: product-owned conclusion/suppression policy
  engine with built-in rules and optional host-provided evidence maps
- `services/__init__.py` and `services/protocols.py`: `llm_registry.get_active()` and `Message`
- `services/b2b/cache_runner.py`: local exact-cache request helpers and no-op lookup/store
- `services/b2b/enrichment_contract.py`: local enrichment contract fallbacks
- `services/scraping/sources.py`: `ReviewSource` enums and allowlist helpers
- `reasoning/wedge_registry.py`: `Wedge`, `get_wedge_meta`, and `validate_wedge`
- `services/blog_quality.py`: blog quality summary/revalidation helpers
- `services/company_normalization.py`: `normalize_company_name`
- `services/vendor_registry.py`: `resolve_vendor_name_cached`
- `services/apollo_company_overrides.py`, `services/b2b/corrections.py`, `services/tracing.py`, and `services/scraping/universal/html_cleaner.py`: local no-op or lightweight helpers
