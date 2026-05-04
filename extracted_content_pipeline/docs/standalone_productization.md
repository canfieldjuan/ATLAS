# Standalone Productization Contract

The extracted package must be installable and runnable without the Atlas
monolith. The scaffold began as a staging mirror for extraction traceability;
the runtime import gate is now clean, but the package is not the customer
product until the minimal adapters are hardened and copied helper scope is
trimmed.

## Non-Negotiable Product Rule

Runtime product code must not import `atlas_brain`.

Allowed during extraction:

- `manifest.json` source paths that point at Atlas source-of-truth files.
- Documentation that references Atlas as the origin.
- Temporary compatibility shims in the staging scaffold.

Not allowed in the sellable module:

- `from atlas_brain... import ...`
- `import atlas_brain...`
- A package install that requires the Atlas repo on `PYTHONPATH`.
- API/task code that reaches directly into Atlas settings, DB pools, auth,
  visibility events, scheduled-task models, LLM registry, or B2B helper modules.

## Required Product Ports

Before the campaign system can be sold as its own module, Atlas infrastructure
must be replaced with product-owned interfaces:

- `CampaignRepository`: campaign, sequence, suppression, audit, analytics, and
  target persistence.
- `IntelligenceRepository`: account/vendor/category inputs used to build
  campaign blueprints.
- `LLMClient`: message completion, JSON cleanup, tracing metadata, batch replay.
- `SkillStore`: prompt contract lookup by name/version.
- `CampaignSender`: Resend/SES send abstraction plus provider message ids.
- `WebhookVerifier`: provider signature verification and event normalization.
- `AuditSink`: immutable campaign lifecycle events.
- `VisibilitySink`: progress/events for hosted dashboards.
- `CampaignReasoningContextProvider`: host-owned reasoning handoff. The
  content product consumes already-compressed context through this port; it
  does not import Atlas synthesis, pool compression, graph state, or
  `extracted_reasoning_core` internals directly.
- `TenantScope`: account ownership and permission filtering.
- `Clock`: deterministic send-window and delay calculations.

The first pass of these boundaries lives in
`extracted_content_pipeline/campaign_ports.py`. Refactors should move copied
campaign code toward those interfaces before adding more raw Atlas modules to
the manifest.

## Decoupling Order

1. Define product-local models, settings, and ports.
2. Move pure helpers first: sequence context, suppression decisions, quality
   validation, send-window logic, webhook event normalization, and analytics
   summaries.
3. Convert services to accept injected ports instead of importing Atlas
   settings, DB pools, LLM registry, or visibility helpers.
4. Convert tasks into orchestrators that receive a dependency bundle.
5. Convert APIs into an app/router factory that receives auth and repository
   adapters from the host app.
6. Keep Atlas adapters in a separate optional package or directory so Atlas can
   still run the product internally without contaminating the customer module.

## First Standalone Slice

`extracted_content_pipeline/campaign_suppression.py` is the first product-owned
campaign module in this scaffold. It implements normalized email/domain
suppression behavior against `SuppressionRepository` and does not import Atlas.
Use it as the pattern for the next slices: keep policy and orchestration in the
product package, and push persistence/provider concerns behind ports.

`extracted_content_pipeline/campaign_sequence_context.py` is the second
standalone slice. It keeps the sequence prompt/storage compaction behavior but
replaces Atlas settings reads with explicit `SequenceContextLimits`.
The copied task-local `_campaign_sequence_context.py` now exports this
product-owned module so Atlas-compatible campaign task imports use the same
standalone limits path.

`extracted_content_pipeline/autonomous/tasks/campaign_audit.py` is product-owned
as the copied task-facing audit writer. It keeps the small never-raise
`log_campaign_event(...)` contract while using extracted logger naming and local
coverage for UUID coercion, metadata serialization, and failure handling.

`extracted_content_pipeline/campaign_sender.py` is the third standalone slice.
It keeps Resend and SES provider behavior but uses explicit provider config and
the product `SendRequest`/`SendResult` dataclasses.

`extracted_content_pipeline/campaign_send.py` is the fourth standalone slice.
It orchestrates due sends through `CampaignRepository`,
`CampaignSuppressionService`, `CampaignSender`, `AuditSink`, and `Clock`.
Atlas-specific quality revalidation, fatigue SQL, sequence scheduling, and
visibility events remain later adapter/service work.

`extracted_content_pipeline/campaign_webhooks.py` is the fifth standalone
slice. It verifies Resend/Svix signatures, normalizes ESP webhook payloads into
the product `WebhookEvent` dataclass, records events through
`CampaignRepository`, and applies bounce/complaint/unsubscribe suppressions
through `CampaignSuppressionService` without importing Atlas API code.

`extracted_content_pipeline/campaign_analytics.py` is the sixth standalone
slice. It wraps analytics refresh through `CampaignRepository.refresh_analytics`
and reports success/failure through optional audit and visibility ports.

`extracted_content_pipeline/campaign_sequence_progression.py` is the seventh
standalone slice. It builds follow-up prompt context, selects the sequence skill,
parses generated JSON, and queues due follow-up steps through
`CampaignSequenceRepository`, `LLMClient`, `SkillStore`, and optional audit
ports.

`extracted_content_pipeline/campaign_generation.py` is the eighth standalone
slice. It reads campaign opportunities through `IntelligenceRepository`, prompts
through `SkillStore` and `LLMClient`, parses generated draft JSON, and persists
`CampaignDraft` rows through `CampaignRepository`.

The generator now owns the first concrete copied-task producer behavior:
multi-channel expansion for one normalized opportunity. Hosts can request
`email_cold` and `email_followup` drafts in one run, and the follow-up prompt
receives the generated cold-email context through the product opportunity
payload. This keeps the cold/follow-up flow inside the product-owned ports
instead of calling the copied `b2b_campaign_generation.py` task.

Reasoning remains a host/product boundary for this slice. The generator accepts
pre-compressed reasoning through `CampaignReasoningContextProvider` and
normalizes it with `services.campaign_reasoning_context`; it must not reach into
Atlas reasoning producers or the extracted reasoning-core internals. The
contract is documented in `docs/reasoning_handoff_contract.md`. The
file-backed reference adapter in `campaign_reasoning_data.py` lets standalone
examples consume host-generated reasoning JSON through that same port without
adding a reasoning runtime dependency.

`extracted_content_pipeline/campaign_postgres_generation.py` wires the
database-backed product path. Hosts pass an async Postgres pool and the runner
combines `PostgresIntelligenceRepository`, `PostgresCampaignRepository`,
`PipelineLLMClient`, and the local skill registry to read opportunities,
generate drafts, and save them back to Postgres.

`extracted_content_pipeline/campaign_example.py` is the runnable product example
for campaign generation. It wires in-memory ports, a static prompt store, and an
offline deterministic LLM so customer opportunity JSON can be converted into
drafts without Atlas, a database, or provider credentials.

`extracted_content_pipeline/skills/registry.py` is the product-owned prompt
registry. It reads packaged markdown skills by default and accepts host-provided
skill roots that override bundled prompt contracts without importing Atlas.

`extracted_content_pipeline/campaign_customer_data.py` is the customer-export
adapter slice. It loads JSON or CSV opportunity rows, normalizes them through
the product opportunity contract, returns non-fatal data-quality warnings, and
exposes `FileIntelligenceRepository` for running generation from customer files.

`extracted_content_pipeline/campaign_postgres.py` owns the Postgres adapter path
for customer opportunity reads through `PostgresIntelligenceRepository`. It
reads the product-owned `campaign_opportunities` table and normalizes rows
through the same contract as the JSON/CSV adapters before generation.

`extracted_content_pipeline/campaign_postgres_import.py` owns the customer-data
ingest path for database-backed installs. It loads normalized opportunity rows
into `campaign_opportunities`, supports dry-run validation, and keeps imports
append-only unless a host explicitly requests replace-existing behavior for the
selected account, target mode, and target ids.

`extracted_content_pipeline/campaign_postgres_export.py` owns the read-only
review path for generated drafts. It filters saved `b2b_campaigns` rows by
account, status, target mode, channel, vendor, or company and emits JSON or CSV
for host review workflows.

`extracted_content_pipeline/campaign_postgres_review.py` owns the write side of
that host review loop. It updates selected `b2b_campaigns` rows by explicit
campaign id, optional account scope, and source-status guard so hosts can move
reviewed drafts to `approved`, `queued`, `cancelled`, or `expired` without
writing ad hoc SQL.

`extracted_content_pipeline/campaign_postgres_send.py` owns the DB-backed send
worker seam. It composes `PostgresCampaignRepository`,
`PostgresSuppressionRepository`, `PostgresCampaignAuditSink`, and
`CampaignSendService` so hosts can send queued drafts through an injected
`CampaignSender` without importing Atlas task code.

`extracted_content_pipeline/campaign_postgres_analytics.py` owns the DB-backed
analytics refresh worker seam. It composes `PostgresCampaignRepository`,
`PostgresCampaignAuditSink`, and `CampaignAnalyticsRefreshService` so hosts can
refresh the packaged campaign funnel materialized view without importing Atlas
scheduled-task code.

`extracted_content_pipeline/campaign_postgres_sequence_progression.py` owns the
DB-backed sequence progression worker seam. It composes
`PostgresCampaignSequenceRepository`, `PostgresCampaignAuditSink`, the product
LLM port, and the local skill registry so hosts can queue due follow-up drafts
without importing Atlas scheduled-task code.

`extracted_content_pipeline/storage/migration_runner.py` owns the standalone
schema installation path. It lists packaged SQL migrations, tracks applied
versions in a product metadata table, supports dry runs, and accepts either a
host async pool or direct connection. The
`scripts/run_extracted_content_pipeline_migrations.py` CLI exposes that runner
through `EXTRACTED_DATABASE_URL`, `DATABASE_URL`, or an explicit
`--database-url` argument.

`extracted_content_pipeline/docs/host_install_runbook.md` ties the standalone
DB-backed path together for hosts: configure a DSN, apply migrations, validate
and import customer opportunity data, optionally attach reasoning JSON and
custom skill roots, run Postgres-backed generation, and verify saved drafts.

`extracted_content_pipeline/campaign_opportunities.py` owns the host/customer
opportunity input contract. It accepts loose customer rows, preserves custom
fields, and adds stable prompt/storage keys (`target_id`, `company_name`,
`vendor_name`, contact fields, scores, pain points, competitors, and evidence)
before `CampaignGenerationService` calls reasoning providers or the LLM.

`extracted_content_pipeline/campaign_llm_client.py` is the provider-routing
slice for LLM access. It adapts extracted LLM infrastructure services to the
product `LLMClient` port and exposes `PipelineLLMClientConfig` so hosts can
wire workload, OpenRouter model, and fallback behavior without importing Atlas
settings.

`extracted_content_pipeline/pipelines/notify.py` is the first product-owned
visibility slice. It preserves the copied task-facing
`send_pipeline_notification(...)` API but emits through the `VisibilitySink`
port when a host configures one. With no sink configured it remains a safe
no-op, so standalone task imports do not require Atlas notification services.

The first helper-surface trim moved `_execution_progress`, `_google_news`,
`_blog_ts`, and `_blog_deploy` out of manifest sync and into product ownership.
These are still used by copied task modules, but future changes now happen in
the extracted product boundary instead of being pulled from Atlas.

`extracted_content_pipeline/autonomous/tasks/_b2b_batch_utils.py` is now
product-owned as the Anthropic batch utility boundary. It preserves the copied
task-facing helpers for metadata flags, request fingerprints, LLM resolution,
and existing artifact reconciliation, but resolves product environment keys and
fails safe when a standalone host has not installed an activatable LLM registry.

`extracted_content_pipeline/autonomous/tasks/_blog_matching.py` is now
product-owned for campaign-to-blog matching. It preserves the copied relevance
scoring rules while resolving blog URLs from extracted settings or
`EXTRACTED_*_BLOG_BASE_URL` environment variables instead of Atlas settings.

`extracted_content_pipeline/reasoning/archetypes.py` is the first product-owned
reasoning policy slice. It scores vendor evidence against deterministic churn
archetypes, returns thresholded matches, and exposes falsification conditions
without a database, LLM, or Atlas import.

`extracted_content_pipeline/reasoning/temporal.py` is the second product-owned
reasoning policy slice. It computes velocity, acceleration, long-term trends,
category percentiles, anomalies, recency weights, and evidence serialization
from a host-provided async `fetch` interface instead of Atlas helper imports.

`extracted_content_pipeline/reasoning/evidence_engine.py` is the third
product-owned reasoning policy slice. It evaluates conclusion gates, section
suppression gates, and confidence labels from built-in product defaults or an
optional host-provided JSON/YAML evidence map.

## Readiness Gate

Run:

```bash
python scripts/audit_extracted_standalone.py --fail-on-debt
```

The command must pass before this package is considered customer-usable.

## Current Campaign-Specific Blockers

- `b2b_campaign_generation.py` imports Atlas config, DB, visibility, skills,
  LLM routing, B2B batch helpers, vendor target selection, product matching,
  and B2B intelligence readers.
- `campaign_send.py` imports Atlas config, DB, visibility, campaign quality,
  sender, and suppression helpers.
- The copied Atlas `autonomous/tasks/campaign_sequence_progression.py` imports
  Atlas config, DB, scheduled-task model, skills, LLM routing, tracing, and
  protocol classes. The product-owned Postgres worker above is the standalone
  path.
- `api/b2b_campaigns.py`, `api/seller_campaigns.py`, and
  `api/campaign_webhooks.py` need an app-factory boundary and host-provided
  auth/tenant dependencies.
- SQL migrations and customer opportunity imports now have product-owned
  runners and a host install runbook, but customer installation still needs
  final base-schema hardening.
