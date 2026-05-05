# Extracted Content Pipeline Status

## Current state

- Scaffold root exists at `extracted_content_pipeline/`.
- Primary task modules are copied and import-smokeable:
  - `blog_post_generation.py`
  - `b2b_blog_post_generation.py`
  - `b2b_campaign_generation.py`
  - `b2b_vendor_briefing.py`
  - `complaint_content_generation.py`
  - `complaint_enrichment.py`
  - `article_enrichment.py`
- B2B helper siblings required by copied blog/campaign flows are present.
- Campaign and sequence prompt contracts are copied into `skills/digest/`, and
  the local skill registry implements `SkillStore.get_prompt()`.
- Core campaign schema migrations are copied into `storage/migrations/`
  (`b2b_campaigns`, `campaign_sequences`, suppressions, analytics, vendor and
  seller targets, outcomes, score components, and engagement timing).
- Compatibility bridge modules no longer map extracted package imports back to
  `atlas_brain` at runtime.
- LLM-facing content bridges now target `extracted_llm_infrastructure`
  (`pipelines.llm`, `services.b2b.anthropic_batch`, `services.llm.anthropic`)
  instead of pointing directly at `atlas_brain`.
- `campaign_llm_client.PipelineLLMClient` adapts extracted LLM infrastructure
  services to the standalone campaign `LLMClient` port. Its provider-routing
  config can be built from explicit mappings, settings objects, or
  `EXTRACTED_CAMPAIGN_LLM_*` environment variables.
- `campaign_postgres` provides async Postgres adapters for the intelligence,
  campaign, sequence, suppression, and audit ports against the copied campaign
  schema plus the product-owned `campaign_opportunities` source table.
- `campaign_postgres_import` loads normalized JSON/CSV opportunity rows into
  `campaign_opportunities`, with dry-run validation and optional
  replace-existing semantics for repeatable customer imports.
- `campaign_postgres_export` provides a read-only draft export path over
  generated `b2b_campaigns` rows so hosts can review JSON/CSV outputs without
  handwritten SQL.
- `campaign_postgres_review` provides a product-owned draft review/status
  update path so hosts can approve, queue, cancel, or expire generated
  `b2b_campaigns` rows after export without handwritten SQL.
- `api.b2b_campaigns` provides a FastAPI router factory around the draft
  list/export/review seams. Hosts inject pool providers, tenant scope, and any
  auth dependencies instead of importing Atlas API globals.
- `campaign_postgres_seller_targets` provides product-owned CRUD/list helpers
  for `seller_targets`, the Amazon seller outreach recipient table copied in
  the product migrations.
- `campaign_postgres_seller_opportunities` prepares Amazon seller
  `campaign_opportunities` rows from active `seller_targets` and cached
  `category_intelligence_snapshots`, giving hosts a standalone bridge from
  seller recipients plus category intelligence into the existing campaign
  generation runner.
- `campaign_postgres_seller_category_intelligence` refreshes broad Amazon
  seller category snapshots from host `product_reviews` and `product_metadata`
  tables, so the seller target -> opportunity -> draft loop can run without the
  Atlas seller scheduled task.
- `api.seller_campaigns` provides a FastAPI router factory around seller
  targets, hosted category refresh, opportunity preparation, and seller draft
  review/export routes. Seller draft review is guarded to
  `target_mode="amazon_seller"`.
- `campaign_postgres_send` provides a DB-backed queued send worker seam. Hosts
  inject a Resend or SES sender and reuse the product campaign, suppression,
  and audit ports to send rows already moved to `queued`.
- `campaign_postgres_analytics` provides a DB-backed analytics refresh worker
  seam. Hosts can refresh the packaged campaign funnel materialized view and
  audit the result without importing Atlas scheduled-task code.
- `campaign_postgres_webhooks` provides a DB-backed webhook ingestion seam.
  Hosts can verify Resend/Svix webhooks, record campaign engagement, apply
  suppression policy, and audit the result without importing Atlas API code.
- `api.campaign_webhooks` provides a FastAPI router factory around the
  DB-backed webhook seam. Hosts inject pool/signing-secret providers, an
  unsubscribe-token verifier, and any auth dependencies instead of importing
  Atlas API globals.
- `api.campaign_operations` provides a FastAPI router factory around hosted
  send, sequence progression, and analytics refresh triggers. Hosts inject the
  database, sender, optional LLM/skill providers, and auth dependencies; HTTP
  payloads only control batch sizing.
- `campaign_postgres_sequence_progression` provides a DB-backed follow-up
  generation worker seam. Hosts reuse due `campaign_sequences` rows, packaged
  or custom sequence prompts, and the product LLM port to queue the next
  follow-up draft without importing Atlas scheduled-task code.
- `storage.migration_runner` is product-owned and applies the packaged SQL
  migrations through a host-provided async pool or connection. The
  `scripts/run_extracted_content_pipeline_migrations.py` CLI wires it to
  `EXTRACTED_DATABASE_URL` / `DATABASE_URL` for standalone installs.
- `campaign_postgres_generation` wires the DB-backed generation path from
  `campaign_opportunities` to saved `b2b_campaigns` drafts through product
  ports.
- `CampaignGenerationService` supports multi-channel draft expansion through
  product config (`channels=("email_cold", "email_followup")`), including
  passing the generated cold-email context into follow-up prompts without
  importing the copied Atlas campaign task.
- Small utility shims now default to local extracted implementations:
  `config`, `pipelines.notify`, `reasoning.wedge_registry`,
  `reasoning.archetypes`, `reasoning.evidence_engine`, `reasoning.temporal`,
  `skills.registry`, `storage.database`, `storage.models`,
  `storage.repositories.scheduled_task`,
  `services.__init__`, `services.protocols`, `services.blog_quality`,
  `services.b2b.cache_runner`, `services.b2b.enrichment_contract`,
  `services.company_normalization`, `services.vendor_registry`,
  `services.apollo_company_overrides`, `services.b2b.corrections`,
  `services.tracing`, `services.scraping.sources`, and
  `services.scraping.universal.html_cleaner`.
- Standalone readiness audit reports 0 Atlas runtime import findings.
- `pipelines.notify` is product-owned and dispatches through the
  `VisibilitySink` port when configured, while staying a no-op when no host
  visibility adapter is installed.
- Small task utility helpers are product-owned rather than Atlas-synced:
  `_execution_progress`, `_google_news`, `_blog_ts`, `_blog_deploy`, and
  `_b2b_batch_utils`, and `_blog_matching`.
- `_b2b_batch_utils` is product-owned and keeps Anthropic batch metadata gates,
  request fingerprints, LLM slot resolution, and existing artifact
  reconciliation inside the extracted boundary.
- `_blog_matching` is product-owned and matches campaign targets to generated
  blog posts with extracted base URL configuration.
- `campaign_sequence_context` is product-owned, and the copied task-local
  `_campaign_sequence_context` module now exports that standalone
  implementation for Atlas-compatible task imports.
- `campaign_audit` is product-owned and writes campaign state-change audit
  rows without importing Atlas logging or task helpers.
- Vendor briefing and campaign generation import seams are product-owned:
  `services.campaign_sender`, `services.vendor_target_selection`,
  `autonomous.tasks.campaign_suppression`, `templates.email.vendor_briefing`,
  `services.b2b.account_opportunity_claims`, `services.campaign_quality`,
  `services.campaign_reasoning_context`, and `autonomous.visibility`.
- `skills.registry` is product-owned and markdown-backed. Hosts can pass a
  custom skill root to override packaged prompt contracts while retaining
  bundled fallback prompts.
- `CampaignReasoningContextProvider` is the campaign-core boundary for
  upstream reasoning. Hosts pass already-compressed witness/anchor/account
  context into the generator; `_b2b_pool_compression.py` stays outside the
  standalone campaign product.
- `campaign_reasoning_data.FileCampaignReasoningContextProvider` is the
  reference file-backed adapter for that boundary. It lets examples and hosts
  provide precomputed reasoning JSON keyed by target id, company, email, or
  vendor without importing a reasoning producer.
- Both the offline and Postgres campaign generation runners can consume that
  JSON through `--reasoning-context`, so file-backed host reasoning is available
  on demo and DB-backed generation paths.
- `reasoning.archetypes` is product-owned and provides deterministic
  churn-archetype scoring, best-match selection, top-match filtering, and
  falsification-condition lookup without Atlas dependencies.
- `reasoning.temporal` is product-owned and computes snapshot velocity,
  acceleration, trend, category-baseline, anomaly, and recency-weight outputs
  through a host-provided async `fetch` interface.
- `reasoning.evidence_engine` is product-owned and evaluates deterministic
  conclusion gates, section suppression gates, and confidence labels from
  built-in rules or an optional host-provided evidence map.
- Reasoning generation is explicitly host-owned. AI Content Ops consumes
  compressed reasoning through `CampaignReasoningContextProvider` and the
  contract documented in `docs/reasoning_handoff_contract.md`; it does not
  import Atlas synthesis, pool compression, or extracted reasoning-core
  internals.
- `docs/host_install_runbook.md` documents the end-to-end host path for
  database-backed installs: migrations, opportunity import, optional reasoning
  JSON, optional skill roots, generation, and output verification.

## Validation gates in repo

- `scripts/sync_extracted_content_pipeline.sh`
- `scripts/validate_extracted_content_pipeline.sh`
- `scripts/check_ascii_python.sh`
- `scripts/check_extracted_imports.py`
- `scripts/smoke_extracted_pipeline_imports.py`
- `scripts/run_extracted_pipeline_checks.sh` (includes
  `audit_extracted_standalone.py --fail-on-debt`)

The sync, validation, ASCII, and relative-import entry points are stable
product wrappers over `extracted/_shared/scripts/`. The standalone audit remains
content-specific because it checks runtime `atlas_brain` coupling across the
extracted package, not just manifest-relative import resolution.

## Remaining extraction work

1. Continue trimming copied helper surface to only the modules required by
   target sellable workflows. The first utility group is no longer
   manifest-synced from Atlas.
2. Move copied task imports and package layout toward native extracted modules instead of manifest-synced mirrors.
3. Add focused unit tests around extraction-specific contracts (manifest sync, importability, runner smoke).
4. For each new content type, state whether reasoning is required, optional, or
   absent; if required, consume it through a host/provider port instead of
   copying reasoning producer internals.

See `docs/remaining_productization_audit.md` for the current campaign-core
import blockers and the recommended next PR sequence.

## Operational note

The runtime import gate is clean, but this scaffold remains an in-repo extraction staging area until adapters are productionized and copied helper scope is narrowed.
