# Extracted Content Pipeline Status

## Current state

- Scaffold root exists at `extracted_content_pipeline/`.
- Primary task modules are copied and import-smokeable:
  - `blog_post_generation.py`
  - `b2b_blog_post_generation.py`
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
- `campaign_postgres` provides async Postgres adapters for the campaign,
  sequence, suppression, and audit ports against the copied campaign schema.
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
  `_execution_progress`, `_google_news`, `_blog_ts`, and `_blog_deploy`.
- `reasoning.archetypes` is product-owned and provides deterministic
  churn-archetype scoring, best-match selection, top-match filtering, and
  falsification-condition lookup without Atlas dependencies.
- `reasoning.temporal` is product-owned and computes snapshot velocity,
  acceleration, trend, category-baseline, anomaly, and recency-weight outputs
  through a host-provided async `fetch` interface.
- `reasoning.evidence_engine` is product-owned and evaluates deterministic
  conclusion gates, section suppression gates, and confidence labels from
  built-in rules or an optional host-provided evidence map.

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

## Operational note

The runtime import gate is clean, but this scaffold remains an in-repo extraction staging area until adapters are productionized and copied helper scope is narrowed.
