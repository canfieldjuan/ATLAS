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
- Compatibility bridge modules no longer map extracted package imports back to
  `atlas_brain` at runtime.
- LLM-facing content bridges now target `extracted_llm_infrastructure`
  (`pipelines.llm`, `services.b2b.anthropic_batch`, `services.llm.anthropic`)
  instead of pointing directly at `atlas_brain`.
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

## Validation gates in repo

- `scripts/sync_extracted_content_pipeline.sh`
- `scripts/validate_extracted_content_pipeline.sh`
- `scripts/check_ascii_python.sh`
- `scripts/check_extracted_imports.py`
- `scripts/smoke_extracted_pipeline_imports.py`
- `scripts/run_extracted_pipeline_checks.sh`

## Remaining extraction work

1. Harden minimal local adapters into customer-grade ports for DB/LLM/skills/notify/reasoning.
2. Trim copied helper surface to only the modules required by target sellable workflows.
3. Move copied task imports and package layout toward native extracted modules instead of manifest-synced mirrors.
4. Add focused unit tests around extraction-specific contracts (manifest sync, importability, runner smoke).

## Operational note

The runtime import gate is clean, but this scaffold remains an in-repo extraction staging area until adapters are productionized and copied helper scope is narrowed.
