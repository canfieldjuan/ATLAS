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
- Compatibility bridge modules still map some extracted package imports back to `atlas_brain` for side-by-side operation.
- LLM-facing content bridges now target `extracted_llm_infrastructure`
  (`pipelines.llm`, `services.b2b.anthropic_batch`, `services.llm.anthropic`)
  instead of pointing directly at `atlas_brain`.
- Small utility shims now default to local extracted implementations:
  `pipelines.notify`, `reasoning.wedge_registry`, `services.__init__`,
  `services.protocols`, `services.blog_quality`,
  `services.company_normalization`, `services.vendor_registry`,
  `services.apollo_company_overrides`, `services.b2b.corrections`,
  `services.tracing`, `services.scraping.sources`, and
  `services.scraping.universal.html_cleaner`.
- Standalone readiness audit is down to 19 Atlas runtime import findings
  (15 hard imports, 4 bridge shims).

## Validation gates in repo

- `scripts/sync_extracted_content_pipeline.sh`
- `scripts/validate_extracted_content_pipeline.sh`
- `scripts/check_ascii_python.sh`
- `scripts/check_extracted_imports.py`
- `scripts/smoke_extracted_pipeline_imports.py`
- `scripts/run_extracted_pipeline_checks.sh`

## Remaining extraction work

1. Replace remaining bridge-module delegation (`from atlas_brain...`) with native extracted implementations package by package.
2. Trim copied helper surface to only the modules required by target sellable workflows.
3. Introduce standalone config and runtime wiring for DB/LLM/skills/notify.
4. Add focused unit tests around extraction-specific contracts (manifest sync, importability, runner smoke).

## Operational note

Until bridge-module delegation is removed, this scaffold remains an in-repo extraction staging area (not yet a fully detached standalone runtime).
