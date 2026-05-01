# Extracted Content Pipeline (Staging Copy)

This directory is an additive extraction scaffold copied from `atlas_brain`.
It is intentionally kept side-by-side with Atlas so pipeline logic can be
carved out safely without removing or changing production code.

## Current contents

- `autonomous/tasks/`: copied task implementations
- `services/`: copied support shims and staged service dependencies
- `skills/digest/`: copied prompt skill contracts
- `storage/migrations/`: copied persistence migrations
- `docs/`: extraction maps for productized pipeline slices

## Sync command

To refresh this scaffold from Atlas source of truth:

```bash
bash scripts/sync_extracted_content_pipeline.sh
```

## Manifest

Mirror mappings are declared in `extracted_content_pipeline/manifest.json` so sync and validation use one source of truth.

## Scope

This scaffold preserves code exactly as copied so behavior and signatures remain
unchanged while extraction work continues.

This is not yet the sellable product boundary. A customer-usable module must be
able to install and run without the Atlas monolith on `PYTHONPATH`. Until the
standalone audit reaches zero runtime `atlas_brain` imports, this package is a
staging copy, not a deployable product.


## Validation command

```bash
bash scripts/validate_extracted_content_pipeline.sh
```

## ASCII compliance check

```bash
bash scripts/check_ascii_python.sh
```

## Import debt check

```bash
python scripts/check_extracted_imports.py
```

Known unresolved relative imports are tracked in `extracted_content_pipeline/import_debt_allowlist.txt`.

## Standalone readiness audit

```bash
python scripts/audit_extracted_standalone.py
python scripts/audit_extracted_standalone.py --fail-on-debt
```

The first command reports Atlas runtime coupling. The second is the product gate
we should enable once staged shims have been replaced with product-owned ports
and adapters.

## One-shot checks

```bash
bash scripts/run_extracted_pipeline_checks.sh
```

## Compatibility shims

To keep copied task modules importable inside this repo, package-level bridge modules are provided under `extracted_content_pipeline/` (for example `config.py`, `storage/database.py`, `pipelines/llm.py`, and `services/*`). Some remain compatibility delegates to `atlas_brain`; small utility shims now use product-owned local implementations by default.

B2B helper siblings required by `b2b_blog_post_generation.py` are also copied into `extracted_content_pipeline/autonomous/tasks/`.

These shims are temporary extraction scaffolding. They should not ship in the
customer product.

The email/campaign generation slice is mapped in `docs/email_campaign_generation_pipeline.md`, with standalone productization requirements in `docs/standalone_productization.md`.

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

## Standalone mode toggle

Set `EXTRACTED_PIPELINE_STANDALONE=1` to use `extracted_content_pipeline/settings.py` instead of delegating config to `atlas_brain.config`.

## Pipeline shims

`extracted_content_pipeline/pipelines/notify.py` provides a local no-op notifier so task modules can execute without Atlas pipeline services.

Content-pipeline LLM bridge modules delegate to
`extracted_llm_infrastructure` instead of `atlas_brain`. That keeps the content
generation product boundary pointed at the extracted LLM/cost-optimization
product rather than at the monolith.

## Standalone storage shims

In standalone mode, `extracted_content_pipeline/storage/database.py` and `extracted_content_pipeline/storage/models.py` provide minimal local fallbacks (`get_db_pool`, `ScheduledTask`) so task entrypoints can execute without Atlas storage imports.

## Standalone skill registry

In standalone mode, `extracted_content_pipeline/skills/registry.py` uses local markdown files under `extracted_content_pipeline/skills/` for `get_skill_registry()` lookups.

## Local utility shims

Several small utility shims provide product-owned local behavior by default so task imports do not require Atlas service modules:

- `services/__init__.py` and `services/protocols.py`: `llm_registry.get_active()` and `Message`
- `services/scraping/sources.py`: `ReviewSource` enums and allowlist helpers
- `reasoning/wedge_registry.py`: `Wedge`, `get_wedge_meta`, and `validate_wedge`
- `services/blog_quality.py`: blog quality summary/revalidation helpers
- `services/company_normalization.py`: `normalize_company_name`
- `services/vendor_registry.py`: `resolve_vendor_name_cached`
- `services/apollo_company_overrides.py`, `services/b2b/corrections.py`, `services/tracing.py`, and `services/scraping/universal/html_cleaner.py`: local no-op or lightweight helpers

## Standalone B2B contract shims

In standalone mode, `extracted_content_pipeline/services/b2b/enrichment_contract.py` provides local fallbacks (`pain_category_for_bucket`, `quote_grade_phrases`, `resolve_pain_confidence`) used by extracted B2B helpers.
