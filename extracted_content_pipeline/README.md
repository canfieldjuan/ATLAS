# Extracted Content Pipeline (Staging Copy)

This directory is an additive extraction scaffold copied from `atlas_brain`.
It is intentionally kept side-by-side with Atlas so pipeline logic can be
carved out safely without removing or changing production code.

## Current contents

- `autonomous/tasks/`: copied task implementations
- `skills/digest/`: copied prompt skill contracts
- `storage/migrations/`: copied blog persistence migrations

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

## One-shot checks

```bash
bash scripts/run_extracted_pipeline_checks.sh
```

## Compatibility shims

To keep copied task modules importable inside this repo, package-level bridge modules are provided under `extracted_content_pipeline/` (for example `config.py`, `storage/database.py`, `pipelines/llm.py`, and `services/*`) that delegate to `atlas_brain` implementations.

B2B helper siblings required by `b2b_blog_post_generation.py` are also copied into `extracted_content_pipeline/autonomous/tasks/`.

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

## Standalone pipeline shims

In standalone mode (`EXTRACTED_PIPELINE_STANDALONE=1`), `extracted_content_pipeline/pipelines/llm.py` and `extracted_content_pipeline/pipelines/notify.py` provide local fallback behavior (no-op notifier and safe JSON/cleaning helpers) so task modules can execute without Atlas pipeline services.

## Standalone storage shims

In standalone mode, `extracted_content_pipeline/storage/database.py` and `extracted_content_pipeline/storage/models.py` provide minimal local fallbacks (`get_db_pool`, `ScheduledTask`) so task entrypoints can execute without Atlas storage imports.

## Standalone skill registry

In standalone mode, `extracted_content_pipeline/skills/registry.py` uses local markdown files under `extracted_content_pipeline/skills/` for `get_skill_registry()` lookups.

## Standalone service shims

In standalone mode, `extracted_content_pipeline/services/__init__.py` and `extracted_content_pipeline/services/protocols.py` provide minimal local fallbacks (`llm_registry.get_active()`, `Message`) so task imports do not require Atlas service modules.

## Standalone source shims

In standalone mode, `extracted_content_pipeline/services/scraping/sources.py` provides local `ReviewSource` enums and allowlist helpers used by B2B tasks without requiring Atlas source modules.

## Standalone reasoning shims

In standalone mode, `extracted_content_pipeline/reasoning/wedge_registry.py` provides local `Wedge`, `get_wedge_meta`, and `validate_wedge` fallbacks used by B2B extracted modules.

## Standalone quality shims

In standalone mode, `extracted_content_pipeline/services/blog_quality.py` provides local fallback quality helpers (`blog_quality_summary`, `blog_quality_revalidation`, and `merge_blog_first_pass_quality_data_context`) for B2B blog pipeline paths.

## Standalone normalization shims

In standalone mode, `extracted_content_pipeline/services/company_normalization.py` provides a local `normalize_company_name` fallback used by extracted B2B helpers.

## Standalone vendor registry shims

In standalone mode, `extracted_content_pipeline/services/vendor_registry.py` provides a local `resolve_vendor_name_cached` fallback used by extracted B2B helpers.
