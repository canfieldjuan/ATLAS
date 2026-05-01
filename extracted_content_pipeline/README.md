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
