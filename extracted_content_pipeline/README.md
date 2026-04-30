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
