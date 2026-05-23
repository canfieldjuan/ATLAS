# PR: Content Ops Blog Blueprint JSON Smoke

## Why this slice exists

PR #836 proved the hosted Content Ops `blog_post` path end to end with a
default seeded blueprint. That proves the pipe works, but operators still cannot
point the live smoke at a specific blog blueprint file before testing a real
topic, audience, or content angle.

This slice adds the thinnest operator-controlled proof: pass one custom blog
blueprint JSON file to the same live smoke, seed it through the existing
blueprint ingestion contract, execute the matching row, and verify the saved
draft path still works.

## Scope (this PR)

Ownership lane: content-ops/blog-blueprint-json-smoke

1. Add `--blog-blueprint-json` to
   `scripts/smoke_content_ops_live_generation.py` for `--output blog_post`.
2. Reuse `load_blog_blueprints_from_file()` so the smoke shares the existing
   host blueprint schema instead of hand-parsing a second format.
3. Require the custom file to normalize to exactly one blueprint for a
   one-draft smoke run.
4. Execute with the seeded blueprint's actual `topic_type` and `slug` filters so
   a custom row cannot be skipped by the smoke-only default filter or displaced
   by another row in the same topic type.
5. Preserve the current default seeded blueprint when no custom file is passed.
6. Add focused tests for the custom JSON path, invalid/multi-row input, and
   default behavior.
7. Document the operator command and minimal JSON shape in the README and host
   runbook.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Blog-Blueprint-Json-Smoke.md` | Plan doc for this slice. |
| `scripts/smoke_content_ops_live_generation.py` | Add custom blog blueprint JSON loading, validation, and seeded-filter alignment. |
| `tests/test_smoke_content_ops_live_generation.py` | Cover custom JSON seeding and failure contracts. |
| `extracted_content_pipeline/blog_blueprint_postgres.py` | Let blueprint reads filter by slug as well as topic type. |
| `tests/test_extracted_blog_blueprint_postgres.py` | Lock the exact topic type plus slug read predicate. |
| `extracted_content_pipeline/README.md` | Document the custom blog blueprint smoke option. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Add host-facing runbook guidance for the same option. |

## Mechanism

The existing default blog smoke still works:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_123 \
  --env-file /path/to/Atlas/.env \
  --json
```

Operators can now pass one custom blueprint file:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_123 \
  --blog-blueprint-json ./blog-blueprint.json \
  --env-file /path/to/Atlas/.env \
  --json
```

The smoke loads the file with `load_blog_blueprints_from_file()`, applying the
CLI `target_mode` and the smoke topic type as defaults. It rejects files that
produce zero or multiple blueprints because this smoke executes `limit=1` and
should be deterministic.

After saving the blueprint, the smoke updates `payload["inputs"]["filters"]` to
the saved row's actual `topic_type` and `slug`. The Postgres repository now
honors both filters, so the live smoke selects the exact row it just seeded
instead of relying on recency within a topic type.

## Intentional

- This is a file-input smoke, not a new blueprint schema. The existing
  `blog_blueprint_ingest` adapter remains the source of truth for accepted JSON
  shapes.
- The smoke still seeds only one blueprint. Bulk import is already covered by
  `scripts/load_extracted_blog_blueprints.py`; live generation smoke should stay
  deterministic.
- The generated topic can still be overridden with `--input topic=...`. A
  custom blueprint controls the source row; the per-run topic remains the
  existing Content Ops input.

## Deferred

- HTTP-level `/content-ops/execute` authenticated smoke remains deferred because
  it needs a real B2B session and route auth setup.
- Bulk custom blueprint live smoke is deferred. Use the dedicated import CLI for
  bulk loading, then run generation separately.
- Parked hardening: existing `ATLAS-HARDENING.md` items are for the older Atlas
  blog/deep-dive content pipeline, not this Content Ops `blog_blueprints` smoke
  path. They were scanned and remain parked.

## Verification

- `pytest tests/test_smoke_content_ops_live_generation.py -q` -> 8 passed.
- `python -m py_compile scripts/smoke_content_ops_live_generation.py tests/test_smoke_content_ops_live_generation.py` -> passed.
- `git diff --check` -> passed.
- `bash scripts/validate_extracted_content_pipeline.sh` -> passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -> passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -> passed.
- `bash scripts/check_ascii_python.sh` -> passed.
- `pytest tests/test_smoke_content_ops_live_generation.py tests/test_extracted_blog_blueprint_ingest.py -q` -> 21 passed.
- `bash scripts/local_pr_review.sh origin/main` -> passed.
- After review fix: `pytest tests/test_smoke_content_ops_live_generation.py tests/test_extracted_blog_blueprint_postgres.py -q` -> 17 passed.
- After review fix: `python -m py_compile scripts/smoke_content_ops_live_generation.py tests/test_smoke_content_ops_live_generation.py extracted_content_pipeline/blog_blueprint_postgres.py tests/test_extracted_blog_blueprint_postgres.py` -> passed.
- After review fix: `git diff --check` -> passed.
- After review fix: `bash scripts/validate_extracted_content_pipeline.sh` -> passed.
- After review fix: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -> passed.
- After review fix: `python scripts/audit_extracted_standalone.py --fail-on-debt` -> passed.
- After review fix: `bash scripts/check_ascii_python.sh` -> passed.
- After review fix: `bash scripts/local_pr_review.sh origin/main` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~120 |
| Smoke script | ~100 |
| Repository | ~15 |
| Tests | ~90 |
| Docs | ~60 |
| **Total** | **~385** |
