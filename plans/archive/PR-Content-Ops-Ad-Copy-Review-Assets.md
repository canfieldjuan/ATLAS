# PR: Content Ops Ad Copy Review Assets

## Why this slice exists

PR #1275 made `ad_copy` executable for marketer review inputs and PR #1278
threaded ad-copy into the review and competitive package defaults. The product
gap now matches the social-post gap closed by PR #1271: a run returns ad-copy
drafts in the execution payload, but those drafts are not persisted anywhere.
Operators cannot review, approve, reject, export, or revisit them later.

This slice is the persistence prerequisite for the generated-asset review
queue. It saves generated ad-copy drafts as tenant-scoped rows when DB-backed
Content Ops services are enabled, while leaving the default deterministic
service dependency-free. The generated-assets route and UI switchboard can then
be added in the next slice without inventing storage in the API layer.

The diff is expected to exceed the 400 LOC soft cap for the same reason as the
social-post persistence slice: schema, package port, adapter, generator save
hook, host wiring, and focused tests need to land together or the persistence
boundary is either dead schema or unexercised wiring.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add a package-owned `AdCopyDraft` persistence contract and Postgres adapter.
2. Add the package migration for `ad_copy_drafts` review rows.
3. Teach `AdCopyGenerationService` to persist generated ads when a repository
   is injected, returning `saved_ids` in the result.
4. Wire the host DB-backed service bundle to use the Postgres ad-copy
   repository when `enable_db_services=True`.
5. Add focused package and host tests proving tenant-scoped persistence and the
   non-persistent fallback.

### Files touched

- `plans/PR-Content-Ops-Ad-Copy-Review-Assets.md`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/ad_copy_ports.py`
- `extracted_content_pipeline/ad_copy_postgres.py`
- `extracted_content_pipeline/ad_copy_generation.py`
- `extracted_content_pipeline/storage/migrations/332_ad_copy_drafts.sql`
- `atlas_brain/_content_ops_services.py`
- `tests/test_extracted_ad_copy_generation.py`
- `tests/test_extracted_ad_copy_postgres.py`
- `tests/test_atlas_content_ops_execution_services.py`

## Mechanism

`AdCopyGenerationService` already turns normalized source material into ad
dictionaries. This slice mirrors the social-post pattern: convert those
dictionaries into `AdCopyDraft` values and call:

```python
await ad_copy_drafts.save_drafts(drafts, scope=scope)
```

only when a repository is injected. The default constructor keeps today's
dependency-light extracted behavior, so hosts without DB services still receive
the same `ads` payload, now with an empty `saved_ids` list.

The Postgres adapter stores tenant scope in `account_id`, keeps the original
source ad in JSONB metadata, starts rows at `status = 'draft'`, and exposes the
same save/list/status update shape as the social-post adapter. Host wiring
injects the adapter only inside the existing `enable_db_services=True` branch.

## Intentional

- No generated-assets API or frontend tab in this PR. The storage source must
  exist before list/review/export routes can read ad-copy drafts.
- No LLM or quality-gate changes. `ad_copy` remains deterministic and uses the
  same source-material evidence parsing from #1275.
- No persistence when DB services are disabled. The default service remains
  dependency-free and returns empty `saved_ids`.

## Deferred

- Next PR: add `ad_copy` to generated-assets API list/export/review/batch
  switchboards and expose it to the asset review UI.

## Parked hardening

None.

## Verification

- Passed: pytest tests/test_extracted_ad_copy_generation.py tests/test_extracted_ad_copy_postgres.py -q (9 passed)
- Passed: pytest tests/test_atlas_content_ops_execution_services.py -q (22 passed)
- Passed: python -m py_compile extracted_content_pipeline/ad_copy_generation.py extracted_content_pipeline/ad_copy_postgres.py extracted_content_pipeline/ad_copy_ports.py atlas_brain/_content_ops_services.py
- Passed: git diff --check
- Passed: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main (OK: 144 matching tests are enrolled.)
- Passed: bash scripts/validate_extracted_content_pipeline.sh
- Passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
- Passed: python scripts/audit_extracted_standalone.py --fail-on-debt
- Passed: bash scripts/check_ascii_python.sh
- Passed: bash scripts/run_extracted_pipeline_checks.sh (2988 passed, 10 skipped, 1 warning)
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-ad-copy-review-assets-pr-body.md

## Estimated diff size

Actual: 10 files, +815 / -14. This is above the 400 LOC soft cap because the
ad-copy persistence prerequisite is indivisible: the table, port, adapter,
service save hook, host wiring, and focused tests have to land together or the
next generated-assets route slice has no durable draft source to read.

| Area | Estimated LOC |
|---|---:|
| Ad-copy contract, adapter, migration, and manifest | ~320 |
| Generator save hook and host wiring | ~85 |
| Focused package and host tests | ~260 |
| Plan doc | ~85 |
| **Total** | **~815** |
