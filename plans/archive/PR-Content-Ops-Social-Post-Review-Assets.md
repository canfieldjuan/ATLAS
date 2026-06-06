# PR: Content Ops Social Post Review Assets

## Why this slice exists

PR #1266 made `social_post` executable and PR #1270 put it into the review and
competitive input-package defaults. The remaining productization gap is that a
run returns social posts in the execution payload, but those drafts are not
persisted anywhere. Operators cannot review, approve, or export them later.

This slice is the persistence prerequisite for the review queue: generated
social posts are saved as tenant-scoped draft rows when DB-backed Content Ops
services are enabled. Keeping the API/UI switchboard for a follow-up preserves
the thin-slice boundary while making the next route slice mechanical.

The diff is over the 400 LOC soft cap because the persistence boundary is not
safe to split below the table/contract/adapter/save-hook/host-wiring/tests set.
A table without a save hook is dead schema; a save hook without the tenant
adapter is not reviewable; and host wiring without focused tests risks another
"declared but not exercised" generated-output gap.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add a package-owned `SocialPostDraft` persistence contract and Postgres
   adapter.
2. Add the package migration for `social_posts` draft rows.
3. Teach `SocialPostGenerationService` to persist generated posts when a
   repository is injected, returning `saved_ids` in the result.
4. Wire the host DB-backed service bundle to use the Postgres social-post
   repository when `enable_db_services=True`.
5. Add focused package and host tests proving tenant-scoped persistence and the
   non-persistent fallback.

### Files touched

- `plans/PR-Content-Ops-Social-Post-Review-Assets.md`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/social_post_ports.py`
- `extracted_content_pipeline/social_post_postgres.py`
- `extracted_content_pipeline/social_post_generation.py`
- `extracted_content_pipeline/storage/migrations/331_social_posts.sql`
- `atlas_brain/_content_ops_services.py`
- `tests/test_extracted_social_post_generation.py`
- `tests/test_extracted_social_post_postgres.py`
- `tests/test_atlas_content_ops_execution_services.py`

## Mechanism

`SocialPostGenerationService` already produces a tuple of post dictionaries from
normalized source material. This slice converts those dictionaries into
`SocialPostDraft` values and calls:

```python
await social_posts.save_drafts(drafts, scope=scope)
```

only when a repository is injected. The default constructor keeps today's
deterministic in-memory behavior, so dependency-light extracted tests and hosts
without DB services still receive the same `posts` payload.

The Postgres adapter mirrors the existing generated-asset adapters: tenant scope
is written to `account_id`, original source metadata remains in JSONB
`metadata`, drafts start as `status = 'draft'`, and update/list helpers scope by
`account_id`.

## Intentional

- No generated-assets API or frontend tab in this PR. The table and repository
  must exist before the route switchboard can list/review/export social posts.
- No LLM or quality-gate changes. `social_post` remains deterministic and uses
  the same source-material checks from #1266.
- No fallback to unscoped/global persistence. Host persistence only happens when
  `enable_db_services=True` supplies the tenant-aware DB-backed service bundle.

## Deferred

- Next PR: add `social_post` to generated-assets API list/export/review/batch
  switchboards and expose it to the asset review UI.
- Ad copy and stat/quote card package defaults remain future output slices.

## Parked hardening

None.

## Verification

- Passed: pytest tests/test_extracted_social_post_generation.py tests/test_extracted_social_post_postgres.py -q (9 passed)
- Passed: pytest tests/test_atlas_content_ops_execution_services.py -q (20 passed)
- Passed: python -m py_compile extracted_content_pipeline/social_post_generation.py extracted_content_pipeline/social_post_postgres.py extracted_content_pipeline/social_post_ports.py atlas_brain/_content_ops_services.py
- Passed: git diff --check
- Passed: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main (OK: 144 matching tests are enrolled.)
- Passed: bash scripts/validate_extracted_content_pipeline.sh
- Passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
- Passed: python scripts/audit_extracted_standalone.py --fail-on-debt
- Passed: bash scripts/check_ascii_python.sh
- Passed: bash scripts/run_extracted_pipeline_checks.sh (2972 passed, 10 skipped, 1 warning)
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-social-post-review-assets-pr-body.md

## Estimated diff size

Actual: 10 files, +783 / -12. This is above the 400 LOC soft cap because the
persistence prerequisite is indivisible: the table, port, adapter, service save
hook, host wiring, and focused tests have to land together or the next
generated-assets route slice has no durable draft source to read.

| Area | Estimated LOC |
|---|---:|
| Social-post contract, adapter, migration, and manifest | ~320 |
| Generator save hook and host wiring | ~80 |
| Focused package and host tests | ~278 |
| Plan doc | ~94 |
| **Total** | **~795** |
