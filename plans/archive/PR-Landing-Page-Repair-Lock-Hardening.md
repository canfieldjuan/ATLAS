# PR: Landing Page Repair Lock Hardening

## Why this slice exists

HARDENING.md has two parked items from the landing-page repair session:

1. Remove the rollout-only legacy advisory lock that still uses `hashtext()`.
2. Stop holding a pooled DB connection during the LLM repair call.

The second item cannot be solved by keeping session advisory locks, because a
Postgres session advisory lock only protects work while its connection remains
checked out. This slice replaces the long-running advisory lock with an atomic
tenant-scoped repair claim stored on the landing-page row metadata. The claim
uses a token and expiry so one repair can proceed without holding a pool
connection while the LLM runs.

The diff is expected to land slightly above 400 LOC because the two parked
items share the same repair concurrency path; splitting them would keep the
legacy advisory lock or connection-hold behavior alive between slices.

Ownership lane: content-ops/landing-page-repair-session

## Scope (this PR)

1. Add tenant-scoped landing-page repair claim/release methods to the Postgres
   landing-page repository.
2. Use the metadata claim in the landing-page repair API route before calling
   the LLM repair service.
3. Fence the repaired draft write with the active repair claim token.
4. Remove the legacy `hashtext()` and widened `hashtextextended()` advisory lock
   path from generated-asset repair.
5. Log release misses when a claim token no longer matches during success or
   cleanup, so TTL-steal/no-op release anomalies stay operator-visible.
6. Update focused API and repository tests for claim conflict, release miss,
   optional real-Postgres contract coverage, and connection-free repair
   execution.
7. Remove the drained landing-page repair items from `HARDENING.md`.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Repair-Lock-Hardening.md` | Plan doc for draining the landing-page repair hardening items. |
| `HARDENING.md` | Remove the drained landing-page repair session entries. |
| `extracted_content_pipeline/api/generated_assets.py` | Replace the advisory lock context with a metadata repair claim in the repair route. |
| `extracted_content_pipeline/manifest.json` | Claim the landing-page Postgres adapter as extracted-owned. |
| `extracted_content_pipeline/landing_page_ports.py` | Document the optional repair-claim fence on landing-page draft updates. |
| `extracted_content_pipeline/landing_page_postgres.py` | Add atomic claim/release helpers for landing-page repair. |
| `tests/test_extracted_content_asset_api.py` | Update repair route coverage for claim conflicts and no checked-out lock connection. |
| `tests/test_extracted_landing_page_postgres.py` | Cover repair claim SQL, release SQL, miss behavior, and optional real-Postgres claim semantics. |

## Mechanism

PostgresLandingPageRepository.claim_repair(...) performs one atomic
UPDATE ... WHERE ... RETURNING against the tenant-scoped landing page row. It
writes landing_page_repair_claim into metadata with a random token and
expiry. The update only succeeds when the row is not approved and no active
claim exists, or when the existing claim has expired.

repair_landing_page_draft(...) claims the row before resolving LLM or skill
providers so duplicate repair attempts fail fast with 409 before provider work.
The repair service receives a claimed repository wrapper whose update_draft
method passes the active repair token into PostgresLandingPageRepository. That
makes the final repaired-content write conditional on the same claim still
belonging to the current worker. The route releases the claim in a `finally`
block and logs a warning when release returns `False`, which means the claim was
already cleared or the token no longer matched. On success, release happens
before the refreshed review row is loaded so transient claim metadata is not
returned to operators.

## Intentional

- No schema migration. The lock state lives in the existing JSONB metadata
  column because this is a per-row transient repair claim.
- No advisory lock compatibility path. The old rollout transition is complete,
  and retaining the legacy path is the parked debt this slice is draining.
- The claim has an expiry so a crashed worker does not block future repair
  forever.
- The optional real-Postgres contract test skips when no `EXTRACTED_DATABASE_URL`
  or `DATABASE_URL` is available; the normal unit path remains DB-free.

## Deferred

- Parked hardening: none. This slice drains the landing-page repair session
  entries currently in HARDENING.md.

## Verification

- pytest tests/test_extracted_content_asset_api.py tests/test_extracted_landing_page_postgres.py -q -> 77 passed, 1 skipped.
- bash scripts/validate_extracted_content_pipeline.sh -> passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -> passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -> passed.
- bash scripts/check_ascii_python.sh -> passed.
- python -m py_compile extracted_content_pipeline/api/generated_assets.py extracted_content_pipeline/landing_page_postgres.py tests/test_extracted_content_asset_api.py tests/test_extracted_landing_page_postgres.py -> passed.
- bash scripts/local_pr_review.sh -> passed with non-blocking warnings for diff-size
  estimate drift and overlap with open PR #812 in
  tests/test_extracted_content_asset_api.py.

## Estimated diff size

| File | Estimated LOC |
| --- | ---: |
| `extracted_content_pipeline/api/generated_assets.py` | 150 |
| `extracted_content_pipeline/manifest.json` | 5 |
| `extracted_content_pipeline/landing_page_ports.py` | 15 |
| `extracted_content_pipeline/landing_page_postgres.py` | 125 |
| `tests/test_extracted_content_asset_api.py` | 180 |
| `tests/test_extracted_landing_page_postgres.py` | 240 |
| `HARDENING.md` | 20 |
| `plans/PR-Landing-Page-Repair-Lock-Hardening.md` | 100 |
| **Total** | **835** |
