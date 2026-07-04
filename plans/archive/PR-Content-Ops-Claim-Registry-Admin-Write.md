# PR-Content-Ops-Claim-Registry-Admin-Write

## Why this slice exists

The calibration-library admin surface (#1496) explicitly deferred "the claim
registry's parallel admin write surface" as a follow-up. Today the claim
registry has a table (migration `334`) and a full repository CRUD
(`create/update/expire/archive/list_registry_claim_records` in
`atlas_brain/_content_ops_claim_registry.py`), but **no HTTP write path** -- a
tenant's approved-wording rows can only be written by hand in the database. This
slice closes that gap with a tenant-scoped, admin-gated FastAPI router, mirroring
the calibration-library and brand-voice-profiles admin pattern. The marketer
verify MCP surface is deliberately verify-only, so curation lives as an
authenticated API. No migration and no repo changes are needed -- this slice is
purely the router, its registration, and tests.

The ~640 LOC total is over the 400 soft cap, but the net new logic is small: the
two largest files (router + router test) mirror the already-reviewed
calibration-library slice (#1496) almost verbatim, differing only in the
registry's extra fields and the expire endpoint. Splitting transport from its
test would ship an untested router, so they land together.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. `atlas_brain/api/content_ops_claim_registry.py`: a tenant-scoped admin router
   (list / create / update / expire / archive) with the shared admin gate and
   account-UUID scoping, mirroring the calibration-library router.
2. Register the router in the api aggregator with the same
   `_capture_content_ops_auth_user` auth dependency.
3. Enroll the new router test in the content-ops review-workflow CI workflow.

### Files touched

- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `atlas_brain/api/__init__.py`
- `atlas_brain/api/content_ops_claim_registry.py`
- `plans/PR-Content-Ops-Claim-Registry-Admin-Write.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_claim_registry_api.py`

### Review Contract

Acceptance criteria:
- All routes are tenant-scoped by `account_id`; create/update/expire/archive
  require an admin/owner role (403 otherwise); list requires only auth.
- Create validates the payload via the existing repo `_validated_claim`
  (required registry_id + approved_wording, known risk tier) -> `ValueError` ->
  HTTP 400 before any DB write.
- Update/expire/archive of a missing or cross-tenant row returns 404; a
  registry-id unique conflict on create/update returns 409.
- The expire endpoint accepts an optional `expires_on`; a malformed date is 400,
  a missing one defers to the repo default (today).
- An invalid tenant scope returns 401; an invalid path id returns 404.
- No change to the verify MCP surface, the reader, the review logic, the
  repository, or the migration.

Affected surfaces: a new admin API router + its registration + the CI
enrollment. Pure addition.

Risk areas: tenant scoping in every route; the admin gate; the DB
unique-violation -> 409 translation (matched by class name to avoid a hard
asyncpg import); the optional-date parse on expire.

Reviewer rules triggered: R1, R2, R5, R14.

## Mechanism

The router factory `create_content_ops_claim_registry_router` takes a
`pool_provider` + `auth_dependency` (the same pattern as the calibration and
brand-voice routers), exposes `GET/POST/PUT/DELETE
/content-ops/claim-registry` plus a registry-specific
`POST /content-ops/claim-registry/{row_id}/expire`, gates writes through
`_require_claim_registry_admin`, scopes by `_account_uuid(user)`, and maps repo
records to a `ClaimRegistryView`. It delegates to the existing repo functions
(note the repo param is `claim_id`, not `example_row_id`). A `ValueError`
becomes 400, a missing row 404, and a DB `UniqueViolationError` (matched by
class name) becomes 409. It is registered in `atlas_brain/api/__init__.py`
alongside the other content-ops admin routers with
`_capture_content_ops_auth_user`.

## Intentional

- **Expire endpoint included.** Unlike the calibration template, the claim
  registry repo already supports `expire_registry_claim`, and expiry is a
  first-class claim mutation (a claim's approved wording can lapse without being
  archived), so the admin surface exposes it. Optional `expires_on`; blank/absent
  defers to the repo default.
- **Admin API, not an MCP tool.** The marketer verify MCP is verify-only by
  contract; curation is a separate authenticated surface, following the
  established brand-voice/calibration admin pattern.
- **Unique-violation matched by class name.** Avoids a hard `asyncpg` import in
  the router (kept out of the lightweight import path) while still translating
  the conflict to 409.
- **The router test loads the module by file path** (importlib), like the
  calibration and brand-voice API tests, to bypass the heavy
  `atlas_brain.api.__init__` chain (asyncpg/numpy) so it runs without a full
  runtime install.
- **No repo or migration change.** Both already exist (migration `334`, repo
  CRUD). This slice is the missing transport only.

## Deferred

- A claim-registry admin UI screen (deferred per operator: the calibration UI
  ships first; the claim-registry UI is a later follow-up).
- A seed/import path for bulk claim curation.
- Parked hardening: none new this slice.

## Verification

- Reviewer rules triggered: R1, R2, R5, R14.
- Passed: pytest tests/test_content_ops_claim_registry_api.py -- 18 passed
  (list, create-201/403/400/409, update-200/403/404/409, expire-200/403/404 +
  default + invalid-date-400, archive-204/403/404, 401 on bad tenant scope).
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.
- Passed: bash scripts/run_extracted_pipeline_checks.sh -- 3890 passed,
  10 skipped.
- The new test is enrolled in
  `.github/workflows/atlas_content_ops_review_workflow_checks.yml` (path filter +
  pytest run step), alongside the calibration API test.
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file
  tmp/content_ops_claim_registry_admin_write_pr_body.md.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_review_workflow_checks.yml` | 9 |
| `atlas_brain/api/__init__.py` | 8 |
| `atlas_brain/api/content_ops_claim_registry.py` | 257 |
| `plans/PR-Content-Ops-Claim-Registry-Admin-Write.md` | 136 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_claim_registry_api.py` | 285 |
| **Total** | **696** |
