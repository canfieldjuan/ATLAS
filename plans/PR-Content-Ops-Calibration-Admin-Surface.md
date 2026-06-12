# PR-Content-Ops-Calibration-Admin-Surface

## Why this slice exists

Slice B (#1495) persisted calibration anchors and made the verify flow read
them, but there is no way for a tenant to *curate* that library -- the rows can
only be written by hand in the database. This slice C adds the admin write
surface: a tenant-scoped, admin-gated FastAPI CRUD router for calibration
examples, plus the repository write functions behind it. The verify MCP surface
is deliberately verify-only, so curation lives as an authenticated API, mirroring
the existing `content_ops_brand_voice_profiles` / claim-registry admin pattern.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Repo write CRUD in `_content_ops_calibration_library.py`: a display record,
   payload validation, and `create` / `update` / `list_records` / `archive`
   functions mirroring the claim-registry repository.
2. `atlas_brain/api/content_ops_calibration_library.py`: a tenant-scoped admin
   router (list / create / update / archive) with the shared admin gate and
   account-UUID scoping, mirroring the brand-voice router.
3. Register the router in the api aggregator with the same
   `_capture_content_ops_auth_user` auth dependency.

### Files touched

- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `atlas_brain/_content_ops_calibration_library.py`
- `atlas_brain/api/__init__.py`
- `atlas_brain/api/content_ops_calibration_library.py`
- `plans/INDEX.md`
- `plans/PR-Content-Ops-Calibration-Admin-Surface.md`
- `plans/archive/PR-Content-Ops-Calibration-Reader-Persistence.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_calibration_library.py`
- `tests/test_content_ops_calibration_library_api.py`

### Review Contract

Acceptance criteria:
- Create/update validate the payload (required example_id/excerpt/reasoning, a
  known label) and reject invalid input with `ValueError` -> HTTP 400 before any
  DB write.
- All routes are tenant-scoped by `account_id`; create/update/archive require an
  admin/owner role (403 otherwise); list requires only auth.
- Update/archive of a missing or cross-tenant row returns 404; a unique-id
  conflict returns 409.
- An invalid tenant scope returns 401; an invalid path id returns 404.
- The repository functions mirror the claim-registry CRUD (soft-delete, tenant
  scoping in the WHERE clause).

Affected surfaces: the calibration repository (write functions) + a new admin
API router + its registration. No change to the verify MCP surface, the reader,
the review logic, or the migration.

Risk areas: tenant scoping in every write; the admin gate; label validation
against the enum; the DB unique-violation -> 409 translation.

Reviewer rules triggered: R1, R2, R3, R5, R10, R14.

## Mechanism

The repository gains a `ContentOpsCalibrationLibraryRecord` display dataclass and
`_validated_example` (required example_id/excerpt/reasoning, label must be a
`CalibrationLabel` value), then `create_calibration_example`,
`update_calibration_example`, `list_calibration_example_records`, and
`archive_calibration_example` -- byte-for-byte the claim-registry CRUD shape
(tenant-scoped WHERE, soft-delete via `archived_at`, `None`/`False` on
missing/cross-tenant rows).

The router factory `create_content_ops_calibration_library_router` takes a
`pool_provider` + `auth_dependency` (the same pattern as brand-voice), exposes
`GET/POST/PUT/DELETE /content-ops/calibration-library`, gates writes through
`_require_calibration_admin`, scopes by `_account_uuid(user)`, and maps records
to a `CalibrationExampleView`. A `ValueError` becomes 400, a missing row 404, and
a DB `UniqueViolationError` (matched by class name to avoid a hard asyncpg
import) becomes 409. It is registered in `atlas_brain/api/__init__.py` alongside the other
content-ops admin routers with `_capture_content_ops_auth_user`.

## Intentional

- **Admin API, not an MCP tool.** The marketer verify MCP is verify-only by
  contract; curation is a separate authenticated surface, following the
  established brand-voice/claim-registry admin pattern.
- **Unique-violation matched by class name.** Avoids a hard `asyncpg` import in
  the router (kept out of the lightweight import path) while still translating
  the conflict to 409.
- **The router test loads the module by file path** (importlib), like the
  brand-voice API test, to bypass the heavy `atlas_brain.api.__init__` chain
  (asyncpg/numpy) so it runs without a full runtime install.
- **Calibration-only, not joint with the claim registry.** The admin pattern is
  already established by brand-voice; mirroring it for calibration is the thin
  slice. The claim registry's own unwired write surface stays a separate
  follow-up.

## Deferred

- Wiring an admin UI screen for the new endpoints.
- A seed/import path for bulk calibration curation.
- The claim registry's parallel admin write surface (separate follow-up).
- Parked hardening: none new this slice.

## Verification

- Reviewer rules triggered: R1, R2, R3, R5, R10, R14.
- Passed: pytest tests/test_content_ops_calibration_library.py
  tests/test_content_ops_calibration_library_api.py -- 27 passed (5 repo CRUD +
  10 router fixtures new).
- Passed: python3 scripts/audit_extracted_pipeline_ci_enrollment.py -- OK, 170
  matching tests are enrolled (new API test added to the runner + the content-ops
  review-workflow CI workflow).
- Passed: python3 scripts/audit_extracted_standalone.py --fail-on-debt -- 0 findings.
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_review_workflow_checks.yml` | 5 |
| `atlas_brain/_content_ops_calibration_library.py` | 212 |
| `atlas_brain/api/__init__.py` | 8 |
| `atlas_brain/api/content_ops_calibration_library.py` | 203 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Content-Ops-Calibration-Admin-Surface.md` | 119 |
| `plans/archive/PR-Content-Ops-Calibration-Reader-Persistence.md` | 0 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_calibration_library.py` | 108 |
| `tests/test_content_ops_calibration_library_api.py` | 192 |
| **Total** | **851** |
