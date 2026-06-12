# PR-Content-Ops-Calibration-Reader-Persistence

## Why this slice exists

Slice A (#1494) landed the `TenantCalibrationLibraryReader` seam with a no-op
default reader. This is slice B: the persistence behind it -- a tenant-scoped
`content_ops_calibration_library` table and a Postgres-backed repository
implementing the port -- so a marketer's curated calibration anchors actually
live server-side and surface in the verdict without being resent on every call.
It mirrors the proven claim-registry persistence (migration 334 +
`ContentOpsClaimRegistryRepository`), with the calibration-specific divergence:
anchors are evidence, not a gate, so the reader is best-effort.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Migration `335_content_ops_calibration_library.sql`: tenant-scoped table,
   teachable-only (excerpt + reasoning required), label CHECK matching the
   `CalibrationLabel` enum, soft-delete, partial active + unique indexes.
2. `atlas_brain/_content_ops_calibration_library.py`:
   `ContentOpsCalibrationLibraryRepository` implementing
   `list_calibration_examples(*, scope)` + the read query + row mapping.
3. Swap the verify server's `_get_calibration_reader()` from the no-op default to
   the repository (lazy `get_db_pool()`), exactly like `_get_registry_reader()`.

### Files touched

- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `atlas_brain/_content_ops_calibration_library.py`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `atlas_brain/storage/migrations/335_content_ops_calibration_library.sql`
- `plans/INDEX.md`
- `plans/PR-Content-Ops-Calibration-Reader-Persistence.md`
- `plans/archive/PR-Content-Ops-Calibration-Reader-Port.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_calibration_library.py`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

Acceptance criteria:
- The migration's label CHECK lists exactly the `CalibrationLabel` enum values
  (a drift test asserts the set equality).
- The reader returns the tenant's teachable examples, deduped by `example_id`,
  filtered to known labels; an unknown label or a row missing excerpt/reasoning
  is skipped.
- An invalid tenant scope returns `()` without querying (evidence, not a gate);
  a database read failure raises `ContentOpsCalibrationLibraryReadError`, which
  the review's merge helper catches and degrades on.
- The review surfaces repository anchors on the success path; a failing repo
  degrades to request-supplied anchors with the verdict unaffected.
- The verify server's default reader is now the repository (no override).

Affected surfaces: a new migration + host repository module; the verify server's
reader factory. No change to the review merge/degradation logic (slice A), the
verify tool args, or the verdict computation.

Risk areas: label CHECK <-> enum drift; tenant-scope UUID handling; the reader's
best-effort divergence from the fail-closed registry reader.

Reviewer rules triggered: R1, R2, R4, R5, R10, R14.

## Mechanism

The migration mirrors `334_content_ops_claim_registry.sql`: `account_id` FK to
`saas_accounts`, soft-delete `archived_at`, a partial unique index on
`(account_id, lower(btrim(example_id))) WHERE archived_at IS NULL`, and a partial
active index. It is teachable-only -- `excerpt` and `reasoning` are `NOT NULL`
with non-empty CHECKs -- because a non-teachable anchor cannot illustrate a
failure mode, and the verify selector already drops them. The `label` CHECK
enumerates the nine `CalibrationLabel` values; a test asserts that set equals the
enum to catch drift.

`ContentOpsCalibrationLibraryRepository.list_calibration_examples` resolves the
account UUID (returning `()` for an unusable scope rather than raising -- the
calibration divergence from the fail-closed registry reader), reads active rows
newest-first, and maps each to a `CalibrationExample`, skipping rows with an
unknown label or missing text and deduping by `example_id`. A database error is
wrapped in `ContentOpsCalibrationLibraryReadError`; the review's
`_merged_calibration_examples` (slice A) catches any exception and degrades to
request-supplied anchors. `_get_calibration_reader()` now constructs the
repository with `get_db_pool()`, the one-line swap slice A was built for.

## Intentional

- **Read-only repository.** The write surface (create/update/archive) lands with
  the admin slice (C); shipping write functions with no caller here would be dead
  code. The registry bundles them, but this slice keeps the read path clean.
- **Invalid scope returns `()`, does not raise.** Calibration is evidence; an
  unusable scope means "no server anchors", consistent with the reader being
  best-effort. Database errors still raise (and the review degrades on them).
- **Teachable-only at the DB.** The store only holds anchors that can teach;
  enforcing it in schema keeps the table consumer-shaped and the reader simple.
- **Defensive row mapping** despite DB CHECKs -- a row that lost its text or a
  recognized label is skipped, never surfaced as a broken anchor.

## Deferred

- **Slice C -- admin write surface:** create/update/archive per tenant + the
  operator path, decided jointly with the claim registry's unwired write gap.
- **Anchor metadata / verdict / failure_category columns:** omitted until a
  consumer needs them (the verify selector uses only label + excerpt + reasoning).
- **Parked hardening:** none new this slice.

## Verification

- Reviewer rules triggered: R1, R2, R4, R5, R10, R14.
- Passed: pytest of the calibration-repo, host-workflow, verify-MCP, claim-registry,
  and launcher-contract test files -- 124 + 7 passed.
- Passed: python3 scripts/audit_extracted_pipeline_ci_enrollment.py -- OK, 169
  matching tests are enrolled (includes the new repo test).
- Passed: python3 scripts/audit_extracted_standalone.py --fail-on-debt -- 0 findings.
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_review_workflow_checks.yml` | 7 |
| `atlas_brain/_content_ops_calibration_library.py` | 136 |
| `atlas_brain/mcp/content_ops_marketer_verify_server.py` | 20 |
| `atlas_brain/storage/migrations/335_content_ops_calibration_library.sql` | 44 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Content-Ops-Calibration-Reader-Persistence.md` | 129 |
| `plans/archive/PR-Content-Ops-Calibration-Reader-Port.md` | 0 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_calibration_library.py` | 210 |
| `tests/test_mcp_content_ops_marketer_verify.py` | 15 |
| **Total** | **565** |
