# PR-Deflection-Delta-Persistence

## Why this slice exists

#1316 D1 now computes a pure `deflection_delta.v1` payload, but the result is
still ephemeral. A monthly subscription needs an auditable stored artifact keyed
to the paid report pair that produced it, and it needs a safe baseline-selection
primitive before any API/MCP/email surface can read deltas.

Root cause: the codebase can compare two `deflection.v1` models, but the
deflection report store has no delta table, no tenant-scoped delta record, and
no previous-paid-report selector. This PR fixes that root at the storage/access
boundary only. Public read surfaces, autonomous monthly runs, and delivery stay
deferred.

Diff-size note: this exceeds the 400 LOC target because the smallest useful D2
storage slice needs the migration, both store adapters, the compute/persist
helper, migration coverage, and CI-enrolled tenant/paid-gating tests together.
Splitting the tests or migration enrollment would leave the new paid storage
boundary under-protected.

## Scope (this PR)

Ownership lane: issue-1316/deflection-delta-persistence
Slice phase: Vertical slice

1. Add a `content_ops_deflection_deltas` migration keyed by tenant,
   `current_request_id`, and `baseline_request_id`.
2. Extend the deflection artifact store with in-memory and Postgres methods to:
   select the previous paid report for a current paid report; save a
   `deflection_delta.v1` payload; and fetch a persisted delta by pair.
3. Add a small async helper that computes and persists the previous-paid-report
   delta using the D1 pure core and the store boundary.
4. Add focused tests for tenant scoping, paid gating, ordering, missing
   baseline/current cases, Postgres SQL shape, and migration coverage.

### Review Contract

Acceptance criteria:
- Baseline selection is tenant-scoped and paid-only; unpaid, other-tenant, same
  request, and newer reports are not selected.
- The previous baseline is selected by `created_at` before the current report,
  newest first, with `None` when no comparable paid baseline exists.
- Persisted delta records are keyed by `(account_id, current_request_id,
  baseline_request_id)` and store only the computed `deflection_delta.v1`
  payload, not raw ticket exports or new customer-facing fields.
- The compute-and-save helper returns `None` instead of fabricating a delta when
  current/baseline report models are absent or unpaid.
- Postgres methods use the existing `DeflectionReportArtifactStore` boundary and
  never query outside `account_id`.

Affected surfaces:
- `atlas_brain/storage/migrations/340_content_ops_deflection_deltas.sql`
- `extracted_content_pipeline/deflection_report_access.py`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_deflection_migrations_apply.py`
- `tests/maturity_sweep/deflection_product_surface_manifest.json`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/extracted_pipeline_checks.yml`

Risk areas:
- Accidentally diffing across tenants.
- Treating unpaid reports as paid monthly history.
- Persisting an arbitrary delta for a missing or schema-drifted report model.
- Over-reaching into D3 read surfaces before the persistence contract is proven.

Reviewer rules triggered: R1, R2, R4, R8, R10, R13, R14.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `atlas_brain/storage/migrations/340_content_ops_deflection_deltas.sql`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Delta-Persistence.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/maturity_sweep/deflection_product_surface_manifest.json`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_deflection_migrations_apply.py`

## Mechanism

The migration creates an idempotent delta table with composite foreign keys back
to `content_ops_deflection_reports`, so deleting a source report removes derived
deltas. The primary key is the tenant/current/baseline pair, which makes
recomputing the same pair idempotent while preserving the pair's audit identity.

The store method `select_previous_paid_report(...)` uses the current report's
`created_at` as the comparison anchor and only considers paid reports for the
same `account_id`. The helper `compute_and_save_previous_deflection_delta(...)`
loads the current and selected baseline report models through
`DeflectionReportAccessRecord.report_model()`, calls
`compute_deflection_delta(...)`, then saves and returns the stored delta record.

## Intentional

- No public API, MCP tool, email, PDF, result-page, or autonomous monthly task in
  this slice.
- Baseline selection is intentionally the immediately previous paid report by
  `created_at`. Source-window/calendar-month matching and explicit override are
  deferred until we have the persisted primitive.
- The table stores the computed delta payload, not raw evidence export rows or
  source-ticket bodies.
- Recomputing the same current/baseline pair updates the delta payload in place
  for idempotency. Multi-version delta history is deferred until there is a
  product need for historical recompute audit trails.

## Deferred

- #1316 D3 paid-gated API/MCP read surface.
- #1316 D4 monthly automation and delivery/upsell email.
- Calendar/source-window baseline resolver and explicit baseline override.
- Multi-version recompute history for the same current/baseline pair.

Parked hardening: none.

## Verification

- Focused delta persistence pytest -- 5 passed.
- Focused delta/migration pytest -- 14 passed, 2 skipped.
- Python compile for touched Python files -- passed.
- Deflection product surface manifest check -- passed.
- Extracted CI enrollment audit -- 190 matching tests enrolled.
- Broad extracted-content maturity ratchet -- passed.
- Deflection content-ops maturity ratchet -- passed.
- Extracted content pipeline validation -- passed.
- Extracted reasoning-import guard -- clean.
- Extracted standalone audit -- Atlas runtime import findings: 0.
- ASCII Python policy check -- passed.
- Full extracted pipeline bundle -- reasoning core 295 passed; extracted content
  4878 passed, 15 skipped.
- Pending before push: local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `atlas_brain/storage/migrations/340_content_ops_deflection_deltas.sql` | 24 |
| `extracted_content_pipeline/deflection_report_access.py` | 287 |
| `plans/PR-Deflection-Delta-Persistence.md` | 145 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/maturity_sweep/deflection_product_surface_manifest.json` | 1 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 297 |
| `tests/test_deflection_migrations_apply.py` | 60 |
| **Total** | **817** |
