# PR-Deflection-Report-Retention

## Why this slice exists

Issue #1734 closed the report-storage privacy exposure in two upstream steps:
#1730 scrubbed paid and free report payloads before persistence, and #1738
widened the local detector behind that same scrub seam. The remaining privacy
lifecycle gap is retention: `content_ops_deflection_reports` stores `artifact`,
`snapshot`, and `delivery_email` indefinitely, and the report store exposes no
retention/deletion primitive for production rows.

Root cause: the persisted deflection report store was built around create/read
and paid-state transitions, with only a smoke-test cleanup path deleting rows
ad hoc. Because deletion does not live at the store boundary, any future cron,
operator script, or account deletion path would have to invent its own SQL and
could drift from tenant/request semantics.

This change fixes the upstream root that is safe in this slice: add a tested
store-level retention deletion boundary and a guarded operator runner that can
dry-run or apply that boundary. It does not silently choose the product retention
window or enable automatic scheduled deletion; that policy/scheduler decision is
deferred until the operator chooses the access window buyers should retain.

The diff is over the 400 LOC soft cap because deletion tooling needs both sides
of the guard proved: dry-run versus destructive apply, old-row deletion versus
fresh-row preservation, invalid argument rejection before DB access, migration
enrollment, and workflow path enrollment. Splitting those probes away would make
the data-loss boundary weaker than the code it is supposed to guard.

## Scope (this PR)

Ownership lane: content-ops/deflection-privacy
Slice phase: Production hardening

1. Add a retention/deletion API to the deflection report artifact store for rows
   older than an explicit cutoff, with a matching Postgres implementation and
   in-memory behavior for focused tests.
2. Add a guarded operator runner for `content_ops_deflection_reports` retention:
   dry-run by default, explicit `--confirm-delete` before destructive writes,
   required positive retention window, production-safe DB URL env/file inputs,
   optional limit, preflighted output, and JSON summary output.
3. Add an additive concurrent migration index for global cutoff deletes by
   `created_at`.
4. Add focused tests proving old rows are counted/deleted, fresh rows are
   preserved, limits are honored, dry-run does not delete, output and secret
   boundaries are guarded, and invalid/falsy destructive inputs fail closed.

### Review Contract

- Acceptance criteria:
  - [ ] A Postgres store caller can count and delete only rows with
        `created_at` older than the supplied cutoff.
  - [ ] The in-memory store mirrors the same old-row deletion behavior for
        deterministic tests.
  - [ ] The operator runner defaults to dry-run and never deletes unless
        `--confirm-delete` is present.
  - [ ] The runner rejects non-positive retention windows and non-positive
        limits before touching the database.
  - [ ] The runner accepts an explicit non-argv database URL source for
        production purge runs.
  - [ ] The runner preflights the output target before deleting rows and falls
        back to stdout if a late output write fails.
  - [ ] Deletion summaries expose cutoff, retention days, dry-run/apply state,
        candidate count, deleted count, and limit without emitting report
        payloads or delivery email values.
  - [ ] The destructive delete count parser fails closed instead of silently
        reporting zero when the database command tag is malformed.
  - [ ] Existing save/list/fetch/paid-state behavior stays unchanged.
- Affected surfaces: DB retention, operator script, deflection report store,
  tests, migration index.
- Risk areas: data-loss, privacy/security, migration, operator ergonomics,
  CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R4, R6, R8, R10, R12, R14.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `.github/workflows/atlas_deflection_migration_apply_checks.yml`
- `.github/workflows/extracted_pipeline_checks.yml`
- `atlas_brain/storage/migrations/339_content_ops_deflection_reports_retention_index.sql`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Report-Retention.md`
- `scripts/purge_content_ops_deflection_reports.py`
- `tests/maturity_sweep/deflection_product_surface_manifest.json`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_deflection_migrations_apply.py`

## Mechanism

The store gets a small retention surface:

- Count rows older than a supplied UTC cutoff.
- Delete rows older than that cutoff, optionally bounded by a limit.

The Postgres adapter performs the delete in one statement using `created_at` and
returns the parsed deleted count from the command tag. If the database command
tag is malformed, the parser raises instead of returning zero so a destructive
purge cannot be silently under-reported. When a limit is supplied, it deletes by
a limited CTE over `(account_id, request_id)` so retries are bounded and
repeatable. The in-memory store records a comparable created-at value for tests
and applies the same cutoff/limit behavior without touching production code
paths.

The runner computes the cutoff from an explicit `--retention-days` value, opens
Postgres only after preflight validation, resolves the DB URL from exactly one
explicit source, reports candidate rows in dry-run, and deletes only when
`--confirm-delete` is set. The production-safe DB URL sources are env-var name
or file path so the DSN does not need to appear in process argv. The output path
is prepared before delete; if a late write fails after the purge, the summary is
printed to stdout instead of making a successful purge look failed. The JSON
summary deliberately contains counts and policy values only; it does not print
`artifact`, `snapshot`, `delivery_email`, or request payload content.

## Intentional

- No automatic cron/scheduler is enabled in this PR. The safe root fix is the
  deletion boundary; the business retention window and automatic execution
  cadence are operator policy decisions.
- The runner requires an explicit retention window and exactly one explicit DB
  URL source. Direct URL argv remains available for local/disposable runs, while
  env/file sources are the production-safe operator path.
- This PR deletes whole report rows. It does not partially null individual JSON
  columns, because paid/free access semantics are row-based today.
- No account-deletion cascade is added here; that should call the same store
  boundary once the account-deletion workflow is scoped.

## Deferred

- Automatic scheduled retention once the operator chooses the buyer access
  window and cadence.
- Account-deletion integration for deflection report rows.

Parked hardening: none.

## Verification

- Focused retention and review-fix pytest command - 9 passed.
- Focused parser/output/index regression pytest command - 6 passed.
- Full deflection report plus migration apply pytest command - 83 passed, 2 skipped.
- Maturity sweep extracted-content ratchet command - passed.
- Maturity sweep deflection content-ops ratchet command - passed.
- Migration apply plus manifest pytest command - 4 passed, 2 skipped.
- Product surface manifest checker command - passed.
- Deflection report workflow pytest bundle - 58 passed.
- Python compile command for touched Python files - passed.
- Extracted content pipeline validation command - passed.
- Atlas reasoning import guard command - passed.
- Extracted standalone audit command - passed.
- ASCII Python command - passed.
- Extracted pipeline CI-enrollment audit command - passed.
- Local PR review command - pending rerun after verification text update.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_report_checks.yml` | 12 |
| `.github/workflows/atlas_deflection_migration_apply_checks.yml` | 2 |
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `atlas_brain/storage/migrations/339_content_ops_deflection_reports_retention_index.sql` | 4 |
| `extracted_content_pipeline/deflection_report_access.py` | 133 |
| `plans/PR-Deflection-Report-Retention.md` | 167 |
| `scripts/purge_content_ops_deflection_reports.py` | 234 |
| `tests/maturity_sweep/deflection_product_surface_manifest.json` | 1 |
| `tests/test_content_ops_deflection_report.py` | 303 |
| `tests/test_deflection_migrations_apply.py` | 25 |
| **Total** | **883** |
