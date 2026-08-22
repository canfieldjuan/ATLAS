# PR-H18-Migration-387-Attestation-Precision

## Why this slice exists

The controlled Atlas missed-call-recovery deployment checkpoint for #2474
correctly refused pending migration 389. The target-confirmed, read-only H-18
preflight showed that the existing source-controlled reconciliation for
`387_eom_recurring_invoice_dedup_recovery` can never attest the canonical
ledger: the database recorded `2026-08-21T01:30:46.082989Z`, while the evidence
record rounds it to whole seconds and the implementation deliberately compares
the timestamp exactly. This is a real evidence-record defect, not a reason to
relax migration admission.

The same receipt also reports a separate migration-386 mismatch and six
missing-source records. They remain intentionally blocking and are owned by
#2476; this thin slice repairs only the proven bad data in the existing 387
source-controlled record.

### Problem-derived contract

- Root cause: `MIGRATION_387_RECONCILIATION.observed_applied_at` loses the
  microseconds stored by PostgreSQL `CURRENT_TIMESTAMP`, while
  `_attest_migration_387` requires exact timestamp equality. Consequently the
  one reconciliation that was intended to be admissible is permanently
  `not_attested` on its documented target.
- Correct fix must touch/change: the immutable 387 evidence literal must retain
  the observed UTC microseconds; focused read-only preflight regressions must
  prove exact live-shaped evidence attests and a seconds-truncated timestamp is
  rejected. Required previous-slice plan retirement moves only the merged
  missed-call-recovery plan into `plans/archive/` and regenerates the index.
- Must not change: the migration runner's global fail-closed admission policy,
  SQL migrations, `schema_migrations`, database contents, migration-386 or
  missing-source reconciliation, provider/lead/payment behavior, deployment
  configuration, or the active tracker/Website CRM lanes.

## Scope (this PR)

Ownership lane: eom/migration-content-integrity
Slice phase: Production hardening

1. Replace the truncated 387 UTC timestamp with the exact target-confirmed
   microsecond value in the source-controlled reconciliation record.
2. Add a regression that proves the precise record attests and a value truncated
   to seconds remains non-attested with no database write.
3. Complete the required post-merge archive of the merged #2475 plan only;
   do not bulk-archive concurrent plans.

### Review Contract

- Acceptance criteria:
  1. `MIGRATION_387_RECONCILIATION.observed_applied_at` equals the
     target-confirmed UTC timestamp including microseconds, settled by
     `tests/test_migration_content_integrity_preflight.py`.
  2. The real read-only preflight helper returns `attested` evidence for a
     ledger row equal to that exact record, while keeping the base integrity
     report `unresolved_drift`; settled by the existing attestation test and
     its fake transaction's zero `execute_calls` assertion.
  3. A seconds-truncated otherwise-identical ledger timestamp stays
     `not_attested` and performs no write; settled by the new regression test.
  4. No other historic evidence is cleared: the runner still blocks any
     remaining mismatched/missing-source record, settled by the existing
     multi-evidence migration-runner tests and a post-change target-confirmed
     read-only receipt recorded on #2476.
  5. The only plan retirement is `PR-EOM-Missed-Call-Recovery.md`; the generated
     index is synchronized by `scripts/archive_plans.py index`.
- Reachability proof: `run_migration_content_integrity_preflight` is the same
  read-only function behind `scripts/check_migration_content_integrity.py` and
  the `run_migrations` admission path. Its observable result is the structured
  attestation payload; no public/customer surface is added.
- Affected surfaces: `atlas_brain.storage.migrations.reconciliation`, the
  migration-content preflight tests, and mandatory plan retirement metadata.
- Risk areas: accidentally broadening admission, rounding another evidence
  timestamp, a false positive from a seconds-only row, and unrelated plan
  archival.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: `_attest_migration_387` compares a source-controlled
  historical timestamp to the read-only ledger row; the evidence literal is
  corrected, while the comparison operation stays exact.
- Replaced-path behaviors: only the documented canonical microsecond value
  changes from false to true. A whole-second value, a different microsecond,
  duplicate row, digest change, package change, or failed schema predicate
  remains non-attested.
- Guard-relevant fields: ledger row cardinality, historical digest, packaged
  digest, exact aware UTC `applied_at`, source-order predicate, final recurring
  schema readiness, and zero active NULL-period recurring rows.
- Caller x input shape: the CLI preflight and `run_migrations` both consume the
  same attestation. For the exact 387 target it can clear only 387's mismatch;
  every other mismatch/missing source remains in the runner's refusal result.

### Deployed-config probing

- Deployed/default config values: no configuration changes. The canonical
  database target is read only through the existing typed `ATLAS_DB_*` settings.
- Explicit value probe: target-confirmed preflight runs with its log-safe target
  label and reports the 387 attestation payload.
- Absent value probe: existing config/default tests remain unchanged; an absent
  database configuration cannot make the evidence attested.
- Default-session/default-context probe: the preflight opens a read-only
  transaction and records no `execute` calls in focused tests.
- Side-effect ordering: all evidence reads occur before any pending migration
  is admitted; this PR itself runs no migration SQL.

### Files touched

- `atlas_brain/storage/migrations/reconciliation.py`
- `plans/INDEX.md`
- `plans/PR-H18-Migration-387-Attestation-Precision.md`
- `plans/archive/PR-EOM-Missed-Call-Recovery.md`
- `tests/test_migration_content_integrity_preflight.py`

## Mechanism

The reconciliation retains a target-confirmed UTC datetime literal with all
six recorded fractional-second digits. The existing equality comparison remains
unchanged, so source evidence stays exact instead of becoming a tolerance or an
allowlist. The new negative fixture proves that returning to a seconds-only
literal would stop attestation again. The old provider plan is moved verbatim
to the archive and the index is regenerated by the repository script.

## Intentional

- This does not make the preflight return overall success: it continues to
  report `unresolved_drift` while the immutable 387 evidence is separately
  eligible for runner admission.
- The timestamp is exact target evidence, not a generic precision-normalizing
  rule. Other reconciliation records must separately prove their own precision
  and catalog predicate.
- Migration 389 remains blocked after this PR until #2476 resolves the distinct
  migration-386 and missing-source evidence.

## Deferred

- #2476 owns the migration-386 historical/final-schema recovery design and
  every missing-source record. It must not be folded into this one-literal fix.
- The missed-call provider deployment, tracker relay, and Website CRM action
  remain blocked by the unchanged admission guard and retain their existing
  safe disabled state.

Parking predicate: any additional historical migration identity, reconciliation
predicate, runner-policy, SQL, provider, or consumer change is parked unless it
is required to correct this exact 387 evidence literal.

Parked hardening: #2476 is the required active follow-up; no additional
hardening is hidden in this PR.

## Verification

- Passed: Ruff lint and byte compilation for
  `atlas_brain/storage/migrations/reconciliation.py` and
  `tests/test_migration_content_integrity_preflight.py`; the focused preflight
  and runner pytest selection passed (`78 passed, 1 skipped`); and the Git diff
  whitespace check passed.
- Passed: target-confirmed production read-only preflight from this candidate
  worktree reports migration 387 `attested` while the overall result remains
  `unresolved_drift` / exit `2` because the independent 386 and missing-source
  blockers remain. The command wrote no database state.
- `ruff format --check` is not a gate for this slice: it proposes formatting
  pre-existing layout outside the root-cause hunk. The final diff preserves
  that layout and `ruff check` passes.
- Pending before push: plan synchronization and repository PR/body audits.
  GitHub owns the broad unit and disposable-PostgreSQL workflows.
- Deployment proof is intentionally deferred: this PR cannot deploy 389 while
  the unrelated admission evidence remains unresolved.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/storage/migrations/reconciliation.py` | 15 |
| `plans/INDEX.md` | 3 |
| `plans/PR-H18-Migration-387-Attestation-Precision.md` | 177 |
| `plans/archive/PR-EOM-Missed-Call-Recovery.md` | 0 |
| `tests/test_migration_content_integrity_preflight.py` | 35 |
| **Total** | **230** |
