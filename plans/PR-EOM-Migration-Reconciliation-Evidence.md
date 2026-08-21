# PR-EOM-Migration-Reconciliation-Evidence

## Why this slice exists

H-18 phase two is required by [#2461](https://github.com/canfieldjuan/ATLAS/issues/2461): the production checkpoint for #2452 found that the recorded migration-387 ledger digest and application time predate the earliest retained source revision, although the final recurring-invoice catalog is present. The read-only preflight in #2462 correctly reports that as unresolved drift, but its only durable evidence is an issue comment. A later operator or admission-policy slice needs a reviewed, source-controlled record of exactly what is known without pretending unavailable source bytes were verified.

The historical evidence is deliberately narrow: `387_eom_recurring_invoice_dedup_recovery` was recorded at `2026-08-20T20:30:46-05:00` with ledger digest `1dae95d216bfdc836461943af1c6ce382ff7dd21b92eff41d4c94088f72315b2`; the final packaged source digest is `f6382a07d807f7b38772e9823c66f1e47e4118841611e259220d9ab654c84f3d`; and the earliest retained final-source commit is `2026-08-21T03:21:35Z`. The original source bytes remain unavailable.

This exceeds the 400-line soft budget because the static evidence, the one real CLI reachability path, and the false-predicate/no-write matrix are indivisible: splitting them would either ship a historical exception record that no operator can inspect or a read-only command with no proof that it refuses false evidence. It introduces no migration, financial write, or adjacent subsystem.

### Problem-derived contract

- Root cause: the known migration-387 mismatch has no immutable, executable reconciliation record. Operators can see an unresolved digest but cannot distinguish this documented historical source gap from a new or altered mismatch, and a later policy cannot safely rely on issue prose alone.
- Correct fix must touch/change: add a source-controlled immutable 387 evidence record and a read-only, explicitly opted-in preflight attestation that verifies the recorded digest/time, current packaged digest, and current recurring-invoice catalog evidence. It must return the attestation alongside the existing report while retaining the existing unresolved-drift classification and exit status.
- Must not change: migration-387 SQL or any other migration, `schema_migrations`, invoices, payments, startup migration behavior, pending-migration selection, invoice writers, credentials, or default preflight output. Do not claim migration `388`, which is owned by #2465.

## Scope (this PR)

Ownership lane: eom/migration-content-integrity
Slice phase: Production hardening
Max files: 4

1. Add one immutable, named reconciliation record for the documented migration-387 historical source gap, including the two exact digests, observed UTC application instant, retained-source cutoff, and the permanent `historical_source_unavailable` source-evidence state.
2. Add `--attest-known-reconciliations` to the existing target-confirmed, read-only operator preflight. The opt-in result reports only whether exactly one named ledger row and the known record's ledger/package/time/catalog predicates are currently attested; it never changes report categories, relabels 387 as verified, or makes a mismatched run exit zero.
3. Reuse the existing recurring-invoice schema-readiness predicate and a bounded aggregate for active NULL-period recurring rows. The data probe runs only after schema readiness succeeds and returns no customer, invoice, payment, or credential data.
4. Add focused tests for the attested path, every false evidence side represented by the record, no-write/read-only execution, opt-in target admission, and the unchanged default output.

### Review Contract

- Acceptance criteria:
  - [ ] `run_migration_content_integrity_preflight(..., attest_known_reconciliations=True)` emits one named 387 reconciliation result only when exactly one named ledger row, its stored digest, observed application instant, packaged digest, retained-source ordering, `recurring_invoice_dedup_schema_ready`, and zero-active-NULL predicate all match; `tests/test_migration_content_integrity_preflight.py` proves the full attested result.
  - [ ] The attested result retains `source_verification="historical_source_unavailable"`, keeps 387 in `report.mismatched`, and returns `UNRESOLVED_DRIFT_EXIT`; `tests/test_migration_content_integrity_preflight.py` asserts each observable field and exit code.
  - [ ] Duplicate ledger rows, a changed ledger digest, application instant, package bytes, source-cutoff ordering, readiness result, or NULL-period aggregate produce `not_attested`, never a verified source claim; focused tests exercise the predicate failures.
  - [ ] The new catalog reads occur inside the existing `connection.transaction(readonly=True)` and perform no `execute`; the fake connection's query and write receipts settle that invariant.
  - [ ] Without the opt-in flag, the command's existing JSON payload and behavior are byte-for-byte unchanged in the established default tests; `--show-target` plus the attestation flag is rejected before any database connection.
  - [ ] The existing migration-integrity CI workflow runs the updated focused test file because `.github/workflows/atlas_migrations_runner_checks.yml` already watches `atlas_brain/storage/migrations/**`, `scripts/check_migration_content_integrity.py`, and `tests/test_migration_content_integrity_preflight.py`.
- Reachability proof: the real `scripts/check_migration_content_integrity.py` parser passes the opt-in to `_main`, then to `run_migration_content_integrity_preflight`; the focused test invokes that entrypoint and asserts the JSON artifact plus database receipts. This is an operator-only read-only command, not a product UI surface.
- Affected surfaces: migration-content preflight CLI, source-controlled migration-provenance artifact, read-only PostgreSQL catalog queries, structured JSON output, and the existing migration-runner CI path.
- Risk areas: migration/history safety, false historical-source verification, default-command backward compatibility, production catalog-read safety, target-confirmation admission, and future pending-migration policy interpretation.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: `scripts/check_migration_content_integrity.py:main` admits an opt-in reconciliation read only after the existing exact `--expected-target` confirmation; `--show-target` remains connection-free and rejects the incompatible opt-in.
- Replaced-path behaviors: none. With no opt-in, the current target confirmation, report categories, JSON fields, and exit mapping remain unchanged. With opt-in, mismatched/missing-source remains unresolved drift rather than becoming a policy exception.
- Guard-relevant fields: `--show-target`, `--expected-target`, `--attest-known-reconciliations`; immutable 387 migration name/digests/timestamps; catalog readiness and aggregate result.
- Caller x input shape: CLI with no target, incorrect target, correct target without opt-in, correct target with opt-in, and `--show-target` plus opt-in. The latter must reject before opening a connection.

### Deployed-config probing

- Deployed/default config values: no new configuration or environment variable. Existing typed `db_settings` and its log-safe target label remain the only connection configuration.
- Explicit value probe: a matching `--expected-target` reaches the async read-only preflight and permits the explicit attestation.
- Absent value probe: no `--expected-target` still emits `target_confirmation_required` before connection, even if the opt-in is supplied.
- Default-session/default-context probe: no opt-in preserves current payload/exit behavior and no new catalog query; `--show-target` remains a connection-free target display.
- Side-effect ordering: target admission happens before connection; the operator preflight opens only the existing read-only connection and transaction; immutable evidence evaluation only reads ledger/catalog state; no migration runner or write API is invoked.

### Files touched

- `atlas_brain/storage/migrations/reconciliation.py`
- `plans/PR-EOM-Migration-Reconciliation-Evidence.md`
- `scripts/check_migration_content_integrity.py`
- `tests/test_migration_content_integrity_preflight.py`

## Mechanism

`atlas_brain.storage.migrations.reconciliation` will hold a frozen dataclass for this one historical incident. It stores the exact ledger digest, packaged digest, observed UTC application instant, earliest retained-source cutoff, and literal source-evidence state. Its attestation helper reads at most two named ledger rows and accepts evidence only when exactly one exists, hashes only the packaged 387 source file, verifies both time predicates, and then asks the existing recurring-invoice readiness helper whether the final catalog is valid. Only if that readiness check succeeds does it execute a `NOT EXISTS` aggregate for active, NULL-period `monthly_auto` / `eom_commercial_billing` invoices.

The helper returns an evidence-oriented JSON-safe result with individual predicate states and `attested` / `not_attested`; it never returns `verified` for the unavailable historical source. The CLI calls it only behind `--attest-known-reconciliations` in the same already read-only transaction as the generic classifier. `_report_payload` continues to determine the status and exit code exclusively from the generic classifier, so a known historical attestation remains an explicitly documented mismatch for Slice 3 to policy-gate later.

The implementation is schema-free because #2465 currently owns migration `388`. A static source artifact can be evaluated before any future migration is applied, avoids a migration-number collision, and introduces no financial-state change.

## Intentional

- The result may say `attested` while the overall run remains `unresolved_drift` / exit 2. Catalog evidence establishes current recovery state, not the unavailable historical source bytes.
- The record is intentionally a singleton, not a generic exception registry. Each future historical mismatch needs its own reviewed evidence record and scope decision rather than a runtime allowlist.
- A schema table, migration rewrite, migration replay, ledger correction, or invoice repair is rejected for this slice. Each would either collide with #2465's migration 388 or expand this evidence-only hardening into an irreversible financial-data decision.
- The CLI output includes only names, fixed evidence predicates, and booleans; it does not output database connection details or invoice/customer rows.

## Deferred

Parking predicate: new historical mismatches, pending-migration admission behavior, dynamic migration self-recording policy, and any data repair are parked unless they are required to make this one 387 evidence attestation truthful, read-only, and backward compatible.

- H-18 Slice 3 in #2461 will define the pending-migration admission/rollout/recovery policy using this evidence. It must decide policy explicitly; this PR does not block startup or migration execution.
- H-18 Slice 4 in #2461 will document/validate dynamic self-recording authoring policy around the existing atomic-bookkeeping marker.
- Any new mismatch needs a separate issue/plan and may not be added as a runtime exception by editing this singleton record without its own provenance proof.
- No new hardening finding is parked by this slice beyond #2461's ordered H-18 follow-ups.

Parked hardening: none.

## Verification

- Executed before the original push: `pytest -q tests/test_migration_content_integrity_preflight.py` — 18 passed.
- Executed before the original push: Ruff check covering `atlas_brain/storage/migrations/reconciliation.py`, `scripts/check_migration_content_integrity.py`, and `tests/test_migration_content_integrity_preflight.py` — passed.
- Executed before the original push: `python -m compileall -q atlas_brain/storage/migrations scripts/check_migration_content_integrity.py` — passed.
- Executed before the original push: `python scripts/sync_pr_plan.py --check plans/PR-EOM-Migration-Reconciliation-Evidence.md`, plan-shape/code/file/diff-size/reviewer-rule audits, `git diff --check`, and safe CLI target-admission probes — passed.
- Executed for the review repair: `pytest -q tests/test_migration_content_integrity_preflight.py` — 19 passed.
- Executed for the review repair: Ruff check covering `atlas_brain/storage/migrations/reconciliation.py`, `scripts/check_migration_content_integrity.py`, and `tests/test_migration_content_integrity_preflight.py`; `python -m compileall -q atlas_brain/storage/migrations scripts/check_migration_content_integrity.py`; and `git diff --check` — passed.
- Executed for the review repair: `python scripts/maturity_sweep.py atlas_brain/storage --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_storage.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/billing/*.py' --sensitive-glob 'atlas_brain/storage/**'` — passed with no baseline update; the two new first-party mocks caused the prior ratchet failure and were removed.
- Executed for the review repair: `python scripts/sync_pr_plan.py --check plans/PR-EOM-Migration-Reconciliation-Evidence.md` — passed after this plan synchronization.
- GitHub CI will run the full migration-runner workflow and repository-required checks; per Juan's direction, do not duplicate broad GitHub suites locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/storage/migrations/reconciliation.py` | 203 |
| `plans/PR-EOM-Migration-Reconciliation-Evidence.md` | 110 |
| `scripts/check_migration_content_integrity.py` | 46 |
| `tests/test_migration_content_integrity_preflight.py` | 344 |
| **Total** | **703** |
