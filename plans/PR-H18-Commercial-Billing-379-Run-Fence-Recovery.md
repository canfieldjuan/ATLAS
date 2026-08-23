# PR-H18-Commercial-Billing-379-Run-Fence-Recovery

## Why this slice exists

H-18 issue [#2476](https://github.com/canfieldjuan/ATLAS/issues/2476) keeps
the pending missed-call provider migration blocked until every named historical
migration discrepancy has a target-specific admission path. The initial,
target-confirmed, read-only receipt identified one more real safety condition:
the canonical target has exactly one NULL-digest synthetic-version `-10`
receipt named `379_commercial_billing_candidate_review_decisions`, while the
retained package begins that commercial-review source at `380_*`.

That initial target had the later table shape but an `invoices` review-fence
function lacking the current source's `commercialBillingRunId` validation and
`billing_run_id` predicates. The current provider writes that identity in
`atlas_brain/services/commercial_billing_approvals.py`, and retained migration
382 defines the same run-scoped database fence. A pure missing-source
attestation would therefore have hidden a real cross-run review-isolation drift.

A subsequent read-only receipt found an existing `391` ledger record at
`2026-08-23T08:31:59.159960Z`. Its SHA-256 exactly matches the migration source
published here, and the target now has the reviewed run-scoped function body.
This coding arc did not deploy or invoke target SQL; repository evidence cannot
identify the external operation that created that receipt. It does prove the
exact bytes and final catalog now present, so the current target is `attested`
for 379 while the original 379 source remains unavailable.

This is a source-publication and forward-recovery-attestation slice. It keeps
the exact legacy classifier and atomic recovery available for compatible
targets, without rewriting historical ledger facts, creating invoices, or
contacting customers.

The approximately 2,000-LOC diff exceeds the soft cap because the exact
recovery SQL, its closed admission selector, immutable target classifier, and
fake plus real PostgreSQL proof form one financial safety boundary. Splitting
them would publish a recovery without proof, an attestation that hides the
run-isolation defect, or an EOM entrypoint unable to select its prerequisite.

### Problem-derived contract

- Root cause: the historical 379 commercial-review source is unavailable, and
  the target's legacy invoice fence is an older function body that does not bind
  candidate/override lookup to `commercialBillingRunId`. The generic migration
  integrity gate correctly refuses later SQL, but cannot safely tell this
  target-specific recoverable state from an arbitrary unknown catalog change.
- Correct fix must touch/change:
  1. Add one immutable, named historical-missing-source forward-recovery record
     for only `379_commercial_billing_candidate_review_decisions`. It must
     classify the exact target ledger identities, final review-decision catalog,
     and exact legacy fence body as `recovery_required`; it must become
     `attested` only after its own recovery receipt and the current run-scoped
     fence body are present.
  2. Add atomic migration
     `391_eom_commercial_billing_run_fence_recovery.sql`. Before replacing the
     function, it must re-check the exact legacy function SHA-256 and required
     table/trigger catalog state. It then restores the existing migration-382
     function/trigger contract, including mandatory `commercialBillingRunId`
     parsing and both `billing_run_id` predicates. It performs no business-row
     DML and relies on the normal runner to record its own source digest.
  3. Extend the existing closed forward-recovery selector so an explicitly
     selected 391 recovery can run before 390 when the only unresolved evidence
     is the known 379 legacy record plus the independently known 386 recovery
     state. Unknown names, an omitted 391 selection, altered catalog evidence,
     or a pre-recorded/non-attested recovery must execute no pending SQL. The
     runner re-reads the report after 391 and stops while 386 remains unresolved;
     a later invocation may then select 390. Ordinary migration 389 cannot run
     until both recoveries attest.
  4. Add focused fake-runner, preflight, EOM-readiness-profile, and disposable
     PostgreSQL coverage. Its `schema_migrations` fixture must retain the
     canonical `applied_at` default because the runner records its own receipt
     without supplying a timestamp. The real database proof must show the legacy
     fence rejects a clean run because another run's override is consulted, while
     the recovered fence accepts that same current-run invoice without changing
     historical review/override rows. It must never use the canonical target.
- Must not change:
  1. Historical 379/380/381/382 source bytes or `schema_migrations` rows;
     no digest backfill, source replay, broad allowlist, invoice/payment/lead
     DML, or customer-facing email behavior.
  2. Existing commercial billing service/API contracts, website/tracker code,
     ordinary migration ordering, and the 386 recovery's own legacy predicate.
  3. The existing generic integrity report: it remains visibly
     `unresolved_drift` while a historical source is absent; the named
     attestation is admission evidence, never source verification.
  4. Production deployment, runtime restart, migration execution, or any
     configuration. This PR's target verification remains catalog-only/read-only;
     the separately observed 391 receipt is not claimed as a PR action.

## Scope (this PR)

Ownership lane: h18-migration-content-integrity
Slice phase: Production hardening

Max files: 11

1. Model only the observed 379 missing-source/run-fence recovery state and
   reserve 391 from the ordinary pending-migration loop.
2. Restore the current run-scoped invoice-fence function through a single
   atomic, revalidated, forward-only migration.
3. Add 391 to the already closed missed-call readiness set so the production
   EOM migration entrypoint can select it deliberately; prove its two-step
   interaction with 386 and rejection of unrecognized drift.
4. Enroll the dedicated disposable PostgreSQL regression in the existing
   migration job and add the new migration to existing EOM path coverage.

### Review Contract

- Acceptance criteria:
  - [ ] The exact old 379 catalog is `recovery_required`, while the current
    target is `attested` only after its own exact 391 digest receipt and the
    run-scoped fence body; neither state claims the unavailable historical
    source was recovered. Settled by
    `tests/test_migration_content_integrity_preflight.py` and the controlled
    target preflight recorded in #2476.
  - [ ] A legacy 379 state with selected 391 runs only 391 under the existing
    advisory lock and atomic-bookkeeping path, records its digest exactly once,
    re-reads evidence, and leaves ordinary pending SQL blocked while 386 is
    still unresolved; settled by `tests/test_migrations_runner.py`.
  - [ ] An unknown discrepancy, changed legacy fence hash, missing required
    decision-table catalog, an omitted 391 `only=` selection, or an already
    recorded but non-attested 391 causes no target SQL/ledger mutation; settled
    by negative fake-runner and preflight cases.
  - [ ] In an isolated PostgreSQL schema, recovery preserves all existing
    decision and override data, changes no row values, and restores run isolation:
    a candidate in run B is not blocked by an override stored only for run A;
    settled by `tests/test_commercial_billing_runs.py`.
  - [ ] A retry after the 391 receipt is a no-op; a legacy provider that omits
    the run identity remains fail-closed at the recovered database boundary;
    settled by the same disposable PostgreSQL regression.
  - [ ] The EOM missed-call readiness entrypoint explicitly selects 391 and the
    relevant workflows run when its migration, selector, or dedicated proof
    changes; settled by `tests/test_eom_render_profile.py` and both workflow
    command/path assertions.
- Reachability proof: `atlas_brain.main_eom._apply_eom_missed_call_recovery_migrations()`
  passes its closed set to `run_migrations(..., only=...)`; that runner applies
  a recovery prelude before ordinary selected files and re-evaluates the same
  integrity report.
- Affected surfaces: Atlas migration selection, immutable migration evidence,
  commercial billing invoice database fence, and EOM migration CI coverage.
- Risk areas: financial run isolation, migration ordering, wrong-target recovery,
  atomicity, interrupted retry, source/digest drift, and mixed-version rollback.
- Reviewer rules triggered: R1, R2, R4, R5, R8, R10, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: `run_migrations()` content-integrity admission before
  any selected pending SQL; `pending_historical_forward_recovery_migration()`
  chooses at most one explicit, named recovery.
- Replaced-path behaviors:
  - healthy/attested target: preserve ordinary selected migration behavior and
    leave both reserved recovery files inert;
  - exact 379 legacy plus 386 recovery-required: select 391 first only when it
    is included in the caller's pending set, then stop before ordinary SQL;
  - recovered 379 plus exact 386 legacy: select 390 on the next invocation;
  - any unknown, incomplete, or wrong catalog: fail before all pending SQL.
- Guard-relevant fields: unresolved mismatch/missing-source names, 379's exact
  ledger versions/NULL digests/timestamps, successor ledger receipts, decision
  relation kind/columns/constraints/indexes/triggers, invoice-fence trigger,
  legacy and recovered `pg_proc.prosrc` SHA-256 values, recovery source digest,
  recovery ledger row, and selected pending migration names.
- Caller x input shape:
  - EOM closed `only=` set including 391 x target-shaped 379/386 legacy state;
  - explicit `only=` set omitting 391 x the same legacy state;
  - full runner x target-shaped 379/386 legacy state;
  - either runner x unknown extra missing/mismatched evidence;
  - rerun after 391 is recorded; and
  - direct current/legacy invoice insert x two runs sharing a candidate identity
    with an override only in the other run.

### Deployed-config probing

- Deployed/default config values: N/A; this adds no configuration or fallback.
- Explicit value probe: N/A.
- Absent value probe: N/A.
- Default-session/default-context probe: the recovery and its attestation use
  the existing connection schema and query only catalog metadata; disposable
  tests run under an isolated schema and prove the trigger binds there.
- Side-effect ordering: read-only integrity evidence selects 391; migration 391
  re-checks its legacy function/catalog in SQL; function replacement and its
  ledger receipt commit atomically; the runner then re-reads evidence before it
  can consider any later recovery or ordinary migration.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `.github/workflows/atlas_migrations_runner_checks.yml`
- `atlas_brain/main_eom.py`
- `atlas_brain/storage/migrations/391_eom_commercial_billing_run_fence_recovery.sql`
- `atlas_brain/storage/migrations/reconciliation.py`
- `plans/PR-H18-Commercial-Billing-379-Run-Fence-Recovery.md`
- `tests/test_commercial_billing_runs.py`
- `tests/test_eom_render_profile.py`
- `tests/test_migration_content_integrity_preflight.py`
- `tests/test_migrations_runner.py`

## Mechanism

The new reconciliation record does not map historical 379 to current 380 or
claim to recover unavailable source bytes. It records the observed target
receipt and final catalog preconditions as a recovery boundary. Its evidence
can have three outcomes: `recovery_required` for the exact old global fence,
`attested` only after 391's own digest receipt plus the current run-scoped
function body, or `not_attested` for every other target.

Migration 391 is `atomic-bookkeeping`. Its first catalog guard compares the
current function body against the reviewed legacy SHA-256, verifies the known
table/trigger state, and rejects a lookalike target before changing anything.
It then replaces only the invoice-fence function with the already-packaged
migration-382 contract. No table rows, financial history, invoices, payments,
or ledger facts are rewritten. The normal runner records the new migration
digest in the same transaction.

The selector remains closed rather than becoming a generic exception system.
It may choose 391 only for the exact known 379 missing-source state, and only
when all other unresolved names are the known 386 forward-recovery state. It
uses the existing single-prelude/re-read sequence: 391 commits, the runner
observes 386 still unresolved and stops; a fresh invocation can choose the
existing 390 route; only then can normal pending migration 389 be considered.

Rollback is operationally forward-only. For a compatible legacy target before
391, retain the previous runtime and make no target change. The canonical target
already has the separate 391 receipt observed above; do not repoint Atlas to a
pre-#2408 provider that omits `commercialBillingRunId`, because the recovered
database fence correctly rejects that old writer. This PR does not execute any
deployment sequence.

## Intentional

- Preserve the original source-availability limit and generic report instead of
  editing historical migration bytes or normalizing NULL ledger digests.
- Treat any catalog/function discrepancy as a stop condition rather than
  replacing an arbitrary function that merely has the same name.
- Require two deliberate migration invocations when both 379 and 386 need
  recovery. The existing runner applies and re-attests one forward recovery per
  invocation, which keeps each recovery independently observable and retry-safe.
- Reuse the current 382 run-scoped fence rather than changing commercial billing
  API or product behavior. This is a database recovery of the existing contract.

## Deferred

- #2363: external, non-cooperating migration-evidence serialization remains a
  separate H-18 hardening item; this recovery re-checks its own SQL precondition
  but does not redesign the global execution model.
- The current target's 391 execution was observed only through its exact ledger
  receipt and catalog result; repository evidence cannot determine its operator
  or change-control record. The receipt/digest is recorded in #2476. The 390
  target execution and post-run proof remain protected follow-up operational
  actions. No tracker or Website consumer work is unblocked by this source-only
  PR alone.

Parked hardening: none.

## Verification

- Before push: targeted selector/preflight tests, the dedicated disposable
  PostgreSQL recovery test when its isolated test URL is available, Python
  syntax, Ruff, plan sync, whitespace, and contract checks. GitHub owns the
  full Unit Gate and remaining required checks; no duplicate local Unit Gate.
- Controlled target proof before review: a read-only
  `scripts/check_migration_content_integrity.py` receipt against the exact
  configured target showed 391's exact source digest and 379 `attested`; it did
  not invoke the runner.
- Controlled target proof after the protected 390 deployment: a fresh
  target-confirmed read-only receipt must show 386 attested before migration
  389 is eligible.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 2 |
| `.github/workflows/atlas_migrations_runner_checks.yml` | 16 |
| `atlas_brain/main_eom.py` | 3 |
| `atlas_brain/storage/migrations/391_eom_commercial_billing_run_fence_recovery.sql` | 343 |
| `atlas_brain/storage/migrations/reconciliation.py` | 507 |
| `plans/PR-H18-Commercial-Billing-379-Run-Fence-Recovery.md` | 280 |
| `tests/test_commercial_billing_runs.py` | 225 |
| `tests/test_eom_render_profile.py` | 1 |
| `tests/test_migration_content_integrity_preflight.py` | 290 |
| `tests/test_migrations_runner.py` | 333 |
| **Total** | **2000** |
