# PR-H18-EOM-Public-Onboarding-Source-Attestation

## Why this slice exists

The controlled deployment checkpoint for the already-merged missed-call
recovery provider (#2475) correctly refused pending migration 389. The
target-confirmed read-only H-18 preflight for #2476 shows that
`382_eom_public_onboarding_tokens` is a legacy NULL-digest ledger record whose
source bytes are unavailable because of a historic prefix collision. Unlike the
other unresolved records, the same target has a single exact ledger timestamp
and the complete, immutable public-onboarding token schema installed by the
historical 383/384 recovery path. The current runner treats every
`missing_source` record as equally inadmissible, so it cannot distinguish this
one individually proven record from unknown or incomplete provenance.

This is an admission-evidence gap, not a reason to rewrite migration history or
to weaken the generic integrity report. The migration-386 live-schema mismatch
and five other missing-source records remain independently blocking after this
slice.

### Problem-derived contract

- Root cause: the runner can remove only an attested digest mismatch from its
  pending-admission refusal. It has no separately modeled, exact, read-only
  reconciliation for a known legacy NULL-digest missing source, even when the
  target ledger and current catalog prove that record's final safe state.
- Correct fix must touch/change: source-controlled evidence must name only
  `382_eom_public_onboarding_tokens`, retain the observed UTC timestamp and
  NULL-digest requirement, and run a PII-free catalog predicate for the final
  immutable token projection, constraints, and indexes. The runner must remove
  only that named `missing_source` record from *pending admission* after a
  successful attestation, while the generic preflight continues to report it as
  unresolved drift. Focused tests must prove the exact positive path, every
  relevant negative evidence field, no-write preflight behavior, failure-closed
  runner behavior, and preservation of 387's mismatch path. This PR also folds
  the required archival of its own already-merged #2477 plan into this next
  branch and regenerates the plan index.
- Must not change: migration SQL, `schema_migrations`, production rows,
  migration-386, any other missing-source record, the global diagnostic report,
  migration 389, EOM lead/email/payment behavior, deployment configuration,
  and the active tracker/Website CRM lanes.

## Scope (this PR)

Ownership lane: eom/migration-content-integrity
Slice phase: Production hardening

1. Add an immutable, target-proven reconciliation record and a PII-free
   catalog attestation for only the legacy public-onboarding migration 382
   source absence.
2. Permit `run_migrations` to admit pending SQL only when that exact missing
   source is attested and no other mismatched or missing-source evidence
   remains; retain the generic report and fail-closed behavior for all other
   records.
3. Add focused preflight, runner, and disposable-PostgreSQL proof, then archive
   only the merged #2477 plan and refresh `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  1. `MIGRATION_382_EOM_PUBLIC_ONBOARDING_TOKENS_RECONCILIATION` is the sole
     missing-source record eligible for this path and contains the exact
     target-confirmed UTC `applied_at` value; settled by
     `tests/test_migration_content_integrity_preflight.py` assertions against
     the checked-in record.
  2. A ledger row with a NULL digest, exact timestamp, and final catalog
     predicate produces `attested` evidence without pretending the unavailable
     source is verified; settled by the read-only preflight payload and its
     zero-`execute_calls` assertion.
  3. A non-NULL digest, timestamp mismatch, duplicate/missing ledger row, or a
     missing immutable field, fingerprint check, terminal-state check, issued
     contact uniqueness, or status index remains `not_attested`; settled by
     parameterized focused regressions.
  4. `run_migrations` applies a pending test migration exactly once when the
     target 382 record is the sole unresolved evidence and the attestation
     succeeds; settled by `tests/test_migrations_runner.py` observing the real
     runner's applied SQL and digest-aware ledger insert.
  5. Any other missing source or failed attestation leaves all pending SQL
     unapplied and raises `PendingMigrationContentIntegrityError`; settled by
     focused runner regressions that assert empty applied/inserted state before
     retry.
  6. The generic read-only report remains `unresolved_drift` for 382 and does
     not mutate ledger state; its optional evidence mode probes only names
     reported on that target, settled by its structured preflight payload,
     candidate-name assertion, and read-only transaction fixture.
  7. The pre-existing migration-387 exact-mismatch admission remains intact;
     settled by the existing 387 runner/preflight regressions in the same
     focused selection.
  8. The only retired plan is
     `PR-H18-Migration-387-Attestation-Precision.md`, and the generated index
     is synchronized by `scripts/archive_plans.py index`.
- Reachability proof: `run_migration_content_integrity_preflight` remains the
  read-only entrypoint behind `scripts/check_migration_content_integrity.py`;
  `run_migrations` is the shared pending-SQL admission choke point. Tests
  observe the former's structured evidence payload and the latter's applied
  SQL/ledger write behavior.
- Affected surfaces: source-controlled reconciliation records and probes,
  migration-runner admission, preflight evidence payload, focused fake and
  disposable-PostgreSQL tests, and required plan archival metadata.
- Risk areas: expanding a named exception into a generic allowlist, accepting a
  partially repaired token catalog, exposing token/customer data through
  evidence, allowing SQL before a failed probe, and accidentally clearing 386
  or another missing source.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam:
  `_unresolved_pending_migration_content_evidence` gains a second, explicitly
  named reconciliation category for a missing packaged source. The generic
  `migration_content_integrity_report` is intentionally unchanged.
- Replaced-path behaviors: only an attested 382 source absence can move from
  runner refusal to pending-SQL admission. An unknown name, a different known
  record, malformed/duplicate ledger evidence, an incomplete catalog, a probe
  transport failure, or any coexisting mismatch/missing source remains a
  refusal before SQL.
- Guard-relevant fields: record name, ledger cardinality, NULL digest, exact
  aware UTC `applied_at`, the nine immutable column signatures, signing-key
  fingerprint check, terminal-state check, issued-contact unique partial index,
  and status index.
- Caller x input shape: both the CLI preflight and `run_migrations` load the
  same source-controlled records. The CLI emits evidence only for names the
  current report identifies while keeping that generic report unresolved; the
  runner removes only names whose matching record has `status == "attested"`.

### Closure Declaration

- Set: migration names eligible to clear `missing_source` from *pending
  admission* after a catalog attestation.
- Membership: **CLOSED**. It is one source-controlled evidence record,
  `382_eom_public_onboarding_tokens`; an unlisted migration has no reviewed
  target receipt and is therefore out of scope for this proof.
- Source: **DERIVED** at each runner use from the immutable reconciliation
  record module's `known_historical_missing_source_reconciliation_names()`;
  there is no second runner allowlist.
- Outside-set behavior: **block**. Any unlisted, malformed, unavailable, or
  coexisting missing-source evidence remains unresolved and stops pending SQL;
  false admission could apply a migration to an unproven schema, while refusal
  is safely recoverable through a future target-specific receipt.

### Deployed-config probing

- Deployed/default config values: no configuration is added or changed. The
  existing typed `ATLAS_DB_*` database settings are used only by the read-only
  integrity preflight.
- Explicit value probe: the current target was independently read in a
  read-only transaction; its named 382 row has a NULL digest, the exact recorded
  timestamp, and the complete final catalog predicate.
- Absent value probe: no booking, email, lead, or migration setting can make an
  attestation pass; unavailable database/probe transport raises an admission
  refusal before SQL.
- Default-session/default-context probe: focused preflight tests assert an
  explicit read-only transaction and zero `execute` calls.
- Side-effect ordering: report and target attestation occur before the pending
  migration set is applied; a non-attested or unknown record leaves all pending
  SQL and ledger writes absent.

### Files touched

- `atlas_brain/storage/migrations/__init__.py`
- `atlas_brain/storage/migrations/reconciliation.py`
- `plans/INDEX.md`
- `plans/PR-H18-EOM-Public-Onboarding-Source-Attestation.md`
- `plans/archive/PR-H18-Migration-387-Attestation-Precision.md`
- `scripts/check_migration_content_integrity.py`
- `tests/test_eom_public_onboarding_migration_repair.py`
- `tests/test_migration_content_integrity_preflight.py`
- `tests/test_migrations_runner.py`

## Mechanism

The evidence module will model the preexisting 387 digest mismatch and the
new, structurally different 382 missing-source receipt separately. The 382
probe reads only `schema_migrations` metadata and PostgreSQL catalog metadata;
it returns booleans rather than rows from the token table or any customer
projection. Its `source_verification` remains
`historical_source_unavailable` even when the record is `attested`.

The migration runner will compute independently closed candidate sets for
known mismatches and known missing sources, request the shared attestation
collection only when needed, and subtract only successful attested names from
their original category. The optional read-only CLI evidence mode likewise
passes only names currently reported by that target, avoiding irrelevant
catalog probes on a healthy database. Existing unresolved evidence stays in the
error payload. No migration is replayed, repaired, or marked in this slice. A
real disposable-PostgreSQL test will invoke the catalog predicate against a
complete schema and a deliberately incomplete schema without using production
data.

## Intentional

- This does not turn the generic preflight green or claim source verification:
  382 remains visible as `missing_source` / `unresolved_drift` in the
  diagnostic report.
- This is not a mapping-by-prefix, an arbitrary historic exception list, or an
  allowlist of all NULL-digest rows. Every later missing source must provide its
  own immutable receipt and catalog predicate.
- The existing no-argument reconciliation helper retains its historical
  387-only result. New 382 callers must pass the report-derived candidate name,
  so the change is additive for existing internal consumers.
- The catalog predicate checks final immutable schema evidence and never reads
  token, lead, customer, or email data.
- Migration 389 remains blocked after this PR because 386 and five other
  missing-source records are deliberately unchanged.

## Deferred

- #2476 owns a forward migration and execution-boundary design for the real
  migration-386 catalog mismatch; it must not be misrepresented as an
  attestable historic-source discrepancy.
- #2476 also owns target-specific receipts for
  `022b_presence_unknown_count`, `067_b2b_campaign_partner`,
  `272_b2b_watchlist_alert_events`, `297_b2b_company_signal_canonical_promotion_type`,
  and `379_commercial_billing_candidate_review_decisions`.
- The missed-call provider deployment (#2475), tracker relay, and Website CRM
  action remain safely blocked until every independent H-18 admission item is
  resolved.

Parking predicate: any additional historic migration name, recovery SQL,
`schema_migrations` mutation, lead/email/product behavior, deployment
configuration, or consumer integration is parked unless required to prove this
one target's catalog predicate and closed runner-admission seam.

Parked hardening: #2476 is the active H-18 tracking issue; no separate
hardening item is hidden in this PR.

## Verification

- Passed: `python -m py_compile` and `ruff check` for the six changed Python
  files.
- Passed: focused unit selection:
  `pytest -q tests/test_migration_content_integrity_preflight.py -k
  'migration_382 or known_382 or known_387 or default_known_reconciliation'`
  (`19 passed, 12 deselected`) and
  the scoped runner selection (`10 passed, 52 deselected`).
- Passed: a disposable, local PostgreSQL 16 instance executed the real complete
  382 catalog attestation and the observed-incomplete-catalog refusal through
  the production migration runner (`2 passed, 3 deselected`). No production
  database, token, lead, customer, or email data was used.
- Passed: `git diff --check`.
- Passed: target-confirmed read-only preflight from this candidate worktree.
  It reports 382 `attested` while preserving `unresolved_drift` / exit `2` for
  migration 386 and the five other missing sources. The command's read-only
  transaction performed no migration or ledger write.
- `ruff format --check` is not a gate for this slice: it proposes formatting
  the same pre-existing layout on every touched Python file at `origin/main`.
  The diff does not run a formatting sweep; `ruff check` passes.
- Pending before push: plan synchronization and repository PR/body audits.
  GitHub owns broad unit and integration/disposable-PostgreSQL workflows.
- Deployment proof is intentionally deferred: this PR cannot deploy 389 while
  the independent #2476 evidence remains unresolved.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/storage/migrations/__init__.py` | 31 |
| `atlas_brain/storage/migrations/reconciliation.py` | 389 |
| `plans/INDEX.md` | 3 |
| `plans/PR-H18-EOM-Public-Onboarding-Source-Attestation.md` | 276 |
| `plans/archive/PR-H18-Migration-387-Attestation-Precision.md` | 0 |
| `scripts/check_migration_content_integrity.py` | 7 |
| `tests/test_eom_public_onboarding_migration_repair.py` | 45 |
| `tests/test_migration_content_integrity_preflight.py` | 287 |
| `tests/test_migrations_runner.py` | 171 |
| **Total** | **1209** |

## Diff-budget rationale

The code-and-test change exceeds the usual 400-LOC target because this is one
indivisible admission boundary: provider deployment cannot safely proceed
without both the exact catalog predicate and proof that its one-name exception
neither hides nor admits any other historical source gap. The closed catalog
receipt, fake preflight/runner model, real PostgreSQL proof, plan contract, and
mandatory prior-plan archive are one safety claim; splitting them would make
either the exception or its failure-closed behavior unreviewed.
