# PR-H18-B2B-Campaign-Partner-Source-Attestation

## Why this slice exists

The production rollout of the already-merged missed-call recovery provider
(#2475) remains correctly blocked by H-18 (#2476): its pending migration 389
cannot run while historical migration provenance is unresolved. The exact
target-confirmed read-only preflight has one NULL-digest ledger record named
`067_b2b_campaign_partner`; its source is absent from the package. A targeted
provenance investigation found no source at its expected path in reachable Git
history, GitHub's path history, or 5,529 unreachable Git blobs. The target does
have a single matching ledger row plus an ordinary `b2b_campaigns` relation,
the nullable `partner_id` UUID, the named validated foreign key to
`affiliate_partners(id)` with `ON DELETE SET NULL`, and the named ready partial
index. The generic migration-content report has no individual receipt for this
record, so it keeps all pending migration SQL blocked even when those current
facts are independently true.

This is a narrow pending-admission evidence gap. It is not a recovery of the
unavailable source bytes, a ledger rewrite, or a reason to weaken the generic
integrity report. Migration-386 and the three remaining missing-source records
remain independently blocking after this slice.

### Problem-derived contract

- Root cause: the migration runner can subtract only individually modeled,
  successfully attested historical source discrepancies from its pending-SQL
  refusal. It has no source-controlled, target-specific receipt for the exact
  legacy NULL-digest `067_b2b_campaign_partner` row, despite a bounded catalog
  predicate proving the final safe schema state on the target.
- Correct fix must touch/change: `reconciliation.py` must add only migration
  067's immutable ledger receipt and a PII-free, one-statement PostgreSQL
  catalog predicate. The existing generic missing-source admission seam must
  recognize that one receipt after it attests. Focused preflight and real runner
  tests must prove positive and failure-closed behavior, including malformed
  ledger evidence, relation identity, column, foreign-key, and index failures.
  A disposable PostgreSQL test must exercise the real catalog predicate and
  pending-SQL choke point without using operational data.
- Must not change: migration SQL, `schema_migrations`, production tables or
  rows, generic integrity-report semantics, migration 386, any other
  missing-source receipt, migration 389, EOM lead/email/payment behavior,
  deploy-time configuration, and the active tracker/Website CRM lanes.

## Scope (this PR)

Ownership lane: eom/migration-content-integrity
Slice phase: Production hardening
Max files: 7

1. Add an immutable, source-unavailable receipt and a read-only catalog
   attestation for only `067_b2b_campaign_partner`.
2. Let the existing shared runner admission seam remove only that name from
   `missing_source` after its receipt succeeds; unknown, malformed, or
   coexisting evidence remains a refusal before pending SQL.
3. Add fake preflight/runner and disposable-PostgreSQL proof, and archive only
   this session's merged #2479 plan while refreshing the generated plan index.

### Review Contract

- Acceptance criteria:
   1. The new source-controlled record names only
      `067_b2b_campaign_partner`, version 67, a NULL digest, and the exact
      target-confirmed aware UTC timestamp; settled by
      `tests/test_migration_content_integrity_preflight.py` assertions against
      the checked-in record.
   2. The catalog predicate reads no business rows and returns `attested` only
      when `b2b_campaigns` is an ordinary non-partition table with the expected
      nullable UUID column, named validated `affiliate_partners(id)` foreign key
      with `ON DELETE SET NULL`, and named valid/ready partial partner index;
      settled by its structured evidence payload, single-`fetchrow` fake
      assertion, and zero-`execute_calls` preflight assertion.
   3. A missing, duplicate, wrong-version, non-NULL-digest, or wrong-timestamp
      ledger row; a foreign/partitioned relation; wrong column signature;
      malformed foreign key; or missing/wrong/unready index remains
      `not_attested`; settled by parameterized fake regressions and real
      PostgreSQL fixtures.
   4. `run_migrations` applies one pending test migration exactly once when 067
      is the sole unresolved evidence and the receipt succeeds; settled by
      `tests/test_migrations_runner.py` observing the real runner's applied SQL
      and digest-aware ledger insert.
   5. Any other missing source or failed 067 receipt leaves all pending SQL and
      ledger writes absent; settled by focused runner regressions and a
      disposable-PostgreSQL refusal test.
   6. The generic report remains `unresolved_drift` for 067 and never mutates
      the ledger; settled by the structured preflight payload and read-only
      transaction fixture.
   7. The existing 382, 022b, and 387 admissions remain intact; settled by the
      existing focused preflight/runner regressions in the same selection.
   8. The only archived plan is
      `PR-H18-Presence-Unknown-Count-Source-Attestation.md`, and
      `plans/INDEX.md` is regenerated by `scripts/archive_plans.py index`.
- Reachability proof: `run_migration_content_integrity_preflight` is the
  read-only operator entrypoint used by
  `scripts/check_migration_content_integrity.py`; `run_migrations` is the
  shared pending-SQL admission choke point. Tests observe their structured
  evidence, applied SQL, and ledger behavior.
- Affected surfaces: source-controlled reconciliation records and catalog
  probes, the existing migration-runner admission boundary, preflight evidence
  payload, focused fake and disposable-PostgreSQL tests, and plan archival
  metadata.
- Risk areas: accidental generic allowlist, a wrong/partial historical schema
  receipt, catalog TOCTOU, source-verification overclaim, PII exposure,
  pre-admission SQL, migration-386 masking, and current EOM rollout delay.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

The shared `_unresolved_pending_migration_content_evidence` admission seam is
extended through its existing source-controlled missing-source record set; the
generic report remains unchanged.

- Boundary path/seam: the known missing-source receipt registry and the
  report-derived candidate names passed into
  `attest_known_historical_migration_reconciliations`.
- Replaced-path behaviors: only a successful 067 receipt can move that one
  `missing_source` name from runner refusal to pending-SQL admission. Unknown
  names, stale/malformed ledger evidence, failing catalog evidence, failures
  reading the target, and any coexisting mismatch/missing source remain a
  refusal before SQL.
- Guard-relevant fields: migration name, expected version, ledger cardinality,
  NULL digest, exact aware UTC `applied_at`, relation kind/partition identity,
  partner-column signature/default, foreign-key key/referenced columns/actions,
  index uniqueness/validity/readiness/key/predicate.
- Caller x input shape: the CLI preflight and `run_migrations` pass only names
  reported by their current generic report. The receipt returns source-unavailable
  evidence while the runner subtracts only a matching `status == "attested"` name.

### Closure Declaration

- Set: migration names eligible to clear `missing_source` from pending admission
  after a catalog attestation.
- Membership: **CLOSED**. This slice adds one source-controlled receipt,
  `067_b2b_campaign_partner`; unlisted names have no reviewed receipt.
- Source: **DERIVED** by the existing
  `known_historical_missing_source_reconciliation_names()` registry; there is
  no duplicate runner allowlist.
- Outside-set behavior: **block**. An unlisted, malformed, unavailable, or
  coexisting historical discrepancy leaves pending SQL unapplied.

### Deployed-config probing

- Deployed/default config values: no configuration changes. The existing typed
  `ATLAS_DB_*` configuration chooses the target only for the read-only
  integrity preflight.
- Explicit value probe: the configured operational target was queried under an
  explicit read-only transaction. It contains one 067 NULL-digest row at the
  recorded UTC timestamp and the required schema metadata.
- Absent value probe: no booking, email, lead, or migration setting can make
  this receipt pass. An unavailable target or probe failure raises a
  pre-admission refusal.
- Default-session/default-context probe: focused tests prove an explicit
  read-only transaction and zero `execute` calls; the catalog evidence is read
  in one PostgreSQL statement snapshot.
- Side-effect ordering: generic report and named target attestation occur
  before pending migration discovery can execute SQL; a failed receipt leaves
  all pending SQL and ledger writes absent.

### Files touched

- `atlas_brain/storage/migrations/reconciliation.py`
- `plans/INDEX.md`
- `plans/PR-H18-B2B-Campaign-Partner-Source-Attestation.md`
- `plans/archive/PR-H18-Presence-Unknown-Count-Source-Attestation.md`
- `tests/test_b2b_campaign_partner_migration_repair.py`
- `tests/test_migration_content_integrity_preflight.py`
- `tests/test_migrations_runner.py`

## Mechanism

The receipt is a `HistoricalVersionedMissingSourceReconciliation`: its
`source_verification` is permanently `historical_source_unavailable`, because
neither the target catalog nor the receipt may invent source bytes. Its catalog
probe is a single read-only PostgreSQL statement that returns only booleans and
catalog metadata for the named relation, column, foreign key, and index. It
never queries `b2b_campaigns` data or any lead/customer/email table.

The existing runner already computes a closed candidate set from
`known_historical_missing_source_reconciliation_names()` and subtracts only
successful attestation names from pending admission. Adding the record and its
dispatcher branch makes this receipt additive without another runner exception
or an alternate ledger. The generic diagnostic report keeps the 067 row in
`missing_source`; only the runner's temporary admission set changes after the
catalog proof succeeds.

## Intentional

- The receipt does not claim to reconstruct, checksum, replace, or replay the
  unavailable source bytes; the proof is current-target structural evidence,
  not source verification.
- This does not turn the generic preflight green. It will continue to display
  067 as missing source and return unresolved drift while migration-386 and the
  other named gaps exist.
- This is not a prefix mapping, a generic NULL-digest exception, or a dynamic
  catalog allowlist. Every later source absence needs its own reviewed record
  and predicate.
- The one-statement snapshot prevents a catalog change between relation,
  column, constraint, and index reads from producing a hybrid success result.

## Deferred

- #2476 continues to own migration-386's real mismatch and the remaining
  missing-source records (`272_b2b_watchlist_alert_events`,
  `297_b2b_company_signal_canonical_promotion_type`, and
  `379_commercial_billing_candidate_review_decisions`).
- #2475's missed-call provider deployment and its tracker/Website consumer
  slices remain blocked until every independent H-18 admission item is proven.

Parking predicate: any additional historic migration, schema repair,
`schema_migrations` mutation, generic-report behavior, product workflow,
deployment configuration, or cross-repository consumer change is parked unless
strictly required for this one 067 receipt and the shared admission seam.

Parked hardening: #2476 is the active H-18 tracking issue; no additional
hardening item is hidden in this slice.

## Verification

- Passed: `pytest -q tests/test_migration_content_integrity_preflight.py
  tests/test_migrations_runner.py tests/test_b2b_campaign_partner_migration_repair.py`
  (`134 passed, 5 skipped` without a disposable database URL).
- Passed: a fresh disposable PostgreSQL 16 container executed the real catalog
  predicate, ordinary-relation/foreign-key/index checks, leaf-partition and
  wrong-delete-action refusals, and pending-SQL admission (`4 passed`). No
  operational database or campaign data was used.
- Passed: `python -m py_compile` and `ruff check` on every changed Python file.
- Passed: `git diff --check`.
- Passed: target-confirmed read-only candidate preflight returns 067
  `attested` with `historical_source_unavailable`; it did not run pending SQL
  or write the ledger.
- Pending before push: plan/body audits and the mandatory local wrapper. GitHub
  owns broad unit and integration gates.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/storage/migrations/reconciliation.py` | 368 |
| `plans/INDEX.md` | 3 |
| `plans/PR-H18-B2B-Campaign-Partner-Source-Attestation.md` | 253 |
| `plans/archive/PR-H18-Presence-Unknown-Count-Source-Attestation.md` | 0 |
| `tests/test_b2b_campaign_partner_migration_repair.py` | 301 |
| `tests/test_migration_content_integrity_preflight.py` | 258 |
| `tests/test_migrations_runner.py` | 113 |
| **Total** | **1296** |

## Diff-budget rationale

This exceeds the usual 400-LOC target because a named historical admission
receipt is indivisible: the target predicate, its one-name closed admission
connection, its positive and failure-closed fake tests, real PostgreSQL proof,
and plan metadata form one safety claim. Splitting them would either make a
ledger exception unreviewed or leave the runner's no-SQL-on-failure behavior
unproven.
