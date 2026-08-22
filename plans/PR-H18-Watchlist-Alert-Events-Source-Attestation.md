# PR-H18-Watchlist-Alert-Events-Source-Attestation

## Why this slice exists

H-18 issue #2476 blocks the migration-389 provider rollout because the target
has a separately named, NULL-digest ledger receipt
`272_b2b_watchlist_alert_events` without packaged source bytes. The target
also has a later, distinct `273_b2b_watchlist_alert_events` receipt, so this
is not a rename. Retained history and unreachable-object inspection found no
272 source blob. The only safe treatment is one named, catalog-only receipt.

Diff-budget override: the exact named predicate, closed signature, preflight
matrix, runner no-SQL/retry proof, and disposable-PostgreSQL reachability proof
form one migration-admission claim. Splitting them would publish an unreviewed
historical exception or leave pending-SQL blocking behavior unproven.

### Problem-derived contract

- Root cause: a historical numeric-prefix collision left the canonical target
  with a separately named synthetic-version 272 receipt whose source bytes and
  SHA-256 are unavailable. The generic content-integrity gate correctly blocks
  migration 389 rather than guessing its provenance.
- Correct fix must touch/change: the source-controlled reconciliation registry,
  a read-only catalog predicate for the exact 272 receipt, and preflight/
  runner/real-PostgreSQL proof of named admission and failure closure.
- Must not change: migration SQL, generic classifier, ledger rows, target data,
  configuration, deployment, B2B alert rows/API behavior, EOM workflows, or
  the independently unresolved H-18 records 386, 297, and 379.

### Contract revision (review round 1)

- Evidence: five confirmed P1 threads on #2481 show that the initial receipt
  did not prove durable table storage, could collapse distinct SQL literals,
  left catalog predicates without isolated negative coverage, hand-copied a
  producer fixture, and never enrolled the real PostgreSQL suite in migration
  CI.
- Root cause: the first implementation treated an ordinary relation kind as
  sufficient and reused a lossy general expression normalizer, while test
  coverage proved representative failures rather than every field in the
  closed admission signature.
- Required surface: the named 272 catalog receipt must require
  `relkind = 'r'`, `relpersistence = 'p'`, and non-partition status; its
  272-only expression comparisons must preserve quoted SQL literals; every
  guard must have an isolated fake-catalog failure; and the disposable suite
  must execute retained `273_b2b_watchlist_alert_events.sql` as the actual
  producer of the expected catalog shape and run in the existing migration CI
  job.
- Non-scope: retained 273 remains neither source evidence nor a rename for
  named 272; no migration SQL, production ledger/data, generic canonicalizer,
  or broader H-18 record changes are allowed.
- Verification: focused preflight and runner tests, the disposable PostgreSQL
  suite when its dedicated test database is configured, syntax/lint/plan
  gates locally, and the existing GitHub migration job for full service-backed
  execution. The broad unit gate remains GitHub-only.

### Contract revision (review round 2)

- Evidence: confirmed P1 thread `PRRT_kwDOQ5Uhrs6baKLh` shows the first
  272-only helper lowercased the whole catalog expression before its
  quote-preserving regex ran. Consequently, literal case could still collapse:
  `'OPEN'` was indistinguishable from expected `'open'`.
- Root cause: expression normalization occurred before literal tokenization,
  so preserving quote delimiters could not restore the original literal text.
- Required surface: segment SQL string literals first; normalize case,
  removable casts, and whitespace only in unquoted fragments; preserve every
  quoted literal byte sequence and its case. Both the isolated fake-catalog
  matrix and the retained-273 disposable PostgreSQL fixture must reject an
  upper-case status literal.
- Non-scope: no generic canonicalizer redesign, parser dependency, migration
  SQL, target data, ledger update, or change to retained-273 provenance.
- Verification: focused 272 preflight proof plus GitHub's configured real
  PostgreSQL migration job. The existing escaped-literal collision remains a
  distinct assertion alongside the case-collision proof.

## Scope (this PR)

Ownership lane: `eom/migration-content-integrity`
Slice phase: Production hardening
Max files: 8

- Add one immutable 272 record requiring version `-3`, NULL digest, exact UTC
  recorded time, and the original base-table catalog contract.
- Make the catalog proof metadata-only: permanent ordinary non-partition
  table, 20 source-era columns, named PK/FKs/checks with case- and
  content-preserving literal-safe expressions, and three ready indexes.
- Admit only this reported ledger name after a complete attestation; every
  unlisted or incomplete source gap remains blocking.
- Exercise preflight, runner failure/retry, and a disposable PostgreSQL schema.

### Files touched

- `.github/workflows/atlas_migrations_runner_checks.yml`
- `atlas_brain/storage/migrations/reconciliation.py`
- `plans/INDEX.md`
- `plans/PR-H18-Watchlist-Alert-Events-Source-Attestation.md`
- `plans/archive/PR-H18-B2B-Campaign-Partner-Source-Attestation.md`
- `tests/test_b2b_watchlist_alert_events_migration_repair.py`
- `tests/test_migration_content_integrity_preflight.py`
- `tests/test_migrations_runner.py`

### Review Contract

- Acceptance criteria:
  - [ ] The existing preflight returns an `attested` 272 payload only for one
    `version=-3`, NULL-digest receipt at the immutable timestamp plus complete
    source-era catalog metadata; settled by the focused preflight matrix.
  - [ ] Wrong/missing/duplicate ledger evidence, non-permanent/non-table/
    partition relation metadata, altered source-era columns/constraints/indexes
    (including content- or case-distinct quoted literals), or an unready index returns
    `not_attested`; every closed-signature guard has an isolated fake-catalog
    failure, and the runner leaves pending SQL and its ledger row absent;
    settled by preflight/runner tests and disposable PostgreSQL cases.
  - [ ] A successful 272 predicate clears only its reported ledger name; an
    arbitrary missing source remains in `missing_source`; settled by the runner
    non-generic-admission proof.
  - [ ] The real `run_migrations` entrypoint, against a disposable schema built
    by executing retained `273_b2b_watchlist_alert_events.sql` solely as an
    actual producer fixture (not 272 source evidence), creates exactly one
    probe table/hashed receipt after attestation and no duplicate receipt on
    retry; settled by the PostgreSQL test enrolled in migration CI.
  - [ ] Current target read-only preflight records 272 as attested but remains
    exit 2 because other H-18 discrepancies exist; settled by the target probe.
- Reachability proof: `run_migrations` executes the pending probe from a
  disposable schema; observable state is no probe/ledger row on failure or one
  probe/hashed ledger row on successful retry.
- Affected surfaces: migration admission, read-only provenance preflight, and
  PostgreSQL catalog metadata. No public API or configuration surface changes.
- Risk areas: provenance, migration safety, false admission, privacy, and
  backward compatibility.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R8, R10, R14.

### Closure declaration

- Reconciliation registry: CLOSED. The dispatcher admits one named immutable
  record per discrepancy; unknown source names have no path.
- 272 catalog signature: CLOSED. Required columns, constraints, and indexes
  are enumerated; any missing or altered member fails closed.

### Boundary-change enumeration

- Boundary: report-derived missing-source name -> closed registry -> named
  catalog predicate -> runner admission.
- Replaced behavior: only a complete 272 receipt stops that exact record from
  blocking pending migration SQL. All other source gaps retain current blocking
  behavior.
- Guard fields: name, synthetic version, NULL digest, timestamp, permanent
  relation identity, source-era columns/defaults, named constraints and
  case- and content-preserving expressions, indexes/readiness.

## Mechanism

`HistoricalVersionedMissingSourceReconciliation` records only the named 272
identity. A single read-only PostgreSQL catalog query returns structural
metadata for the source-era table; Python tokenizes quoted literals before
normalizing unquoted SQL and compares a closed, case- and content-preserving
signature, emitting booleans only. The disposable test
executes retained 273 DDL only to produce the expected table catalog contract;
it never asserts that 273 verifies or replaces 272 source bytes. The existing
dispatcher intersects report-derived missing-source names with its closed
registry, so an unlisted name cannot use this receipt. No query selects
alert-event rows or writes `schema_migrations`; rollback is ordinary code
rollback, which returns 272 to generic blocking behavior.

## Intentional

- No generic allowlist or alias mechanism.
- No source reconstruction, hash backfill, or 273-to-272 rename inference.
- No source-era requirement for later `reopen_count`.

## Deferred

H-18 records 386, 297, and 379 remain independent blockers in #2476. The
historical source-number collision root cause remains forensic-only; any broad
renumbering needs its own issue. The target is still blocked from migration 389
until those records independently have reviewed evidence.

Parking predicate: unrelated runner redesign and all other H-18 records remain
parked. Parked hardening: none.

## Verification

- `python -m pytest -q tests/test_migration_content_integrity_preflight.py -k '272 or known_historical'` — 47 passed, 65 deselected.
- `python -m pytest -q tests/test_migrations_runner.py -k '272 or missing_source or historical_attestation'` — 12 passed, 61 deselected.
- `python -m pytest -q tests/test_b2b_watchlist_alert_events_migration_repair.py` — 7 skipped; no disposable test database was configured locally. The same suite is now enrolled in the existing GitHub PostgreSQL migration job.
- `python -m py_compile ...` and `ruff check ...` — passed for all four changed
  Python files.
- `python scripts/check_guard_class_closure.py --base origin/main --strict` —
  passed; the declared closed registry and signature have property proof.
- `python scripts/check_migration_content_integrity.py --expected-target 'host=localhost, port=5433, db=atlas' --attest-known-reconciliations` —
  read-only exit 2; named 272 is attested with permanent storage and literal-safe
  catalog evidence, while independent H-18 records remain blocking. Broad unit
  suite and required workflows are GitHub CI only per operator direction.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_migrations_runner_checks.yml` | 3 |
| `atlas_brain/storage/migrations/reconciliation.py` | 641 |
| `plans/INDEX.md` | 3 |
| `plans/PR-H18-Watchlist-Alert-Events-Source-Attestation.md` | 206 |
| `plans/archive/PR-H18-B2B-Campaign-Partner-Source-Attestation.md` | 0 |
| `tests/test_b2b_watchlist_alert_events_migration_repair.py` | 366 |
| `tests/test_migration_content_integrity_preflight.py` | 488 |
| `tests/test_migrations_runner.py` | 191 |
| **Total** | **1898** |
