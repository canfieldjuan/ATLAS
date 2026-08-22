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

### Contract revision (review round 3)

- Evidence: confirmed P1 thread `PRRT_kwDOQ5Uhrs6baLy2` shows that the first
  catalog query looked up only enumerated names. An additional `CHECK` could
  reject new rows, and a standalone unique/exclusion index could restrict
  writes, without changing the sampled objects.
- Root cause: the receipt treated a list of required objects as sufficient
  evidence without also checking the relevant out-of-set catalog members.
- Required surface: require that the target has no unlisted table constraints
  and no unlisted unique/exclusion index unless it is one of the named source
  indexes or backs a named expected constraint. Keep the post-source
  `reopen_count` column and benign later non-unique indexes outside this
  source-era predicate. Add isolated fake-catalog and retained-273 disposable
  PostgreSQL rejection fixtures for both new guards.
- Non-scope: no broad table-shape freeze, generic catalog predicate change,
  migration SQL, target data, or rejection of later non-unique performance
  indexes.
- Verification: focused 272 preflight/runner tests locally and the configured
  disposable PostgreSQL migration job on GitHub.

### Contract revision (review round 4)

- Evidence: confirmed P1 thread `PRRT_kwDOQ5Uhrs6baSBW` shows the named 272
  catalog query samples only 20 source-era columns. A later unlisted `NOT NULL`
  column with no default can make the actual alert writer's explicit INSERT fail
  while every existing receipt boolean remains true.
- Root cause: the receipt's absence policy closed constraints and
  unique/exclusion indexes but left the writer's omitted-column input set open.
  The source search shows exactly one compatible later column,
  `reopen_count`, from retained migration 281; no arbitrary later column is
  proven safe.
- Required surface: require the 20 source-era columns plus exact signature for
  the known later `reopen_count` column, and reject every other live
  user-defined column through catalog metadata before admitting pending SQL.
  Add isolated fake-catalog and disposable PostgreSQL cases for an unlisted
  required/no-default column, while the retained fixture executes the known
  281 DDL as current writer compatibility evidence.
- Non-scope: no alert-writer change, migration SQL change, generic catalog
  predicate, ledger/data mutation, schema migration, or broad table-shape
  policy outside this named historical receipt.
- Verification: focused 272 preflight and runner tests, the configured
  disposable PostgreSQL migration job on GitHub, and the existing read-only
  target preflight. The target must remain attested only when its column set is
  exactly the documented source-era plus 281 set.

### Contract revision (review round 5)

- Evidence: confirmed P1 thread `PRRT_kwDOQ5Uhrs6baccv` shows that the receipt
  samples columns, constraints, and unique/exclusion indexes but not the open
  class of table-local DML interceptors. A user `BEFORE INSERT` trigger can
  reject the alert writer after every current receipt boolean is true.
- Decision-Seam Analysis — decision: the single `attested`/`not_attested`
  admission verdict for the named 272 receipt. Why the prior decision is
  wrong: it used enumerated catalog object kinds as a proxy for all table-local
  write semantics, so unreviewed trigger/rule/RLS mediation remained an open
  category. Structural default: fail closed when the ordinary target table has
  any noninternal trigger, non-`_RETURN` rewrite rule, enabled or forced RLS,
  or row-security policy. This includes disabled user triggers because the
  source receipt does not prove a later session cannot enable them. The cheap
  error is preserving generic migration blocking rather than admitting a
  historical exception against unreviewed write behavior.
- Required surface: extend only the named 272 metadata query, attestation
  payload/status, fake-catalog matrix, runner retry fixture, and disposable
  retained-DDL fixture. Prove all four catalog families, including multiple
  trigger shapes, block pending SQL before the probe migration can write.
- Explicit residual: confirmed P1 thread `PRRT_kwDOQ5Uhrs6baccy` identifies a
  pre-existing global migration-runner execution model: its session advisory
  lock serializes cooperating Atlas runners but not a privileged external
  database session that alters the ledger or schema after attestation and
  before pending SQL. This provenance slice neither grants such mutation nor
  changes the runner's autocommit model, which must support packaged `CREATE
  INDEX CONCURRENTLY` statements. The bounded operational assumption is that
  external schema/ledger mutation is prohibited while migrations run; if it is
  suspected, stop the rollout and rerun preflight. A compatible locked
  revalidation/serialization design is deferred to [#2476](https://github.com/canfieldjuan/ATLAS/issues/2476), not smuggled into this receipt PR.
- Non-scope: no generic migration-runner transaction/locking redesign,
  migration SQL, target data or ledger mutation, alert-writer change, or broad
  table-shape policy.
- Verification: focused 272 preflight/runner proof, the disposable PostgreSQL
  hook cases in GitHub's configured migration job, syntax/lint/plan gates, and
  read-only current-target metadata inspection. The global execution residual
  is tracked rather than misrepresented as solved by this receipt.

## Scope (this PR)

Ownership lane: `eom/migration-content-integrity`
Slice phase: Production hardening
Max files: 8

- Add one immutable 272 record requiring version `-3`, NULL digest, exact UTC
  recorded time, and the original base-table catalog contract.
- Make the catalog proof metadata-only: permanent ordinary non-partition
  table, 20 source-era columns plus the exact later `reopen_count` signature,
  no unlisted non-dropped user column, named PK/FKs/checks with case- and
  content-preserving literal-safe expressions, no unlisted constraints, three
  ready indexes, no unlisted unique/exclusion index, and no unreviewed
  table-local DML interceptor.
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
    source-era and exact known-later catalog metadata; settled by the focused
    preflight matrix.
  - [ ] Wrong/missing/duplicate ledger evidence, non-permanent/non-table/
    partition relation metadata, altered source-era or known-later column
    signature, an unlisted non-dropped column that can reject the writer, an
    altered constraint/index (including content- or case-distinct quoted
    literals), an unlisted constraint, an unlisted unique/exclusion index, or
    an unready index returns `not_attested`; every closed-signature guard has
    an isolated fake-catalog failure, and the runner leaves pending SQL and its
    ledger row absent; settled by preflight/runner tests and disposable
    PostgreSQL cases.
  - [ ] A noninternal trigger (enabled or disabled), non-`_RETURN` rewrite
    rule, enabled/forced RLS, or any table row-security policy returns
    `not_attested`; all are metadata-only and no pending probe SQL executes;
    settled by the isolated fake-catalog matrix and retained-DDL disposable
    PostgreSQL cases.
  - [ ] A successful 272 predicate clears only its reported ledger name; an
    arbitrary missing source remains in `missing_source`; settled by the runner
    non-generic-admission proof.
  - [ ] The real `run_migrations` entrypoint, against a disposable schema built
    by executing retained `273_b2b_watchlist_alert_events.sql` and the known
    later `281_b2b_watchlist_alert_reopen_count.sql` solely as actual producer
    fixtures (not 272 source evidence), creates exactly one probe table/hashed
    receipt after attestation and no duplicate receipt on retry; settled by the
    PostgreSQL test enrolled in migration CI.
  - [ ] Current target read-only preflight records 272 as attested but remains
    exit 2 because other H-18 discrepancies exist; settled by the target probe.
- Reachability proof: `run_migrations` executes the pending probe from a
  disposable schema; observable state is no probe/ledger row on failure or one
  probe/hashed ledger row on successful retry.
- Affected surfaces: migration admission, read-only provenance preflight, and
  PostgreSQL catalog metadata. No public API or configuration surface changes.
- Risk areas: provenance, migration safety, false admission, privacy, and
  backward compatibility.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R8, R10, R13, R14.

### Closure declaration

- Reconciliation registry: CLOSED. The dispatcher admits one named immutable
  record per discrepancy; unknown source names have no path.
- 272 catalog signature: CLOSED for the 20 source-era columns plus the exact
  known later `reopen_count` column, constraints, and write-restricting
  indexes. No other non-dropped user column is allowed. Required constraints
  are enumerated and no unlisted table constraint is allowed; named indexes are
  exact and no unlisted unique/exclusion index is allowed. The table has no
  user DML mediator: any noninternal trigger, non-`_RETURN` rewrite rule,
  RLS enablement/force flag, or policy closes the receipt. Later non-unique
  performance indexes remain outside this boundary.

### Boundary-change enumeration

- Boundary: report-derived missing-source name -> closed registry -> named
  catalog predicate -> runner admission.
- Replaced behavior: only a complete 272 receipt stops that exact record from
  blocking pending migration SQL. All other source gaps retain current blocking
  behavior.
- Guard fields: name, synthetic version, NULL digest, timestamp, permanent
  relation identity, source-era and known-later column/default signatures, no
  unlisted non-dropped user column, named constraints and case- and
  content-preserving expressions, no unlisted constraints, indexes/readiness,
  no unlisted unique/exclusion indexes, and absence of table-local user DML
  interceptors.
- Caller and catalog-state disposition:

  | Caller | Catalog state | Disposition |
  |---|---|---|
  | `run_migrations` named pending 272 receipt | Complete approved metadata and no user DML mediator | Preserved: admit only the named receipt. |
  | `run_migrations` named pending 272 receipt | Any noninternal trigger, user rewrite rule, RLS enablement/force, or policy | Intentionally changed: `not_attested`; leave pending SQL and probe ledger row absent. |
  | Read-only preflight | Same metadata states | Preserved: expose a boolean-only receipt payload; never query alert rows or write state. |
  | Privileged external database session after the snapshot | Schema/ledger mutation outside the cooperating runner lock | Deferred: global execution-model follow-up in [#2476](https://github.com/canfieldjuan/ATLAS/issues/2476); stop and rerun preflight rather than claiming this receipt serializes external operators. |

## Mechanism

`HistoricalVersionedMissingSourceReconciliation` records only the named 272
identity. A single read-only PostgreSQL catalog query returns structural
metadata for the source-era table and its sole known later writer-required
column; Python tokenizes quoted literals before normalizing unquoted SQL and
compares a closed, case- and content-preserving signature plus metadata-only
absence booleans for unlisted columns, constraints, and write-restricting
indexes, and table-local user DML interceptors, emitting booleans only. The
interceptor predicate fails closed for any noninternal trigger, non-`_RETURN`
rule, RLS relation flag, or policy rather than enumerating individual DML
operations. The disposable test executes retained 273 and 281 DDL only to
produce the expected table catalog contract; it never asserts that either
verifies or replaces 272 source bytes. The existing dispatcher intersects
report-derived missing-source names with its closed registry, so an unlisted
name cannot use this receipt. No query selects alert-event rows or writes
`schema_migrations`; rollback is ordinary code rollback, which returns 272 to
generic blocking behavior. The receipt runs within the existing
cooperating-runner advisory-lock model; it does not claim to serialize a
privileged external schema or ledger writer after its snapshot.

## Intentional

- No generic allowlist or alias mechanism.
- No source reconstruction, hash backfill, or 273-to-272 rename inference.
- No automatic allowance for later columns: `reopen_count` is the one retained,
  writer-required later migration and has an exact reviewed signature.
- No rejection of later non-unique performance indexes; only extra unlisted
  columns, constraints, unique/exclusion indexes, and user DML mediators can
  change write admissibility here.
- No special case for a known trigger shape: every user trigger is unreviewed
  source-era write behavior, including one currently disabled in the target.
- No migration-runner locking redesign in this receipt slice. Session advisory
  locking remains the existing closed component for cooperating runners; the
  external-admin execution residual is explicit below.

## Deferred

H-18 records 386, 297, and 379 remain independent blockers in #2476. The
historical source-number collision root cause remains forensic-only; any broad
renumbering needs its own issue. The target is still blocked from migration 389
until those records independently have reviewed evidence.

The global migration-evidence execution model is deferred in
[#2476](https://github.com/canfieldjuan/ATLAS/issues/2476): determine a
database-lock or locked-revalidation protocol that preserves ledger/catalog
evidence against privileged external schema or ledger mutation while retaining
support for `CREATE INDEX CONCURRENTLY`, then prove its interleavings in a
dedicated runner slice. The safe current operational boundary is cooperative
Atlas runners only; an out-of-band mutation during a migration attempt requires
stopping the rollout and rerunning the read-only preflight.

Parking predicate: generic runner execution-model redesign and all other H-18
records remain parked. Parked hardening: external migration-evidence
serialization, tracked in #2476.

## Verification

- `python -m pytest -q tests/test_migration_content_integrity_preflight.py -k '272 or known_historical'` — 57 passed, 65 deselected.
- `python -m pytest -q tests/test_migrations_runner.py -k '272 or missing_source or historical_attestation'` — 12 passed, 61 deselected.
- `python -m pytest -q tests/test_b2b_watchlist_alert_events_migration_repair.py` — 18 skipped; no disposable test database was configured locally. The same suite is now enrolled in the existing GitHub PostgreSQL migration job.
- `python -m py_compile ...` and `ruff check ...` — passed for all four changed
  Python files.
- `python scripts/check_guard_class_closure.py --base origin/main --strict` —
  passed; the declared closed registry and signature have property proof.
- `python scripts/check_migration_content_integrity.py --expected-target 'host=localhost, port=5433, db=atlas' --attest-known-reconciliations` —
  read-only exit 2; named 272 is attested with permanent storage, the exact
  approved column set, no unlisted constraints, no unlisted unique/exclusion
  indexes, no unreviewed table-local DML interceptors, and case-preserving
  catalog evidence, while independent H-18 records remain blocking. Broad unit
  suite and required workflows are GitHub CI only per operator direction.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_migrations_runner_checks.yml` | 3 |
| `atlas_brain/storage/migrations/reconciliation.py` | 791 |
| `plans/INDEX.md` | 3 |
| `plans/PR-H18-Watchlist-Alert-Events-Source-Attestation.md` | 347 |
| `plans/archive/PR-H18-B2B-Campaign-Partner-Source-Attestation.md` | 0 |
| `tests/test_b2b_watchlist_alert_events_migration_repair.py` | 509 |
| `tests/test_migration_content_integrity_preflight.py` | 565 |
| `tests/test_migrations_runner.py` | 199 |
| **Total** | **2417** |
