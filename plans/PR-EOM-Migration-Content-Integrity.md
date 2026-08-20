# PR-EOM-Migration-Content-Integrity

## Why this slice exists

H-18 in [ATLAS #2363](https://github.com/canfieldjuan/ATLAS/issues/2363)
records two Billing & Payments deployment incidents: a migration filename was
recorded, then later source edits were skipped because the global runner treats
that name as sufficient evidence of application. The resulting provider schema
was incomplete even though the ledger looked current. This is a real money and
deployment-safety risk for the next migration-bearing Billing slice, not generic
process polish.

This first, independently deployable H-18 phase prevents that exact class for
newly applied migrations and makes legacy uncertainty truthful. It deliberately
does not rewrite historical migration records, infer their original contents,
or make an existing deployment fail because its historical ledger predates
content identities.

**Diff-budget exception:** this is expected to approach 1,100 changed lines,
above the 400-LOC soft target, because the indivisible regression proof includes
the real runner's existing fake/executor seam, all four evidence states,
self-recording atomic rollback/retry/persisted-digest validation,
direct-insert recognizer boundaries,
concurrency compatibility, and the opt-in PostgreSQL adapter assertion.
Cutting those tests would leave migration safety dependent on an unproven
internal call shape; no unrelated refactor or product behavior is included.

### Problem-derived contract

- Root cause: `schema_migrations` records only a version and filename. After a
  file has been recorded, `run_migrations` builds its pending set solely from
  the filename, so it has no source identity with which to distinguish an
  unchanged applied file from one whose contents changed after deployment.
- Correct fix must touch/change: the migration-runner bootstrap must
  idempotently add a nullable source-hash column to its own ledger; the runner
  must read each new migration's bytes once, derive SHA-256 from those exact
  bytes, execute the same decoded bytes, and record and verify that digest
  atomically with a direct self-recording SQL insert. The direct-insert
  recognizer must respect PostgreSQL's case distinction for quoted target
  identifiers. A read-only integrity classifier must
  read a present source before it labels a null digest legacy, so it can
  distinguish verified, legacy-unverified, mismatched, and unavailable-source
  records. Focused runner tests must prove normal, self-recording,
  mismatch, missing-source, failure/retry, and legacy-row behavior.
- Must not change: any historical migration SQL file or existing ledger row;
  migration filename/version selection and duplicate-prefix compatibility;
  financial tables, invoices, payments, Gmail, the legacy monthly task,
  deployment credentials, external systems, or current deployment availability
  for a legacy/null ledger row. Phase one must report diagnostics rather than
  block startup on a mismatch; an enforcement/backfill policy is a later H-18
  decision.

## Scope (this PR)

Ownership lane: eom/billing-payments-security-hardening
Slice phase: Production hardening
Max files: 4

1. Add a nullable `content_sha256` column through the runner's already
   idempotent `schema_migrations` bootstrap, under its existing advisory lock.
   It is runner metadata, not a financial-domain migration; the normal
   bootstrap is the only safe way to evolve a ledger that must exist before a
   normal migration can be recorded.
2. Record SHA-256 for migrations newly applied by this runner, including a
   direct migration-owned ledger insert in the same transaction as its SQL and
   digest update, then verify the exact expected digest persisted before that
   transaction commits. Reject a direct self-recording migration with concurrent
   DDL before it writes; a distinct mixed-case quoted lookalike must retain the
   existing autocommit path. Do not
   backfill rows that were already applied before the run began.
3. Add a read-only integrity report for the real runner entrypoint. It derives
   the current migration catalog from the directory and reports verified,
   legacy-unverified, mismatched, and missing-source rows. Mismatch/missing
   diagnostics are observable in logs but do not alter pending selection or
   block phase-one startup.
4. Add focused fake-runner tests plus the existing opt-in PostgreSQL runner
   path where appropriate. No production database or financial record is used.

### Review Contract

- Acceptance criteria:
  1. `run_migrations` hashes the exact source bytes it decodes and executes,
     then passes that digest to `_record_migration`; settled by a temp-file
     runner test that asserts the SQL and stored digest originate from one byte
     payload.
  2. `_ensure_migrations_table` adds a nullable `content_sha256` column
     idempotently before any digest query or record, and it runs under the
     existing session advisory lock; settled by the runner call order and
     focused fake-executor tests.
  3. A direct ledger row written by a pending migration receives and proves the
     exact source digest in that transaction. An injected digest-update failure
     or silent `UPDATE 0` rolls back both its SQL and ledger row so retry
     records it once; a supplied wrong non-null digest is replaced; an existing
     readable legacy null row remains unchanged. Settled by separate
     self-recording, failure/retry, and legacy-row tests.
  4. The integrity classifier derives the migration-file set from the supplied
     directory and classifies four outcomes exactly: matching digest, readable
     null legacy digest, differing/malformed digest, and a recorded name with
     no readable source file; settled by one focused classification test over
     the report, including a present-but-unreadable file.
  5. A mismatch is logged and remains visible while a later pending migration
     can still execute in phase one; settled by a real `run_migrations` test
     that observes the log and resulting record. No hidden fail-open claim is
     made: mismatched rows are never classified as verified.
  6. The runner's existing name/version collision, atomic-bookkeeping,
     concurrently-DDL, retry, no-pending, and unquoted-identifier behavior
     retain their established tests. A mixed-case quoted `"SCHEMA_MIGRATIONS"`
     lookalike does not select the atomic/concurrent rejection path; the
     focused runner suite passes.
- Reachability proof: the real `run_migrations` entrypoint creates/evolves the
  runner ledger, classifies its applied rows, records a digest after executing a
  pending temp migration, and emits the mismatch diagnostic. The observable
  output is the fake/isolated ledger state and captured runner log; there is no
  new public or customer-facing surface.
- Affected surfaces: `atlas_brain.storage.migrations.run_migrations`, its
  internal `schema_migrations` bootstrap/bookkeeping table, and
  `tests/test_migrations_runner.py`.
- Risk areas: migration bootstrap compatibility, self-recording early
  migrations, byte/SQL TOCTOU, legacy ledger truthfulness, repeated startup,
  advisory-lock ordering, and accidental fail-closed deployment behavior.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R8, R10, R12, R13, and R14.

### Boundary-change enumeration

- Boundary path/seam: `run_migrations` is the sole migration-admission and
  bookkeeping entrypoint. Its internal ledger row gains a source identity after
  successfully executing a newly pending file.
- Replaced-path behaviors: filename-only pending selection remains unchanged in
  phase one. The new digest/report path supplements it; it does not rerun a
  mismatched file or reinterpret a legacy row as applied content.
- Guard-relevant fields: `schema_migrations.name` is the existing migration
  identity; nullable `content_sha256` is source evidence. A null, malformed,
  differing, or missing-source value is non-verified by construction.
- Atomic-bookkeeping admission: a migration uses the transaction path when its
  first non-empty line is the existing explicit marker or its executable SQL
  directly inserts `schema_migrations`. A direct self-recorder with executable
  `CONCURRENTLY` is rejected before any write; comments, literals, and an
  unrelated table retain the existing autocommit path.
- SQL-recognizer fields: raw migration SQL after comments and literals are
  masked; the static `INSERT INTO [ONLY] [schema.]schema_migrations` grammar;
  quoted identifiers; and executable `CONCURRENTLY`. Unquoted identifiers are
  PostgreSQL-case-folded, while quoted target identifiers must exactly equal
  lowercase `"schema_migrations"`. Dynamic SQL is outside this static recognizer
  and must use the pre-existing marker.
- Caller x input shape: application/MCP startup calls the real runner with the
  packaged migration directory; focused tests pass a temporary directory and a
  fake or opt-in isolated executor. The runner never accepts a caller-provided
  checksum.

### Guard class-closure declaration

- Content-identity input/set: CLOSED for each run. Migration source membership
  is derived from `sorted(directory.glob("*.sql"))`; applied ledger membership
  is derived from the database rows. Neither list is copied into source or the
  plan as a manually maintained inventory. The report maps each recorded `name`
  to its current source stem exactly once. A row without a current readable
  source is `missing_source`; null, malformed, or unequal evidence is
  non-verified, never silently verified.
- Self-recording recognizer input/set: OPEN raw SQL, with a deliberately narrow
  static grammar rather than a maintained migration inventory. The
  grammar-derived matrix varies SQL operation tokens, executable/comment/literal
  containers, and unquoted/quoted/qualified target keys (including mixed-case
  quoted lookalikes); its independent expected verdict is direct `INSERT` into
  the ledger only. Dynamic SQL is not claimed as auto-detectable: the existing
  marker selects the same atomic path.
- Error direction: reporting rather than blocking is the deliberately cheap
  phase-one direction for ledger evidence. A direct static self-recorder is
  fail-safe transactional; only that self-recorder combined with executable
  `CONCURRENTLY` fails before a write, because PostgreSQL cannot safely run it
  in that transaction.

### Deployed-config probing

N/A — no configuration or environment fallback changes. The existing migration
entrypoint owns the additive bootstrap and all source identity derives from the
local packaged SQL bytes.

### Files touched

- `atlas_brain/storage/migrations/__init__.py`
- `plans/PR-EOM-Migration-Content-Integrity.md`
- `tests/test_commercial_billing_runs.py`
- `tests/test_migrations_runner.py`

## Mechanism

The existing advisory-locked bootstrap will retain `schema_migrations` and
idempotently add nullable `content_sha256`. For each pending file,
`run_migrations` reads the raw file once, computes SHA-256 from those bytes,
decodes the same bytes for `asyncpg.execute`, and sends the digest to its
existing record helper. A direct executable `INSERT INTO schema_migrations`
selects the existing single-connection transaction path, so its SQL and the
just-created row's digest roll back together if either fails; a direct
self-recording migration with concurrent DDL is rejected before execution.
The runner overwrites a newly self-recorded wrong/non-null digest with the
exact source digest, then reads it back and aborts the transaction if it did
not persist. The recognizer applies PostgreSQL case folding only to unquoted
target identifiers, so a distinct mixed-case quoted table is not admitted.
The explicit atomic marker remains available for equivalent dynamic SQL. A row
already in the snapshot is never updated.

The report reads ledger names/digests and derives current source files from the
same migration directory. It reads a present file before classifying its null
or non-null digest, then returns sorted verified, legacy-unverified,
mismatched, and unavailable-source names. The real entrypoint logs
mismatch/unavailable evidence without changing the existing filename-only
pending behavior. That makes new mutation of an applied source detectable while
avoiding a surprise availability change for the historical null ledger
population or an unavailable packaged source.

## Intentional

- `content_sha256` evolves through runner bootstrap rather than a normal SQL
  migration because normal migration bookkeeping needs that internal table
  before it can decide or record any SQL file; `ADD COLUMN IF NOT EXISTS` is
  additive, idempotent, and serialized by the existing advisory lock.
- SHA-256 is calculated over raw bytes, not a re-rendered SQL string, and those
  same bytes are decoded for execution. This avoids a read/hash/execute source
  mismatch.
- Direct ledger inserts select atomic bookkeeping automatically while comments,
  literals, unrelated tables, and mixed-case quoted ledger lookalikes do not.
  Dynamic self-recording SQL must retain the existing explicit marker; this
  keeps the recognition boundary small and does not rewrite historical
  migration text.
- Legacy null rows remain unverified. Writing today's file hash into them would
  falsely claim evidence of the historical deployed content H-18 says is
  missing.
- Mismatch/missing-source evidence logs rather than halts phase-one startup.
  A fail-closed policy, historical reconciliation, deployment gates, and
  rollback runbook require a separately reviewed H-18 phase.
- No content-hash API, environment flag, or customer-visible output is added.

## Deferred

- H-18 phase two: decide whether verified mismatch/missing-source evidence
  blocks a migration run or deployment, with a documented rollout, rollback,
  and operator recovery path.
- Historical forensic reconciliation/backfill: only independently proven
  historical source identities may ever receive evidence, and no automatic
  backfill is permitted.
- Cross-product migration policy for non-Atlas migration runners remains
  separate from this Atlas runner proof.
- H-18 phase two: decide whether dynamic self-recording SQL should receive a
  broader parser or a declared migration-authoring policy. Until that review,
  the existing `-- atlas: atomic-bookkeeping` marker is required for dynamic
  self-recording SQL.

Parking predicate: new migration-framework abstractions, API surfaces,
historical rewrites, and enforcement policy are parked unless required to
record an exact digest for a newly applied migration or to truthfully classify
it.

Parked hardening: dynamic self-recording SQL policy/parser work remains H-18
phase two in ATLAS #2363; it is not required to make direct packaged ledger
inserts rollback-safe today.

## Verification

- Ruff and byte-compilation passed for
  `atlas_brain/storage/migrations/__init__.py` and
  `tests/test_migrations_runner.py`.
- The focused migration/Billing regression selection passed: 198 passed,
  55 skipped, and 1 warning across `tests/test_migrations_runner.py`,
  `tests/test_receivables.py`,
  `tests/test_residential_payment_receipt_delivery.py`,
  `tests/test_commercial_billing_runs.py`, and
  `tests/test_eom_public_onboarding_migration_repair.py`.
- With only a fresh local Docker PostgreSQL 16 instance on loopback and
  `ATLAS_MIGRATION_TEST_DATABASE_URL` set for that command,
  `python -m pytest -q tests/test_migrations_runner.py` passed: 51 passed.
  The workflow-mirror migration/onboarding selection passed: 54 passed. Both
  probes used a disposable container; no production connection was used.
- The exact `atlas_brain/storage` maturity-ratchet command from the workflow
  passed with no new brittleness above baseline. Whitespace and plan-sync checks
  also passed.
- An unrelated cross-layer selection completed its test output but leaked a
  local nested process during teardown, so it is intentionally not counted as
  verification for this runner-only change. The isolated process group was
  stopped; no source, production process, or financial state was changed.
- A broader legacy monthly-invoice selection has two known current-main
  failures in `tests/test_monthly_invoice_generation.py` because the approved
  safety setting disables invoice reminders; the same two tests fail in a
  detached `origin/main` worktree and are already tracked by ATLAS #2271. They
  are not a migration-integrity regression and are not folded into this PR.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/storage/migrations/__init__.py` | 197 |
| `plans/PR-EOM-Migration-Content-Integrity.md` | 294 |
| `tests/test_commercial_billing_runs.py` | 4 |
| `tests/test_migrations_runner.py` | 585 |
| **Total** | **1080** |
