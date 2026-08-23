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
identify the external operation that created that receipt. Earlier read-only
evidence proved the exact bytes and then-observed catalog. The current stricter
trigger-qualification and function-OID predicates have not been re-run against
the target, so this plan does not claim the target currently attests under
those new predicates; the original 379 source remains unavailable.

This is a source-publication and forward-recovery-attestation slice. It keeps
the exact legacy classifier and atomic recovery available for compatible
targets, without rewriting historical ledger facts, creating invoices, or
contacting customers.

The approximately 2,500-LOC diff exceeds the soft cap because the exact
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

### Contract revision after current-head review

- New evidence: three current, non-outdated Codex review threads identify a
  concrete pre-execution failure path. The 379 attestation currently observes
  selected relations, columns, and history triggers but not the declared
  constraint/index member set; the 379 selector admits a named 386 mismatch
  without proving that 386 still attests as `recovery_required`; and the real
  PostgreSQL proof does not execute the recovered trigger's missing-run-ID
  rejection path. A follow-up controlled read-only catalog probe additionally
  shows that a named `CHECK` constraint's `conkey` records only its referenced
  column, not its predicate: replacing `revision > 0` with `revision >= 0`
  would otherwise preserve every currently observed constraint field.
- Revised root cause: the recovery admission decision is incomplete. A target
  that has the expected function hash and selected columns/triggers but lacks a
  required run-isolation constraint or index can reach immutable 391. Likewise,
  a target whose 386 catalog is altered can reach 391 merely because 386 is
  named in the unresolved set. The test suite then proves a successful
  run-scoped insert but not the required failed legacy-provider insert.
- Revised required change surface: before 391 can be selected, the
  reconciliation classifier must attest the closed, declared recovery catalog
  (relations, required columns, required constraint type/key/reference/action
  shape, exact declared `CHECK` predicate semantics, required indexes, and no
  unreviewed catalog members) and require the existing 386 attestation to be
  `recovery_required` whenever its name is the sole concurrent mismatch.
  Focused preflight/fake-runner tests and the isolated PostgreSQL test must
  prove zero prelude SQL/ledger mutations for those rejection states and an
  unchanged invoice count after a missing-run-ID error.
- Revised explicit non-scope: do not edit
  `391_eom_commercial_billing_run_fence_recovery.sql`. The canonical target has
  an observed receipt for source SHA-256
  `117cdd2c509cd89ffaae2efbc4732caf9aea7e155114910a5e5bbe1b5f7d66b7`; changing
  those immutable source bytes would invalidate the very target attestation
  this slice must preserve. The legal correction is the upstream admission
  choke point before the normal runner can execute 391.
- Revised assumptions/blockers: the recovery catalog is intentionally closed
  to the objects needed by the packaged run fence; an unknown member rejects
  admission. No target write is necessary to prove the code path. The isolated
  PostgreSQL test remains conditional on its deliberately configured test URL;
  GitHub owns its broad runnable environment.
- Revised verification plan: run focused reconciliation-preflight,
  migration-runner, and commercial-billing recovery tests plus syntax/plan/
  whitespace checks locally; let GitHub run the full Unit Gate and migration
  suite. Before publish, re-run the target receipt checker in read-only mode
  only if its existing controlled target configuration is available. The
  isolated PostgreSQL catalog matrix includes a same-name, altered `CHECK`
  predicate so the catalog proof cannot pass merely because the object exists.
- Additional target evidence: the controlled read-only catalog query reached
  the new observer and safely returned `not_attested`, not because the target
  lacks the declared recovery objects, but because PostgreSQL stores four
  source identifiers at its physical 63-byte limit and because the exact-source
  candidate index is referenced by snapshot foreign keys on other tables. The
  canonical target therefore proves three observer requirements: normalize
  every explicitly declared catalog name to PostgreSQL's physical identifier
  form, retain the observed physical name for automatically derived foreign-key
  constraints, and treat an index as a constraint-backed index only when that
  constraint belongs to the index-owning relation. Cross-table foreign-key
  references remain catalog evidence for their owning relations; they must not
  make a known direct index look unreviewed.
- Revised required change surface: retain the same closed list and rejection
  semantics, but derive expected physical constraint names from the declared
  source names and join a backing constraint only by both `conindid` and
  `conrelid`. The real PostgreSQL proof must still reject each missing/unknown
  member, while the read-only canonical-target probe must return 379 to its
  prior `attested` state without any write.

### Contract revision after trigger-qualification review

- New evidence: a current Codex blocker demonstrates that the 379 observer
  accepts an enabled invoice-fence trigger with `WHEN (false)`. The existing
  query checks its name, function, type, and enabled state but does not expose
  or require a NULL `tgqual`; a target with the exact 391 receipt and function
  body can therefore be classified `attested` while the database fence never
  executes. The same incomplete trigger-shape predicate is shared by the
  append-only history guards. CI separately proves that the real-catalog
  negative case cannot directly drop the exact-source candidate index because
  three required foreign keys depend on that PostgreSQL key index.
- Revised root cause: the closed catalog admission observes trigger identity
  but not whether each required trigger is unconditional, and one test mutation
  attempts an impossible standalone index drop instead of constructing an
  admissible drifted catalog.
- Revised required change surface: surface `pg_trigger.tgqual` in the 379
  observer and require it to be NULL for the invoice fence and every declared
  append-only trigger. Add fake-query assertions plus both fake and isolated
  PostgreSQL conditional-trigger regressions that prove no 391 SQL or ledger
  receipt occurs. Replace the dependent exact-source index drop with a safe
  rename that preserves its foreign-key dependents while making the declared
  index absent and unreviewed.
- Revised explicit non-scope: do not modify the immutable 391 migration SQL;
  the corrective choke point is the post-receipt reconciliation observer, and
  the disposable mutation changes only its isolated test schema.
- Revised verification plan: run the focused preflight and runner tests plus
  syntax/lint/whitespace locally. GitHub remains the authority for the
  disposable PostgreSQL matrix, which exercises the real `tgqual` catalog
  state and the renamed-index rejection.

### Contract revision after trigger-function binding review

- New evidence: a current Codex blocker demonstrates that the observer obtains
  `pg_trigger.tgfoid` only to read an unqualified function name. The invoice
  predicate joins the body-hashed current-schema function with `ON TRUE`, and
  the append-only history predicates compare only names. A same-named no-op
  trigger function from another schema can therefore leave the reviewed body,
  trigger name, type, enabled state, and `tgqual` unchanged while bypassing
  the fence.
- Revised root cause: trigger attestation is not object-identity attestation.
  The checked function source and the installed trigger are not joined by their
  PostgreSQL function OIDs.
- Revised required change surface: expose every trigger's `tgfoid`, resolve the
  three declared zero-argument trigger functions in `current_schema()`, and
  require each required trigger to reference its corresponding OID. Add fake
  classifier coverage and isolated PostgreSQL foreign-schema same-name
  function mutations for both the invoice fence and an append-only history
  guard. Each mutation must reject before 391 SQL, a recovery receipt, or an
  invoice write.
- Revised explicit non-scope: do not alter immutable 391 migration bytes or
  attempt to rewrite an already recorded recovery. The correction remains the
  upstream reconciliation admission observer.
- Revised verification plan: retain targeted local preflight/runner checks;
  GitHub's disposable PostgreSQL matrix is the runtime authority for the
  cross-schema OID probes. Each foreign-schema mutation must derive and clean
  up a unique scratch schema inside its isolated test run so parameter cases
  cannot share database-level state.

### Contract revision after history-guard body review

- New evidence: a current Codex blocker demonstrates that OID identity is not
  executable-behavior identity. PostgreSQL preserves a function OID across
  `CREATE OR REPLACE FUNCTION`, so an in-schema replacement that returns
  `OLD` leaves the expected trigger identity, type, enabled state, and
  qualification intact while allowing append-only billing-history updates or
  deletes.
- Revised root cause: the invoice fence body is attested, but the two
  append-only history guard bodies are not. A later in-schema function-body
  rewrite can bypass those guards without changing any prior catalog field.
- Revised required change surface: source-derive and pin SHA-256 values for
  the review-decision and override guard bodies, read each current-schema
  `pg_proc.prosrc`, and require both hashes before 379 can be
  `recovery_required` or `attested`. Add fake classifier mutations and isolated
  PostgreSQL `CREATE OR REPLACE FUNCTION ... RETURN OLD` probes for both
  guards; each must reject before 391 SQL, a recovery receipt, or invoice
  writes.
- Revised explicit non-scope: do not alter immutable 391 migration bytes or
  replay historical guard DDL. The fail-closed correction remains the upstream
  reconciliation admission observer.
- Revised verification plan: retain focused local source-hash/classifier and
  runner tests; GitHub's controlled PostgreSQL matrix remains the authority for
  the OID-preserving body-replacement probes.

### Contract revision after invoice-interceptor review

- New evidence: a current Codex blocker demonstrates that the 379 observer
  proves the reviewed invoice fence exists but does not prove it is the only
  non-internal `BEFORE INSERT FOR EACH ROW` trigger on `invoices`. PostgreSQL
  orders triggers of that kind by name, so an unreviewed trigger before the
  fence can change `NEW.source` and another after it can restore the commercial
  source, allowing the reviewed fence to return without checking approval.
- Revised root cause: the closed catalog has no closed set for invoice row
  insert interceptors, leaving a second trigger path that can bypass the
  reviewed function while all currently attested trigger fields remain valid.
- Revised required change surface: expose a catalog boolean requiring the
  reviewed fence to be the only non-internal row-level `BEFORE INSERT`
  interceptor on `invoices`, require it before either 379 admission state, and
  expose it in read-only evidence. Add fake preflight/runner refusal coverage
  and an isolated PostgreSQL pair of alphabetically ordered source-mutating
  interceptors; every drifted case must reject before 391 SQL, a recovery
  receipt, or an invoice write.
- Revised explicit non-scope: do not broaden this recovery into every possible
  invoice rule or policy mechanism, alter immutable 391 bytes, or change the
  production invoice API. The named bypass class is row-level before-insert
  triggers that can mutate `NEW`; the correction is its upstream catalog
  admission predicate.
- Revised verification plan: retain focused local preflight/runner checks,
  syntax, Ruff, plan, and whitespace checks. GitHub's controlled PostgreSQL
  matrix remains the authority for the trigger-order mutation proof.

### Contract revision after behavior-driving-column review

- New evidence: the current 379 observer verifies only a subset of the three
  billing relation columns. Dropping `NOT NULL` from either revision preserves
  the reviewed `CHECK (revision > 0)` and unique key because PostgreSQL accepts
  NULL `CHECK` results and permits multiple NULL key members; the recovered
  `ORDER BY revision DESC LIMIT 1` lookup can then choose an arbitrary event.
- Revised root cause: a partially enumerated column contract leaves type,
  nullability, and newly added user-column drift outside the otherwise closed
  catalog admission predicate.
- Revised required change surface: enumerate every user column of the three
  behavior-driving billing relations with exact PostgreSQL type and nullability,
  reject unreviewed user columns, require both predicates before either 379
  admission state, and expose them in read-only evidence. Add focused fake
  preflight/runner refusal coverage plus isolated PostgreSQL nullable,
  wrong-type, and added-column mutations; each must reject before 391 SQL, a
  recovery receipt, or an invoice write.
- Revised explicit non-scope: do not infer a new business schema, change column
  defaults, rewrite billing history, or edit immutable 391 bytes. This is a
  fail-closed admission correction for the exact source-backed table shape.
- Revised verification plan: retain focused local preflight/runner checks,
  syntax, Ruff, plan, and whitespace checks. GitHub's controlled PostgreSQL
  matrix remains the authority for the catalog-alteration proof.

### Contract revision after fence-read row-security review

- New evidence: the recovered function reads the candidate, override, and
  review-decision tables, but the 379 observer does not inspect RLS flags,
  policies, or rewrite rules on those lookup relations. Forced RLS can hide an
  excluded decision while every previously reviewed fence, column, constraint,
  and index fact remains valid.
- Revised root cause: the closed catalog attests relation shape but omits
  catalog-level query rewriting controls that can alter the function's
  authorization-decision reads.
- Revised required change surface: read `relrowsecurity`,
  `relforcerowsecurity`, `pg_policy`, and non-`_RETURN` `pg_rewrite` entries
  for exactly the three fence-read billing tables; fail closed on any of them,
  require the resulting boolean before either 379 admission state, and expose
  it in read-only evidence. Add fake preflight/runner refusal coverage plus an
  isolated PostgreSQL forced-RLS policy that filters excluded decisions; it
  must reject before 391 SQL, a recovery receipt, or an invoice write.
- Revised explicit non-scope: do not reject the expected append-only mutation
  triggers on review-history tables, change production RLS configuration, or
  edit immutable 391 bytes. The new predicate covers query-rewriting controls
  on the function's reads, not unrelated table behavior.
- Revised verification plan: retain focused local preflight/runner checks,
  syntax, Ruff, plan, and whitespace checks. GitHub's controlled PostgreSQL
  matrix remains the authority for the forced-RLS policy proof.

### Contract revision after trigger-function execution-metadata review

- New evidence: a current Codex blocker demonstrates that the 379 observer
  hashes `pg_proc.prosrc` and binds trigger OIDs, but does not attest the
  runtime metadata of any required trigger function. PostgreSQL can attach a
  function-local `search_path` with `ALTER FUNCTION ... SET` without changing
  either value, and can also change security mode or execution metadata while
  retaining the same object identity. A configured invoice fence can then
  resolve its unqualified candidate/override/decision reads outside the closed
  catalog being attested.
- Revised root cause: object identity plus source-body identity does not prove
  the trigger executes with the reviewed language, security context, planner
  contract, or empty function-local configuration. The closed catalog admits a
  mutable execution environment that can alter fence behavior after recovery.
- Revised required change surface: read the required trigger functions'
  language, kind, volatility, strictness, security-definer, leakproof,
  parallel, support-function, and `proconfig` metadata; require the exact
  source-implied PL/pgSQL, security-invoker, default execution contract with no
  function-local settings before either 379 admission state; and expose the
  result in read-only evidence. Add fake preflight/runner refusal proof plus
  isolated PostgreSQL `ALTER FUNCTION ... SET search_path` and
  `SECURITY DEFINER` cases, each of which must reject before 391 SQL, a
  recovery receipt, or an invoice write.
- Revised explicit non-scope: do not rewrite immutable 391 SQL to add a new
  function configuration clause, modify the existing target's function, or
  treat source hashing alone as an execution-context proof. The correction is
  the fail-closed upstream observer for the existing source-backed runtime
  contract.
- Revised verification plan: run focused preflight and runner cases plus
  syntax/Ruff locally. GitHub's controlled PostgreSQL matrix remains the
  runtime authority for both catalog-level `ALTER FUNCTION` mutations.

### Contract revision after fence-read inheritance review

- New evidence: a current Codex blocker demonstrates that `relispartition`
  does not describe traditional PostgreSQL inheritance. A child of a reviewed
  fence-read table contributes rows to ordinary parent-table queries, while the
  parent's unique constraints and mutation triggers do not constrain that child.
  A higher-revision child decision can therefore supersede an excluded parent
  decision without changing the named parent catalog currently being attested.
- Revised root cause: the closed lookup-read catalog omits `pg_inherits`
  descendants of the three tables read by the recovered fence. The observer
  proves only the named parent relation, not the complete row source of the
  unqualified parent-table reads in the immutable function.
- Revised required change surface: extend the existing closed fence-read
  interceptor predicate to reject every `pg_inherits` edge whose parent is a
  reviewed candidate, override, or review-decision relation. Keep the same
  boolean in the read-only evidence and add static/fake proof plus an isolated
  PostgreSQL inherited review-decision child that must reject before 391 SQL,
  a recovery receipt, or an invoice write.
- Revised explicit non-scope: do not rewrite immutable 391 queries with `ONLY`,
  add an application inheritance feature, or reject inheritance outside the
  three lookup tables. The correction is the closed catalog admission predicate
  for rows visible to this exact fence.
- Revised verification plan: run focused preflight and runner cases plus
  syntax/Ruff locally. GitHub's controlled PostgreSQL matrix remains the
  runtime authority for the parent/child catalog mutation.

### Contract revision after column-physical-contract review

- New evidence: the 379 observer compares a column's base `typname` and
  nullability but omits `atttypmod` and collation. A bounded `VARCHAR` can be
  retagged, or an explicit non-default collation assigned, while the named
  constraint/index catalog remains unchanged and the observer still admits 391.
- Revised root cause: the source-backed column predicate does not attest the
  complete declared physical type contract for the three fence-read billing
  relations. Base type equality alone does not establish `VARCHAR(n)` bounds or
  default-collation semantics.
- Revised required change surface: add each expected PostgreSQL type modifier
  and type-default-collation requirement to the same `required_columns_ready`
  predicate; read `pg_attribute.atttypmod` and compare
  `pg_attribute.attcollation` to `pg_type.typcollation`; retain the existing
  read-only evidence field and admission path. Add static/fake proof plus
  isolated PostgreSQL altered-`VARCHAR`-bound and explicit-collation cases;
  each must reject before 391 SQL, a recovery receipt, or an invoice write.
- Revised explicit non-scope: do not alter the immutable 391 source, prescribe
  a non-default production collation, or create a parallel column-state model.
  The correction closes the existing source-declared column predicate.
- Revised verification plan: run focused preflight and runner checks plus
  syntax/Ruff locally. GitHub's controlled PostgreSQL matrix remains the
  runtime authority for both `ALTER COLUMN` mutations.

### Contract revision after full-index-predicate review

- New evidence: the 379 observer compares an index's identity, key counts,
  readiness, uniqueness, and a definition fragment, but does not inspect
  `pg_index.indpred`. Recreating a declared full index with the same name and
  keys plus `WHERE false` can retain every observed field while making the
  index unusable for the reviewed catalog contract.
- Revised root cause: the source-backed index predicate proves an index exists
  but not that it remains a full index. A definition substring cannot exclude a
  predicate appended after the declared key expression.
- Revised required change surface: expose `pg_index.indpred IS NULL` in the
  existing `target_indexes` evidence and require it for every declared billing
  index before either recovery status. Add fake preflight and runner refusal
  proof plus an isolated PostgreSQL same-name partial-index replacement that
  must reject before 391 SQL, a receipt, or an invoice write. Re-run the
  existing target-confirmed read-only attestation against this final predicate
  and record the result without invoking the runner.
- Revised explicit non-scope: do not alter immutable 391 source bytes, create
  a new index model, or change an index outside the three reviewed billing
  relations. This is an upstream admission check only.
- Revised verification plan: run focused preflight, runner, and recovery tests
  with syntax/Ruff locally; GitHub owns the controlled PostgreSQL matrix and
  full Unit Gate. The target probe must use the existing read-only preflight
  connection and report no credentials or catalog row contents.

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
    target is `attested` only after its own exact 391 digest receipt, the
    run-scoped fence body, and both immutable history-guard bodies; neither
    state claims the unavailable historical source was recovered. Settled by
    `tests/test_migration_content_integrity_preflight.py` and the controlled
    target preflight recorded in #2476.
  - [ ] A legacy 379 state with selected 391 runs only 391 under the existing
    advisory lock and atomic-bookkeeping path, records its digest exactly once,
    re-reads evidence, and leaves ordinary pending SQL blocked while 386 is
    still unresolved; settled by `tests/test_migrations_runner.py`.
  - [ ] An unknown discrepancy, changed legacy fence hash, altered history
    guard body, conditional or foreign-schema same-name required trigger
    function, an unreviewed row-level `BEFORE INSERT` invoice interceptor,
    non-default execution metadata or a function-local setting on a required
    trigger function,
    missing, retagged, nullable, or unreviewed behavior-driving billing column,
    row security, policy, rewrite rule, or inherited child on a fence-read
    relation, missing required
    decision-table catalog member (including a
    declared constraint, its exact `CHECK` predicate, or index), an unreviewed
    catalog member, an omitted 391
    `only=` selection, a 386 mismatch that does not independently attest as
    `recovery_required`, or an already recorded but non-attested 391 causes no
    target SQL/ledger mutation; settled by negative fake-runner and preflight
    cases.
  - [ ] In an isolated PostgreSQL schema, recovery preserves all existing
    decision and override data, changes no row values, and restores run isolation:
    a candidate in run B is not blocked by an override stored only for run A;
    settled by `tests/test_commercial_billing_runs.py`.
  - [ ] A retry after the 391 receipt is a no-op; a legacy provider insert that
    omits `commercialBillingRunId` raises a PostgreSQL error at the recovered
    database boundary and leaves the invoice count unchanged; settled by the
    same disposable PostgreSQL regression.
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
  relation kind/columns/declared constraints including normalized `CHECK`
  predicates/declared indexes/no-unreviewed catalog members/exact user-column
  type-and-nullability set/no query-rewriting control or inherited child on
  fence-read relations/
  closed invoice row-level `BEFORE INSERT` interceptor set/unconditional triggers bound to
  expected current-schema function OIDs/current-schema
  history-guard `pg_proc.prosrc` SHA-256 values/exact trigger-function
  execution metadata (PL/pgSQL, security-invoker/default flags, and empty
  `proconfig`), 386's independently attested
  status, invoice-fence trigger, legacy and recovered invoice-fence
  `pg_proc.prosrc` SHA-256 values, recovery source digest, recovery ledger row,
  and selected pending migration names.
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

The reconciliation observer separately requires every trigger function used by
that fence to retain the source-implied PL/pgSQL, security-invoker, default
execution metadata and no function-local settings. This prevents an
OID-preserving `ALTER FUNCTION` from changing the environment in which the
unqualified fence reads resolve while the source hash remains unchanged.

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
- Restrict the closed-interceptor predicate to non-internal row-level `BEFORE
  INSERT` triggers on `invoices`: this is the PostgreSQL execution class that
  can rewrite `NEW` around the reviewed invoice fence. Statement triggers and
  unrelated invoice mechanisms are not silently claimed as attested by this
  recovery.
- Require the exact source-backed user-column set for the three billing
  relations, including each base type, type modifier, default collation, and
  nullability, before treating a 379 catalog as safe to recover or attested.
  PostgreSQL `CHECK` and unique-key semantics do not substitute for the omitted
  physical-column evidence.
- Reject any RLS flag, policy, or non-`_RETURN` rewrite rule on the three tables
  read by the recovered fence. Expected review-history mutation triggers stay
  governed by their separate immutable-history attestation.
- Reject any traditional inheritance child of the three tables read by the
  recovered fence. A parent scan includes those rows, but a parent table's
  uniqueness and mutation guards do not constrain the child catalog.
- Require the exact source-implied execution metadata for all three trigger
  functions, including an empty function-local configuration. `prosrc` and OID
  equality alone do not constrain `ALTER FUNCTION ... SET search_path` or a
  changed security context.

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
- Historical target proof before later predicate revisions: a read-only
  `scripts/check_migration_content_integrity.py` receipt against the exact
  configured target showed 391's exact source digest and 379 `attested`; it did
  not invoke the runner. The current-head re-attestation is recorded only after
  the final full-index predicate is present.
- Final target proof after the full-index predicate: the same target-confirmed
  read-only preflight ran from this candidate worktree with both the connection
  read-only setting and a read-only transaction. Its outer integrity result was
  `unresolved_drift` solely for the intentionally unavailable historical source;
  the named 379 evidence was `attested` with
  `reviewed_billing_catalog_ready`, `recovery_receipt_ready`, and
  `recovered_catalog_ready` all true. It did not invoke the runner or execute
  migration SQL.
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
| `atlas_brain/storage/migrations/reconciliation.py` | 1061 |
| `plans/PR-H18-Commercial-Billing-379-Run-Fence-Recovery.md` | 669 |
| `tests/test_commercial_billing_runs.py` | 498 |
| `tests/test_eom_render_profile.py` | 1 |
| `tests/test_migration_content_integrity_preflight.py` | 453 |
| `tests/test_migrations_runner.py` | 589 |
| **Total** | **3625** |
