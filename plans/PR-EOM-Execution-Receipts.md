# PR-EOM-Execution-Receipts

## Why this slice exists

The EOM live Calendar importer can mutate CRM contacts and interactions while
leaving no durable artifact bound to the reviewed source, execution outcome, or
affected contact UUIDs. This slice adds that evidence boundary to one real
entrypoint. Portal-sync receipt wiring is a separate follow-up because
production activation is already operator-gated; it does not need to land in
the same PR.

Diff-budget override: the shared lifecycle, isolated startup/source
attestation, mutation-health boundary, and real process proofs are one
indivisible Calendar receipt contract. Splitting any of them would allow a
production write to claim evidence without proving the code or retaining the
result.

### Problem-derived contract

- A live Calendar write must fail before its runtime unless a private receipt
  directory is supplied.
- A receipted run must establish source trust before any mutable
  repository-local entrypoint or dependency can execute. It must pipe a
  launcher selected by `HEAD` into isolated Python, reject Git replacement refs,
  resolve one full Git SHA with replacement-object processing disabled,
  materialize every tracked Python module from that validated Git object into a
  private read-only snapshot, execute the real entrypoint and imports from that
  snapshot, directly compare tracked Python bytes/modes with the same resolved
  revision, and reject untracked divergence, ignored import artifacts, ignored
  package/module-file symlinks, and cached bytecode.
- The receipt must be created before the runtime, use mode 0600, bind the exact
  Git SHA and executed script SHA-256, persist only allowlisted counts and
  changed contact UUIDs, and finalize every normal or exceptional exit without
  overwriting a collision. Snapshot cleanup failures must not overwrite the
  already-committed process result.
- Calendar extraction, identity resolution, write semantics, terminal output,
  credentials, configuration, and customer data must not change.

## Scope (this PR)

Ownership lane: eom/operational-evidence
Slice phase: Production hardening
Max files: 5

1. Keep one shared execution-receipt lifecycle.
2. Bootstrap its source-trust preflight from the reviewed Git object, then load
   the Calendar entrypoint and repository-local dependencies from a private
   read-only snapshot of that exact Git SHA.
3. Wire Calendar dry-run/write evidence and mutation UUID recording end to end.
4. Document the private operator directory and add lifecycle plus real-process
   boundary proofs.

### Decision-Seam Analysis

- Decision: whether repository-local code may execute before the receipt can
  attest the source.
- Root cause: the CLI exposed repository import roots and imported local modules
  before receipt construction invoked the clean-source check. A later filename
  classifier could therefore be erased or bypassed before it ran; ordinary Git
  status also trusts index flags that can hide tracked divergence.
- Structural repair: a receipted invocation pipes the allowlisted launcher
  selected by `HEAD` into Python `-I`. The launcher rejects replacement refs,
  resolves one full SHA with replacement-object processing disabled, compares
  every tracked Python worktree blob and executable mode directly with that
  resolved tree, runs the remaining ignored/cache checks, materializes all
  tracked Python blobs from that exact Git SHA into a private read-only
  snapshot, and executes the Calendar entrypoint plus later repository imports
  from the snapshot.
- Proof: process tests execute with hostile `PYTHONPATH` startup hooks,
  a self-restoring mutable entrypoint, self-removing ignored shadows, ignored
  package/module-file symlinks, replacement refs, a modified `skip-worktree`
  dependency, invalid receipted arguments, a cleanup failure, and a concurrent
  post-preflight dependency rewrite. Rejected inputs never execute; the
  concurrent rewrite cannot displace the reviewed snapshot; cleanup cannot
  displace an already-finalized receipt.

### Execution model

- Selected closed-surface components: Git's content-addressed object database
  supplies the exact immutable blobs for one resolved commit; the operating
  system supplies a private temporary directory and owner-only read/execute
  permissions. This slice composes those established primitives and introduces
  no lock, lease, retry protocol, shared state machine, or cross-process
  coordination component.
- Admitted execution: the launcher rejects replacement refs, resolves one SHA
  before tracked-source validation, disables replacement-object processing for
  attestation/materialization Git subprocesses, batch-reads every tracked Python
  blob by object ID, closes each snapshot file, removes write permission from
  every snapshot file and directory, then executes the entrypoint with only the
  snapshot's repository roots on `sys.path`.
- Invariant for every admitted worktree-edit interleaving: once preflight
  returns, no subsequent worktree Python edit can change entrypoint or
  dependency bytes used by that run. The receipt SHA and script hash therefore
  describe the code that can mutate CRM state.
- Cancellation/crash boundary: normal exceptions and interpreter cancellation
  attempt snapshot removal in `finally`; cleanup failure is reported on stderr
  but cannot change a committed receipt outcome. No CRM runtime starts before
  snapshot completion. An uncatchable process or host termination may leave an
  owner-private, read-only copy of public repository source in the system temp
  directory. If termination lands after receipt construction, the existing
  in-progress recovery artifact remains truthful and unfinalized.
- Explicit assumptions: the local Git object database, Git executable, Python
  interpreter/standard library, OS same-user isolation, and installed
  third-party packages are trusted. A hostile same-UID process that tampers
  with Git objects or the private temp directory is outside this local
  source-attestation model. Operator configuration, credentials, and
  cwd-relative data remain intentionally external inputs rather than reviewed
  code.
- Surface bound: this is the receipt launcher's single source-identity execution
  surface. It does not add a second durability/concurrency subsystem.

### Review Contract

- Acceptance criteria:
  - [ ] Calendar live-write rejects missing `--receipt-dir` before local
    imports or async runtime; unreceipted dry-run remains available.
  - [ ] Receipted dry-run/write starts from the `HEAD`-loaded isolated
    launcher, rejects replacement refs, authenticates the checkout against one
    resolved Git SHA, and executes the allowlisted Calendar entrypoint plus
    later repository imports from a private read-only snapshot of that validated
    Git SHA.
  - [ ] Source trust rejects tracked/untracked divergence, index-hidden tracked
    byte or mode changes, ignored import shadows, package and module-file
    symlinks, and cached bytecode.
  - [ ] A durable in-progress artifact precedes the async runtime; payloads
    contain only fixed lifecycle/source fields, allowlisted non-negative counts,
    and valid sorted contact UUIDs.
  - [ ] Normal, invalid-argument, `SystemExit`, interrupt, and exceptional exits
    finalize truthfully; collisions and failed publication retain recovery
    evidence; snapshot cleanup failures cannot override committed outcomes.
  - [ ] Calendar create, update, claim, interaction-only, and race-merge writes
    record the affected contact UUID. Evidence failure stops before the first
    mutation or, for an in-flight contact, before the next.
  - [ ] Existing focused Calendar behavior remains green.
- Reachability proof: from the repository root, pipe
  git show HEAD:scripts/eom_execution_receipt.py into
  python -I - --launch-reviewed scripts/import_eom_customers_live.py
  --receipt-dir ...; a valid run publishes the observable mode-0600
  .exit-N.json receipt. A write-mode process fixture performs an observable
  mutation and finalizes that receipt; hostile fixtures exit nonzero without
  executing marker payloads, invalid receipted arguments publish exit-2 receipts,
  and a concurrent worktree rewrite does not execute the replacement dependency.
- Affected surfaces: the EOM Calendar operator CLI, its source-authentication
  bootstrap, ignored-import classifier, receipt persistence/finalization, the
  private receipt artifact, operator runbook, and focused process/unit tests.
- Risk areas: false source attestation, preflight-to-import TOCTOU, arbitrary
  local code execution before trust, ignored import shadowing, customer
  mutation without durable evidence, atomic receipt publication, invocation
  compatibility, and CI maturity classification.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10, R12, R13, R14.


### Files touched

- `docs/EOM_RECONCILIATION_RECEIPTS.md`
- `plans/PR-EOM-Execution-Receipts.md`
- `scripts/eom_execution_receipt.py`
- `scripts/import_eom_customers_live.py`
- `tests/test_eom_execution_receipts.py`

## Mechanism

A non-isolated direct invocation first removes its script directory, repository
root, and empty-path entry from `sys.path` before scanning only the
help/dry-run/receipt policy needed to reject live writes before local imports.
A receipted operator command instead pipes the shared launcher selected by
`HEAD` into isolated Python. The launcher rejects replacement refs, resolves
one full SHA, compares tracked Python blobs/modes directly with that resolved
tree, executes the remaining clean-source checks, then batch-loads every
tracked Python blob into a private mode-0500/0400 snapshot with replacement
objects disabled for each Git subprocess. The allowlisted Calendar entrypoint
and all later repository imports resolve from that snapshot, while cwd-relative
operator data remains outside it. Receipt construction reuses the returned Git
SHA, hashes the snapshot Calendar script, and creates the durable in-progress
artifact before full argument parsing or async runtime entry.

The helper has no generic metadata sink: callers can record only allowlisted
non-negative counts and UUID contact IDs. Each evidence update atomically
replaces and syncs the in-progress artifact. Finalization writes a complete
staged payload, hard-links it to an exclusive exit-specific name, syncs the
directory, and only then removes recovery artifacts. Persistence failures remain
deferred through an already-started contact, but an explicit health check blocks
the first mutation and every subsequent contact boundary.

## Intentional

- Dry runs may omit receipts for local development; every receipted operator
  command pipes the `HEAD` launcher into `python -I -`.
- Direct execution of the mutable worktree entrypoint with `--receipt-dir` is
  rejected because isolated imports alone do not authenticate that script.
- Receipts are private operator evidence, not a public API or database audit
  table.
- Final publication uses an exclusive hard link so a collision cannot overwrite
  an earlier artifact.

## Deferred

- Portal sync receipt wiring and portal-specific totals move to a follow-up
  vertical slice.
- Production execution remains operator-gated until all apply blockers merge.
- Retention and rotation wait for real artifact-volume evidence.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_eom_execution_receipts.py
  tests/test_eom_live_calendar_import.py
  tests/test_sync_eom_portal_customers.py -q` — 160 passed, 1 unrelated
  `torch.cuda`/`pynvml` deprecation warning.
- Current-head boundary selection (`isolated or sitecustomize or skip_worktree
  or evidence_failure or reviewed_launcher or module_file_symlink or
  invalid_arguments or replacement_refs or cleanup_failure`) — 19 passed, 35
  deselected.
- Ruff and Python byte compilation on the receipt helper, Calendar entrypoint,
  and receipt tests — passed.
- Exact scripts maturity ratchet — passed with no baseline change.
- Exact failed `atlas-brain-b2c-core-risk` maturity matrix group (reasoning,
  security, and storage) — passed with no baseline change.
- Real-repository snapshot benchmark — 2,306 tracked Python files materialized
  in 0.250 seconds on the committed tree.
- Guard class-closure advisory — passed.
- `git diff --check` — passed.

### Review reconciliation

- The documented receipted CLI pipes the launcher selected by `HEAD` into
  `python -I -`; it rejects replacement refs, resolves one Git SHA with
  replacement-object processing disabled, authenticates the checkout against
  that pinned revision, and loads the entrypoint plus all later repository
  Python imports from a private read-only snapshot of the validated Git SHA.
- Tracked Python content and executable modes are read through no-follow file
  descriptors and compared with the pinned reviewed tree's Git blob identities.
- Ignored module-file symlinks are normalized through the same import-suffix
  classifier as regular module artifacts; .py and .pyc held-out cases both
  fail closed.
- Receipt persistence health is asserted before CRM construction, before every
  record, and after each completed record; an in-flight contact may finish, but
  no later contact starts after evidence storage fails.
- The mid-contact health regression uses an explicit dependency seam instead
  of mocking the first-party database singleton, preserving the maturity
  baseline.
- Process-level write proofs now cover missing-receipt rejection before local
  imports, one observable CRM mutation with a finalized exit-0 receipt, and a
  concurrent post-preflight CRM dependency rewrite that never executes.
- Receipted invalid arguments now publish exit-2 receipts; cleanup failures are
  reported without overriding an already-finalized exit-0 receipt.
- The targeted residential/window usage example is explicitly an unreceipted
  dry run; production targeting continues to use the documented reviewed
  launcher pipeline.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/EOM_RECONCILIATION_RECEIPTS.md` | 50 |
| `plans/PR-EOM-Execution-Receipts.md` | 261 |
| `scripts/eom_execution_receipt.py` | 612 |
| `scripts/import_eom_customers_live.py` | 920 |
| `tests/test_eom_execution_receipts.py` | 1443 |
| **Total** | **3286** |
