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
  launcher from `HEAD` into isolated Python, load the real entrypoint from the
  validated Git object, directly compare tracked Python bytes/modes with
  `HEAD`, and reject untracked divergence, ignored import artifacts, ignored
  package/module-file symlinks, and cached bytecode.
- The receipt must be created before the runtime, use mode 0600, bind the exact
  Git SHA and executed script SHA-256, persist only allowlisted counts and
  changed contact UUIDs, and finalize every normal or exceptional exit without
  overwriting a collision.
- Calendar extraction, identity resolution, write semantics, terminal output,
  credentials, configuration, and customer data must not change.

## Scope (this PR)

Ownership lane: eom/operational-evidence
Slice phase: Production hardening
Max files: 5

1. Keep one shared execution-receipt lifecycle.
2. Bootstrap its source-trust preflight from the reviewed Git object, then load
   the Calendar entrypoint from that exact Git SHA before exposing repository
   import roots.
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
- Structural repair: a receipted invocation pipes the allowlisted launcher from
  `HEAD` into Python `-I`. The launcher compares every tracked Python worktree
  blob and executable mode directly with the `HEAD` tree, runs the remaining
  ignored/cache checks, and only then executes the Calendar entrypoint loaded
  from that exact Git SHA.
- Proof: process tests execute with hostile `PYTHONPATH` startup hooks,
  a self-restoring mutable entrypoint, self-removing ignored shadows, ignored
  package/module-file symlinks, and a modified `skip-worktree` dependency. None
  can execute before rejection.

### Review Contract

- Acceptance criteria:
  - [ ] Calendar live-write rejects missing `--receipt-dir` before local
    imports or async runtime; unreceipted dry-run remains available.
  - [ ] Receipted dry-run/write starts from the `HEAD`-loaded isolated
    launcher, authenticates the checkout, and executes the allowlisted Calendar
    entrypoint from the validated Git SHA before any mutable entrypoint or
    repository dependency can run.
  - [ ] Source trust rejects tracked/untracked divergence, index-hidden tracked
    byte or mode changes, ignored import shadows, package and module-file
    symlinks, and cached bytecode.
  - [ ] A durable in-progress artifact precedes the async runtime; payloads
    contain only fixed lifecycle/source fields, allowlisted non-negative counts,
    and valid sorted contact UUIDs.
  - [ ] Normal, `SystemExit`, interrupt, and exceptional exits finalize
    truthfully; collisions and failed publication retain recovery evidence.
  - [ ] Calendar create, update, claim, interaction-only, and race-merge writes
    record the affected contact UUID. Evidence failure stops before the first
    mutation or, for an in-flight contact, before the next.
  - [ ] Existing focused Calendar behavior remains green.
- Reachability proof: from the repository root, pipe
  `git show HEAD:scripts/eom_execution_receipt.py` into
  `python -I - --launch-reviewed scripts/import_eom_customers_live.py
  --receipt-dir ...`; a valid run publishes the observable mode-0600
  `.exit-N.json` receipt, while hostile process fixtures exit nonzero without
  executing their marker payloads or creating a receipt.
- Affected surfaces: the EOM Calendar operator CLI, its source-authentication
  bootstrap, ignored-import classifier, receipt persistence/finalization, the
  private receipt artifact, operator runbook, and focused process/unit tests.
- Risk areas: false source attestation, arbitrary local code execution before
  trust, ignored import shadowing, customer mutation without durable evidence,
  atomic receipt publication, invocation compatibility, and CI maturity
  classification.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10, R12, R13, R14.


### Files touched

- `docs/EOM_RECONCILIATION_RECEIPTS.md`
- `plans/PR-EOM-Execution-Receipts.md`
- `scripts/eom_execution_receipt.py`
- `scripts/import_eom_customers_live.py`
- `tests/test_eom_execution_receipts.py`

## Mechanism

A non-isolated direct invocation first removes its script directory, repository
root, and empty-path entry from `sys.path` before parsing receipt policy with
the standard library. A receipted operator command instead pipes the shared
launcher from the `HEAD` Git object into isolated Python. The launcher compares
tracked Python blobs/modes directly with the resolved tree, executes the
remaining clean-source checks, and then loads the allowlisted Calendar
entrypoint from that exact Git SHA. The entrypoint adds local import roots only
after those checks succeed. Receipt construction reuses the returned Git SHA,
hashes the matching Calendar script, and creates the durable in-progress
artifact before entering the async runtime.

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
  tests/test_sync_eom_portal_customers.py -q` — 151 passed.
- Current-head boundary selection (`isolated or sitecustomize or skip_worktree
  or evidence_failure or reviewed_launcher or module_file_symlink`) — 12
  passed.
- Ruff and Python byte compilation on the receipt helper, Calendar entrypoint,
  and receipt tests — passed.
- Exact scripts maturity ratchet — passed with no baseline change.
- Exact failed `atlas-brain-b2c-core-risk` maturity matrix group (reasoning,
  security, and storage) — passed with no baseline change.
- Guard class-closure advisory — passed.
- `git diff --check` — passed.

### Review reconciliation

- The documented receipted CLI pipes the launcher from `HEAD` into `python -I
  -`; it authenticates the checkout and loads the entrypoint from the validated
  Git SHA before mutable entrypoint code can execute.
- Tracked Python content and executable modes are read through no-follow file
  descriptors and compared with the `HEAD` tree's Git blob identities.
- Ignored module-file symlinks are normalized through the same import-suffix
  classifier as regular module artifacts; `.py` and `.pyc` held-out cases both
  fail closed.
- Receipt persistence health is asserted before CRM construction, before every
  record, and after each completed record; an in-flight contact may finish, but
  no later contact starts after evidence storage fails.
- The mid-contact health regression uses an explicit dependency seam instead
  of mocking the first-party database singleton, preserving the maturity
  baseline.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/EOM_RECONCILIATION_RECEIPTS.md` | 47 |
| `plans/PR-EOM-Execution-Receipts.md` | 195 |
| `scripts/eom_execution_receipt.py` | 493 |
| `scripts/import_eom_customers_live.py` | 168 |
| `tests/test_eom_execution_receipts.py` | 1005 |
| **Total** | **1908** |
