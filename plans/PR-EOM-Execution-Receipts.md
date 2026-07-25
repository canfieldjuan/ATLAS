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
- A receipted run must establish source trust before any repository-local code
  can execute. It must start Python in isolated mode, directly compare tracked
  Python bytes/modes with `HEAD`, and reject untracked divergence, ignored
  import artifacts, ignored package symlinks, and cached bytecode.
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
2. Bootstrap its source-trust preflight from the reviewed Git object before the
   Calendar CLI exposes repository import roots.
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
- Structural repair: a receipted invocation is admitted only under Python `-I`,
  then loads the receipt module from `HEAD`, compares every tracked Python
  worktree blob and executable mode directly with the `HEAD` tree, and runs the
  remaining ignored/cache checks before restoring local import roots.
- Proof: process tests execute with hostile `PYTHONPATH` startup hooks,
  self-removing ignored shadows, ignored package symlinks, and a modified
  `skip-worktree` dependency. None can execute before rejection.

### Review Contract

1. Calendar live-write rejects missing `--receipt-dir` before local imports or
   async runtime; unreceipted dry-run remains available for development.
2. Receipted dry-run/write establishes source trust before local imports and
   creates a durable in-progress artifact before the async runtime.
   Non-isolated receipted CLI startup is rejected.
3. Receipt payloads contain only fixed lifecycle/source fields, allowlisted
   non-negative counts, and valid sorted contact UUIDs.
4. Normal, `SystemExit`, interrupt, and exceptional exits finalize truthfully;
   collisions and failed publication retain recovery evidence.
5. Calendar create, update, claim, interaction-only, and race-merge writes
   record the affected contact UUID. A receipt persistence failure stops before
   the first mutation or, if a contact is already in flight, before the next.
6. Existing focused Calendar behavior remains green.


### Files touched

- `docs/EOM_RECONCILIATION_RECEIPTS.md`
- `plans/PR-EOM-Execution-Receipts.md`
- `scripts/eom_execution_receipt.py`
- `scripts/import_eom_customers_live.py`
- `tests/test_eom_execution_receipts.py`

## Mechanism

On a direct invocation, the Calendar CLI first removes its script directory,
repository root, and empty-path entry from `sys.path`. It can then use only
standard-library code to parse receipt policy. A receipted run additionally
requires Python isolated mode, reads the shared helper from the `HEAD` Git
object, compares tracked Python blobs/modes directly with that tree, executes
the remaining clean-source checks, and restores local import roots only after
they succeed. Receipt construction reuses the returned Git SHA, hashes the
executed Calendar script, and creates the durable in-progress artifact before
entering the async runtime.

The helper has no generic metadata sink: callers can record only allowlisted
non-negative counts and UUID contact IDs. Each evidence update atomically
replaces and syncs the in-progress artifact. Finalization writes a complete
staged payload, hard-links it to an exclusive exit-specific name, syncs the
directory, and only then removes recovery artifacts. Persistence failures remain
deferred through an already-started contact, but an explicit health check blocks
the first mutation and every subsequent contact boundary.

## Intentional

- Dry runs may omit receipts for local development; every receipted operator
  command uses `python -I`.
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
  tests/test_sync_eom_portal_customers.py -q` — 146 passed.
- Current-head boundary selection (`isolated or sitecustomize or skip_worktree
  or evidence_failure`) — 6 passed.
- Ruff and Python byte compilation on the receipt helper, Calendar entrypoint,
  and receipt tests — passed.
- Exact scripts maturity ratchet — passed with no baseline change.
- Guard class-closure advisory — passed.
- `git diff --check` — passed.

### Review reconciliation

- A receipted direct CLI invocation now refuses non-isolated startup, and the
  documented process uses `python -I`.
- Tracked Python content and executable modes are read through no-follow file
  descriptors and compared with the `HEAD` tree's Git blob identities.
- Receipt persistence health is asserted before CRM construction, before every
  record, and after each completed record; an in-flight contact may finish, but
  no later contact starts after evidence storage fails.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/EOM_RECONCILIATION_RECEIPTS.md` | 35 |
| `plans/PR-EOM-Execution-Receipts.md` | 157 |
| `scripts/eom_execution_receipt.py` | 448 |
| `scripts/import_eom_customers_live.py` | 164 |
| `tests/test_eom_execution_receipts.py` | 903 |
| **Total** | **1707** |
