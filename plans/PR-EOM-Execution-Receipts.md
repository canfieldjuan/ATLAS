# PR-EOM-Execution-Receipts

## Why this slice exists

The EOM live Calendar importer can mutate CRM contacts and interactions while
leaving no durable artifact bound to the reviewed source, execution outcome, or
affected contact UUIDs. This slice adds that evidence boundary to one real
entrypoint. Portal-sync receipt wiring is a separate follow-up because
production activation is already operator-gated; it does not need to land in
the same PR.

### Problem-derived contract

- A live Calendar write must fail before its runtime unless a private receipt
  directory is supplied.
- A receipted run must establish source trust before any repository-local code
  can execute. The trusted preflight must reject tracked/untracked divergence,
  ignored import artifacts, ignored package symlinks, and cached bytecode for
  tracked source.
- The receipt must be created before the runtime, use mode 0600, bind the exact
  Git SHA and executed script SHA-256, persist only allowlisted counts and
  changed contact UUIDs, and finalize every normal or exceptional exit without
  overwriting a collision.
- Calendar extraction, identity resolution, write semantics, terminal output,
  credentials, configuration, and customer data must not change.

## Scope

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
  classifier could therefore be erased or bypassed before it ran.
- Structural repair: for a receipted invocation, remove repository roots from
  `sys.path`, parse and enforce write policy with the standard library, load the
  receipt module from `HEAD` through Git, and run its source preflight before
  restoring either local import root. The same preflight rejects importable
  ignored symlinks by filesystem entry type, not by enumerating package names.
- Proof: process tests execute an unpatched CLI with a self-removing ignored
  standard-library shadow and with ignored symlinks to regular and namespace
  packages. Each must fail before the shadow or package can execute.

### Acceptance criteria

1. Calendar live-write rejects missing `--receipt-dir` before local imports or
   async runtime; unreceipted dry-run remains available for development.
2. Receipted dry-run/write establishes source trust before local imports and
   creates a durable in-progress artifact before the async runtime.
3. Receipt payloads contain only fixed lifecycle/source fields, allowlisted
   non-negative counts, and valid sorted contact UUIDs.
4. Normal, `SystemExit`, interrupt, and exceptional exits finalize truthfully;
   collisions and failed publication retain recovery evidence.
5. Calendar create, update, claim, interaction-only, and race-merge writes
   record the affected contact UUID.
6. Existing focused Calendar behavior remains green.

## Intentional

- Dry runs may omit receipts for local development.
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

- Pending after implementation: receipt and Calendar-focused pytest suites.
- Pending after implementation: Ruff, byte compilation, scripts maturity
  ratchet, guard-class closure advisory, and `git diff --check`.
