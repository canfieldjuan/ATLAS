# PR-EOM-Execution-Receipts

## Why this slice exists

Issue #2190 closes an apply-blocking evidence gap found while auditing the
2026-07-24 EOM handoff. The handoff reports a production Calendar import and a
portal-sync dry run, but both operator tools currently print only ephemeral
terminal output. Nothing durable binds a claimed run to the exact Git commit,
script bytes, execution mode, result counts, or affected records. This
production-hardening slice is justified by an operational data-safety risk:
without a private execution receipt, the operator cannot distinguish a
reviewed apply from an older checkout or reproduce the claimed postconditions.

The shared receipt boundary and two real-entrypoint proofs are one indivisible
fix. Splitting the helper from either caller would leave one mutating EOM path
able to run without evidence. The expected diff may exceed the repository's
400-line soft cap because collision, permissions, failure finalization,
allowlist/privacy, and both CLI reachability cases must ship with the runtime
boundary they prove.

### Problem-derived contract

- Root cause: the two EOM mutation entrypoints have no fail-closed receipt
  contract. They can mutate contacts and interactions while retaining no
  durable, source-bound, non-PII artifact, so production-run claims depend on
  terminal narrative rather than reproducible evidence.
- Correct fix must touch/change: add one shared receipt writer with an explicit
  field allowlist; require `--receipt-dir` before Calendar live-write or portal
  `--apply`; optionally receipt dry runs when the directory is supplied;
  create a mode-0600 in-progress artifact before entering either runtime;
  bind it to a UUID receipt ID, UTC start/end, tool/mode, current Git SHA, and
  SHA-256 of the executed script; collect only non-negative outcome/demotion/
  keep counts and valid changed contact UUIDs; finalize success and failure
  atomically under a unique name without overwriting; and prove those behaviors
  through both real CLI parsers plus filesystem tests. Document the recommended
  operator directory under `${XDG_STATE_HOME:-$HOME/.local/state}`.
- Must not change: Calendar extraction/dedupe/recency, CRM identity or write
  semantics, portal authentication/fetch behavior, demotion eligibility or
  veto policy, endpoints, schemas, configuration, credentials, customer data,
  terminal summaries, or the separate one-Site economics reconciler. Receipts
  must never accept or persist customer names, emails, phones, addresses,
  credentials, tokens, or portal/runtime URLs.

## Scope (this PR)

Ownership lane: eom/operational-evidence
Slice phase: Production hardening

1. Add the shared private receipt lifecycle and wire it to the Calendar-import
   and portal-sync CLI entrypoints.
2. Add exact-entrypoint and filesystem proofs for pre-mutation creation,
   fail-closed argument validation, success/failure finalization, collision
   safety, mode 0600, source binding, and payload privacy.
3. Add a short operator runbook for the local-state receipt directory.

### Review Contract

- Acceptance criteria:
  1. Calendar live-write and portal apply reject missing `--receipt-dir`
     before the async runtime or any mutation; both dry-run modes remain
     runnable without it.
  2. Supplying a receipt directory creates a mode-0600 in-progress JSON
     artifact before either CLI enters its runtime.
  3. A run returning zero or nonzero, raising `SystemExit`, or raising another
     exception finalizes a receipt with the correct exit state and end UTC.
  4. Final publication is atomic, uses an exclusive unique path, and cannot
     overwrite an existing receipt; collision leaves the in-progress artifact
     available for recovery.
  5. Every receipt is source-bound by Git SHA and the executed tool's SHA-256.
  6. The payload schema contains only its fixed top-level fields, allowlisted
     count keys, optional portal demotion/eligible/kept totals, and sorted valid
     changed contact UUIDs. PII/private runtime fields have no input path.
  7. Calendar and portal mutation paths record affected contact UUIDs,
     including created/updated/claimed/stamped/demoted rows and partial-failure
     writes already made before an error.
  8. Existing focused Calendar and portal-sync behavior remains green.
- Reachability proof: invoke each script's real `main(argv)` parser in tests,
  assert the in-progress file exists before its injected async runtime begins,
  then observe the finalized JSON artifact and exit code.
- Affected surfaces: one shared operator-script helper, the two EOM CLI
  boundaries, receipt-focused tests, existing focused tests, maturity baseline
  if the repository gate requires it, and one private-state runbook.
- Risk areas: writing after receipt validation is bypassed, failure without a
  final artifact, permissive file modes, overwrite/collision, partial-write
  UUID omission, accidental PII or credential persistence, stale Git/script
  binding, and regression of dry-run/write semantics.
- Reviewer rules triggered: R1, R2, R3, R4, R6, R8, R10, R12, R14.

### Files touched

- `docs/EOM_RECONCILIATION_RECEIPTS.md`
- `plans/PR-EOM-Execution-Receipts.md`
- `scripts/eom_execution_receipt.py`
- `scripts/import_eom_customers_live.py`
- `scripts/sync_eom_portal_customers.py`
- `tests/test_eom_execution_receipts.py`

## Mechanism

Each CLI parses and validates receipt policy before constructing its async
runtime. When a receipt is requested, the shared helper resolves the repository
HEAD, hashes the executed script, creates a UUID-named in-progress file with
`O_EXCL` and mode 0600, and passes a recorder into the existing run. The
recorder exposes only count, demotion-total, and UUID methods; it has no generic
metadata sink. Existing receipt directories must be real, owned by the current
effective user, and not group/world writable.
Before either runtime can start, a short create/link/remove probe verifies that
the selected filesystem supports the exclusive hard-link publication mechanism.

The CLI wrapper maps normal and exceptional exits to an exit code and finalizes
in a `try`/`except` boundary. Finalization rewrites the owned in-progress
artifact with complete data, hard-links it to an exclusive final filename, and
only then removes the in-progress name. A pre-existing final name makes the
link fail instead of overwriting. The directory is synced after initial
in-progress publication and final renaming/linking so both namespace
transitions are durable. Calendar import and portal sync add UUIDs at the
existing successful mutation points, including interaction-only Calendar
writes and provider race merges even when the follow-up reconciliation is a
no-op; portal demotion adds the returned ID, and the portal run publishes
kept/demotion totals without customer text.

## Intentional

- Receipts are local operator evidence, not a database audit table or public
  API. They deliberately contain no customer-level descriptive data.
- Dry runs without `--receipt-dir` remain convenient for development; the
  production runbook always supplies it so dry-run/apply/no-op artifacts can
  be compared.
- An exclusive hard-link publication is preferred over `os.replace` because
  replacement may overwrite a colliding final filename.

## Deferred

- Running the receipted production dry-run/apply/no-op sequence remains an
  explicit operator action after this PR and the other apply blockers merge.
- Receipt retention/rotation policy is deferred until real artifact volume
  exists; receipts are small and private.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_eom_execution_receipts.py
  tests/test_eom_live_calendar_import.py
  tests/test_sync_eom_portal_customers.py -q - 131 passed.
- Command: python -m ruff check scripts/eom_execution_receipt.py
  scripts/import_eom_customers_live.py tests/test_eom_execution_receipts.py -
  passed.
- Command: python -m py_compile on the three review-touched Python files -
  passed.
- Command: python scripts/maturity_sweep.py scripts --tests-root tests
  --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8
  --sensitive-glob 'scripts/**' - ratchet passed with no baseline change.
- Command: git diff --check - passed.
- Pending at push: managed local PR review through scripts/push_pr.sh.

### Review reconciliation

- Calendar interaction-only inserts now add their contact UUID to the receipt.
- Bare `SystemExit()` now records the Python exit status 0.
- Existing receipt directories with a foreign owner or group/world write bits
  are rejected before artifact creation.
- Initial in-progress publication now syncs its directory entry before the
  operator mutation can begin.
- Provider race merges now record the affected UUID before any follow-up no-op.
- Construction now rejects filesystems without hard-link support before either
  runtime, and explicitly sets artifact mode 0600 independent of the umask.
- `SystemExit` integer-like values now match the effective POSIX process status,
  including booleans, negative values, and values above 255.
- Receipted execution now rejects staged or unstaged tracked-file changes before
  entering either operator runtime, so the recorded `HEAD` cannot conceal
  modified dependencies.
- Non-ignored untracked files now fail the same source check, preventing local
  modules under import roots from shadowing reviewed dependencies.
- Every recorded changed UUID or aggregate count atomically replaces and syncs
  the in-progress artifact, preserving partial-run evidence before finalization.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/EOM_RECONCILIATION_RECEIPTS.md` | 34 |
| `plans/PR-EOM-Execution-Receipts.md` | 188 |
| `scripts/eom_execution_receipt.py` | 282 |
| `scripts/import_eom_customers_live.py` | 71 |
| `scripts/sync_eom_portal_customers.py` | 96 |
| `tests/test_eom_execution_receipts.py` | 569 |
| **Total** | **1240** |
