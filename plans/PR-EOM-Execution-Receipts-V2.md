# PR-EOM-Execution-Receipts-V2

## Why this slice exists

Issue #2190 blocks production EOM reconciliation apply/onboarding because the
operator tools that mutate the EOM customer master can claim a run happened
without leaving a private, reproducible, source-bound artifact.  PR #2195 tried
to close that gap, but it grew into a frozen 34-commit branch with unresolved
stop-sign threads around cancellation/process semantics.  This successor mines
the useful receipt idea and rebuilds it as the narrow issue #2190 contract
instead of patching #2195 in place.

Diff-budget override: issue #2190 is indivisible because one shared receipt primitive, both mutating EOM entrypoints, the operator runbook, and source/privacy/cancellation tests must ship together or either Calendar import or portal sync remains able to run production writes without durable evidence.

### Problem-derived contract

- Root cause: the live EOM Calendar import and portal customer sync are
  side-effectful operator scripts, but their write/apply modes have no durable
  private artifact that binds the run to the executed source and the non-PII
  outcome.  A successful shell output, screenshot, or deployment note can drift
  from the actual code and cannot prove which contacts were changed.  The prior
  #2195 branch also exposed a deeper execution-boundary issue: if cancellation
  lands while a mutation await is in flight, the process may not know whether
  the mutation committed, so finalizing a normal success/failure receipt would
  overstate certainty.
- Correct fix must touch/change:
  1. Add one shared EOM execution-receipt primitive under `scripts/` that
     creates a mode-0600 in-progress JSON receipt before writes, records only
     allowlisted non-PII fields, binds the payload to the current Git SHA and
     executing script hash, finalizes by atomic non-overwriting publication, and
     leaves the receipt in-progress/indeterminate when cancellation interrupts a
     mutation boundary with unknown commit state.
  2. Wire `scripts/import_eom_customers_live.py` so live writes require
     `--receipt-dir`, dry runs may omit it, outcome counts and changed contact
     UUIDs are recorded, and each CRM/timeline mutation is wrapped in the
     mutation-boundary model.
  3. Wire `scripts/sync_eom_portal_customers.py` so `--apply` requires
     `--receipt-dir`, dry runs may omit it, outcome counts, demotion/keep totals,
     and changed contact UUIDs are recorded, and contact create/update/demotion
     mutations are wrapped in the same mutation-boundary model.
  4. Add focused tests for admission, 0600/private payload shape, atomic
     no-overwrite finalization, source binding, changed-contact capture, and the
     indeterminate cancellation path.
- Must not change: CRM matching semantics, Calendar parsing/deduplication,
  portal roster resolution, demotion eligibility/veto rules, credential
  acquisition, database schema, public APIs, Render/service configuration,
  EOM funnel/customer-onboarding behavior, billing/receivables, website UI, or
  the stale #2195 branch itself.

## Scope (this PR)

Ownership lane: eom/operational-evidence
Slice phase: Production hardening

1. Add private execution receipts to the two issue #2190 EOM operator tools:
   `scripts/import_eom_customers_live.py` and
   `scripts/sync_eom_portal_customers.py`.
2. Add source-bound, non-PII, mutation-aware receipt tests without importing
   host database services into extracted-check collection.

### Review Contract

- Acceptance criteria:
  - [ ] Calendar live write admission fails before provider/database work when
        `--dry-run` is absent and `--receipt-dir` is absent.
  - [ ] Portal sync apply admission fails before login/database work when
        `--apply` is present and `--receipt-dir` is absent.
  - [ ] Receipt creation writes a mode-0600 in-progress artifact before the
        first wrapped mutation and finalizes to a mode-0600 unique final JSON
        path without overwriting an existing final receipt.
  - [ ] Receipt payload contains schema version, receipt id, tool, mode,
        started/ended UTC, git SHA, script hash, exit code, allowlisted outcome
        counts, optional demotion/keep totals, and changed contact UUIDs, and
        does not record customer names, emails, phones, addresses, tokens, base
        URLs, or credentials.
  - [ ] Calendar import records changed contact UUIDs for successful CRM/timeline
        mutations and updates receipt outcome counts after progress.
  - [ ] Portal sync records changed contact UUIDs for create/update/demotion
        mutations and records demotion/keep totals.
  - [ ] If cancellation interrupts while a mutation boundary is active, the
        runner leaves the in-progress receipt indeterminate rather than
        publishing a misleading final receipt.
- Reachability proof: real script entrypoints are exercised via focused tests
  for CLI admission and the shared receipt runner; mutation reachability is
  exercised through the existing script-level stub harnesses for Calendar import
  and portal sync.
- Affected surfaces: `scripts/eom_execution_receipt.py`,
  `scripts/import_eom_customers_live.py`, `scripts/sync_eom_portal_customers.py`,
  focused EOM receipt/portal-sync tests, and this plan.
- Risk areas: operator write admission, artifact privacy, source/run
  reproducibility, cancellation/process semantics, and accidental changes to
  existing CRM reconciliation behavior.
- Reviewer rules triggered: R1, R2, R3, R4, R6, R8, R10, R11, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: EOM operator write/apply admission for Calendar import
  and portal sync; receipt finalization/persistence boundary.
- Replaced-path behaviors: live Calendar writes and portal apply previously
  could run without a durable receipt; after this slice they fail before the
  first mutation unless `--receipt-dir` is supplied.
- Guard-relevant fields: `--dry-run`, `--apply`, `--receipt-dir`, receipt tool
  name, receipt mode, exit code, changed contact UUIDs, outcome count keys.
- Caller x input shape: operator CLI invocation for dry-run/write/apply modes;
  script-level mutation helpers receiving CRM/pool stubs and optional receipt.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: no new environment setting; the operator
  supplies `--receipt-dir` explicitly for write/apply.  The documented default
  is under local private state.
- Explicit value probe: focused CLI/receipt tests pass a temporary receipt
  directory and assert in-progress/final artifact behavior.
- Absent value probe: focused CLI admission tests call write/apply without
  `--receipt-dir` and assert pre-mutation parser failure.
- Default-session/default-context probe: N/A - no tenant/default-context config
  changes.
- Side-effect ordering: receipt is constructed before wrapped mutations; each
  mutation await runs inside a receipt mutation boundary; indeterminate
  cancellation leaves the in-progress artifact unfinalized.

### Files touched

- `docs/EOM_RECONCILIATION_RECEIPTS.md`
- `plans/PR-EOM-Execution-Receipts-V2.md`
- `scripts/eom_execution_receipt.py`
- `scripts/import_eom_customers_live.py`
- `scripts/sync_eom_portal_customers.py`
- `tests/test_eom_execution_receipts.py`
- `tests/test_sync_eom_portal_customers.py`

## Mechanism

The shared receipt module owns JSON payload validation, private file creation,
source binding, and finalization.  Scripts create a receipt only when
`--receipt-dir` is provided; write/apply admission rejects missing receipt dirs
before provider/login/database setup.  Mutating awaits run inside a small
receipt boundary that marks the receipt indeterminate if cancellation lands
before the await completes.  Successful mutation returns then record the
changed contact UUID and update aggregate counts/totals.

## Intentional

- This slice does not keep #2195's reviewed-SHA launcher/snapshot execution
  model.  The issue #2190 root need is private durable source-bound receipts;
  reviewed-SHA process attestation was the scope-expanding branch of #2195.
- Receipt directories are CLI-provided and documented, not configured through a
  new env var, so there is no new deployed config to provision.
- Indeterminate cancellation keeps the in-progress receipt instead of forcing a
  final failure receipt, because the mutation result is unknown at that point.

## Deferred

- Reviewed-SHA launcher/snapshot attestation from #2195 remains deferred unless
  a future operator run needs tamper-evident reviewed-source execution.
- Receipt retention/rotation remains deferred until real artifact volume exists.

Parked hardening: none.

## Verification

- `python -m py_compile scripts/eom_execution_receipt.py scripts/import_eom_customers_live.py scripts/sync_eom_portal_customers.py tests/test_eom_execution_receipts.py tests/test_sync_eom_portal_customers.py` — passed.
- `python -m pytest tests/test_eom_execution_receipts.py tests/test_eom_live_calendar_import.py tests/test_sync_eom_portal_customers.py -q` — 116 passed.
- `python -m ruff check scripts/eom_execution_receipt.py scripts/import_eom_customers_live.py scripts/sync_eom_portal_customers.py tests/test_eom_execution_receipts.py tests/test_sync_eom_portal_customers.py` — passed.
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --top 25` — passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/EOM_RECONCILIATION_RECEIPTS.md` | 31 |
| `plans/PR-EOM-Execution-Receipts-V2.md` | 182 |
| `scripts/eom_execution_receipt.py` | 293 |
| `scripts/import_eom_customers_live.py` | 172 |
| `scripts/sync_eom_portal_customers.py` | 132 |
| `tests/test_eom_execution_receipts.py` | 280 |
| `tests/test_sync_eom_portal_customers.py` | 51 |
| **Total** | **1141** |
