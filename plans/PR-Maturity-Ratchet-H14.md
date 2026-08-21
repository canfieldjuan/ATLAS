# PR-Maturity-Ratchet-H14

## Why this slice exists

ATLAS hardening issue [#2363](https://github.com/canfieldjuan/ATLAS/issues/2363)
tracks H-14: the maturity-ratchet baselines omit two findings that already exist
on `origin/main`.  That makes the workflow-equivalent maturity matrix red for
unrelated Billing & Payments provider PRs, including #2378.  The repair must be
a main-based maintenance slice so it does not conceal financial-slice defects.

### Problem-derived contract

- Root cause: `atlas_brain/api/invoicing/receivables.py` already has seven
  first-party-mock findings but no API baseline entry; meanwhile
  `atlas_brain/templates/email/payment_receipt.py` has no directly discoverable
  test file, which triggers the sensitive-path `NO_TEST_FILE` gate.  The
  `maturity_sweep.py` ratchet therefore classifies both unchanged main paths as
  newly introduced debt.
- Correct fix must touch/change: record only the current receivables score and
  `INTERNAL_MOCK` count in `baseline_atlas_brain_api.json`, and add a direct,
  deterministic `test_payment_receipt.py` which exercises the real receipt
  renderer for check and non-check paths.  Keep the existing ratchet comparator
  unchanged so an eighth mock or future sensitive testlessness still fails.
- Must not change: production receipt rendering, Gmail/email delivery,
  receivables API behavior, payment data, migrations, customer-visible copy,
  financial lifecycle semantics, or the #2378 delivery-preference provider
  diff.

## Scope (this PR)

Ownership lane: repo/maturity-ratchet-h14
Slice phase: Workflow/process
Max files: 3

1. Add the evidence-backed API baseline entry for the seven pre-existing
   receivables first-party mocks.
2. Add direct unit coverage for immutable residential receipt rendering so the
   template's sensitive test-discovery finding is resolved by real execution,
   not hidden in a template baseline.
3. Prove the exact local API and b2a-support maturity commands pass, while the
   existing ratchet unit test proves an additional mock still fails.

### Review Contract

- Acceptance criteria:
  - [ ] `tests/maturity_sweep/baseline_atlas_brain_api.json` records exactly
    the current `atlas_brain/api/invoicing/receivables.py` score of 28 and seven
    `INTERNAL_MOCK` findings; the existing
    `tests/test_maturity_sweep.py::test_internal_mock_ratchet_fails_on_new_mock_target`
    proves the comparator still rejects a new mock.
  - [ ] `tests/test_payment_receipt.py` directly executes
    `render_residential_payment_receipt` and asserts a deterministic receipt
    number, money amount, method/reference fields, business contact details,
    and the received-not-cleared wording only for checks; it performs no Gmail
    or external delivery call.
  - [ ] The workflow-equivalent API and b2a-support `maturity_sweep.py`
    commands report no new ratchet failures at this head.
  - [ ] `git diff --name-only origin/main...HEAD` contains only this plan, the
    API baseline, and the direct receipt-renderer test; functional financial
    source and migrations remain unchanged.
- Reachability proof: N/A for a new runtime surface.  The real maturity
  workflow consumes the baseline and pytest discovers the new test; its
  observable result is the local matrix pass.  The test invokes the actual
  renderer and observes its returned immutable receipt artifact.
- Affected surfaces: maturity-sweep CI matrix, API baseline data, direct
  deterministic template rendering test discovery.
- Risk areas: masking future test debt, accepting a changed mock count,
  accidentally asserting a check as cleared, and test-discovery mismatch.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: `scripts/maturity_sweep.py` baseline comparison consumes
  the API baseline entry for `receivables.py`; its code is unchanged.
- Replaced-path behaviors: none; this adds the absent main-state record rather
  than replacing the comparator or relaxing its rule.
- Guard-relevant fields: the fixed `score` and `counts.INTERNAL_MOCK = 7`.
- Caller x input shape: maturity workflow matrix → JSON baseline → scanned API
  and collected pytest sources; an eighth first-party mock remains an
  invalid input to the ratchet.

### Deployed-config probing

N/A - no guard/config boundary change or deployed configuration.

### Files touched

- `plans/PR-Maturity-Ratchet-H14.md`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/test_payment_receipt.py`

## Mechanism

The API baseline expresses the current, already-scanned floor, not a waiver for
future changes: the existing comparator still fails on score growth or when an
`INTERNAL_MOCK` count rises above seven.  The receipt template receives real
test discovery because its module stem matches `test_payment_receipt.py`; the
test calls the renderer directly with a fixed UUID, date, and Decimal values.
No production code or message is emitted.

## Intentional

- Baseline only the exact existing receivables count instead of rewriting the
  older mock-heavy tests; reducing that legacy debt belongs in a dedicated
  behavior-safe refactor.
- Do not baseline `payment_receipt.py`: exercising its real immutable output
  fixes the missing-test signal while keeping the sensitive zero-tolerance rule
  meaningful.
- Keep this prerequisite separate from #2378 so financial-provider review
  stays focused and its diff budget does not absorb repository maintenance.
- Do not absorb #2378's four new delivery-preference-test mock findings: this
  main-based baseline stops at the seven pre-existing findings, leaving that
  slice to reduce or replace its own test seams.

## Deferred

- H-14's broader future maturity findings remain owned by
  [#2363](https://github.com/canfieldjuan/ATLAS/issues/2363).  This slice parks
  no new unrelated item: it only removes the two verified main-level blockers
  required to restore current ratchet execution.

Parked hardening: H-14 issue #2363 remains the coordinating record for any
subsequent baseline-refresh or test-debt work.

## Verification

- `python -m pytest tests/test_payment_receipt.py -q` — 2 passed.
- `python -m pytest tests/test_payment_receipt.py tests/test_eom_payment_receipts.py -q`
  — 13 passed; no receipt-delivery regression.
- `python -m pytest tests/test_maturity_sweep.py
  tests/test_detect_retired_failure_modes.py
  tests/test_maturity_sweep_advisory_workflow.py
  tests/test_retired_failure_detector_workflow.py --noconftest -q` — 86
  passed, including the ratchet failure-side fixture.
- Exact API and all b2a-support matrix `maturity_sweep.py` commands passed at
  the working head; the template JSON now reports only its non-blocking
  `HAPPY_PATH_TESTS` heuristic, not `NO_TEST_FILE`.
- New-test static linting, bytecode compilation, and whitespace validation
  passed. Plan/body and diff-budget audits run again after commit.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Maturity-Ratchet-H14.md` | 148 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 6 |
| `tests/test_payment_receipt.py` | 69 |
| **Total** | **223** |
