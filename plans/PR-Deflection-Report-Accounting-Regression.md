# PR-Deflection-Report-Accounting-Regression

## Why this slice exists

The authoritative source-admission fix (#2071) corrected a report-accounting
defect: a caller-supplied private support-ticket row could re-inject over the
provider's filtered source_material, so the FAQ deflection report counted the
rejected private row as an extra source ("Ticket sources represented: 3" instead
of 2) and emitted a false "appeared only once" singleton warning. That fix is
guarded at the input-provider merge layer
(tests/test_extracted_content_ops_input_provider.py) but has no committed test
at the report-accounting layer -- the mixed public/private execute-path scenario
lived only in an ad hoc validation harness. This slice commits that scenario as
a regression so a future change that reopens the re-injection cannot pass CI.

### Problem-derived contract

- Root cause: the #2071 authoritative-merge behavior is only guarded at the
  package/merge unit layer; the observable report accounting (source count and
  singleton warning on the real execute path) has no regression test, so a
  merge regression would surface only in a live report, not in CI.
- Correct fix must touch/change: add one behavioral regression test on the real
  execute route with the Atlas support-ticket input provider and real FAQ
  deflection report service, asserting the rejected private row is not counted
  as a source and does not trigger the singleton warning.
- Must not change: any runtime or product code, the report shape, the privacy
  classifier, the merge contract, or any adjacent lane. Test-only.

## Scope (this PR)

Ownership lane: resolution-audit/privacy-admission
Slice phase: Robust testing

Max files: 2

1. Add a report-accounting regression test on POST /content-ops/execute that
   submits two public rows plus one caller re-injected private row and asserts
   the report counts only the two admitted sources, omits the singleton warning,
   and never emits the private sentinel.

### Review Contract

- Acceptance criteria:
  - [ ] The test drives the real execute route with the Atlas support-ticket
        input provider, real FAQ deflection report service, and in-memory report
        store -- no mock of the code under test.
  - [ ] It asserts ticket_source_count is 2 and "Ticket sources represented: 2"
        appears in the report markdown, not 3.
  - [ ] It asserts the false "appeared only once" singleton warning is absent.
  - [ ] It asserts the private sentinel is absent from the persisted artifact.
  - [ ] The test fails on pre-#2071 code and passes on current code, so it
        guards the fix rather than asserting vacuously.
  - [ ] No runtime, product, or extracted-package code changes.
- Reachability proof: the test calls the real route end to end and reads the
  persisted artifact back from the store.
- Affected surfaces: test suite only.
- Risk areas: none beyond test determinism; inputs and assertions are fixed.
- Reviewer rules triggered: R14 (reconstruct behavior). No guard is added, so no
  boundary probe is required.

### Files touched

- `plans/PR-Deflection-Report-Accounting-Regression.md`
- `tests/test_extracted_content_ops_live_execute_harness.py`

## Mechanism

The test mirrors the S6A.2 validation scenario as a committed pytest in
tests/test_extracted_content_ops_live_execute_harness.py. It builds
source_material with two public support-ticket rows sharing one question and one
private row whose nested public_comment body is marked "kept private" and
carries a sentinel. It constructs the control-surface router with
build_content_ops_input_provider(), which filters the private row and declares
source_material authoritative, then posts the raw three-row source_material as
caller input -- the re-injection vector. It reads the persisted artifact from
the in-memory store and asserts the source count is 2, the singleton warning is
absent, and the sentinel never reaches the artifact.

## Intentional

- The test asserts on the report markdown and summary ticket_source_count rather
  than provider metadata, because provider included_row_count was already 2
  before #2071; the defect was purely downstream accounting, so the guard must
  live at the report layer.
- The sentinel-absent assertion is defense in depth. Content scrubbing is
  #2061's job and is already covered, but co-asserting it keeps the scenario
  self-documenting.
- Placed in the existing live-execute-harness suite to reuse its real-adapter
  router idiom rather than add a new fixture file.

## Deferred

- The builder's "4 CI proofs" label is not reconciled to named tests; not
  required for correctness, since the superset report and submit suites pass.
  None otherwise.

Parked hardening: none.

## Verification

- run sync_pr_plan.py and the local run before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Report-Accounting-Regression.md` | 109 |
| `tests/test_extracted_content_ops_live_execute_harness.py` | 78 |
| **Total** | **187** |
