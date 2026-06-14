# PR-Deflection-Statement-Customer-Wording-Smoke

## Why this slice exists

Issue #1478 found that the hosted deflection submit smoke fails on real
statement-shaped exports such as CFPB because it requires every
`snapshot.top_questions[*].customer_wording` value to be non-empty. The product
snapshot contract already permits empty customer wording when a top question is
generated from `source_policy` / fallback wording rather than a customer-shaped
question. The smoke should catch real contract breaks without rejecting valid
statement-shaped exports before launch validation.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Update the deflection submit smoke snapshot validator so blank
   `customer_wording` is allowed for the current real snapshot shape that omits
   `question_source`, and for explicit non-customer generated sources
   (`source_policy` or `topic_fallback`).
2. Add validator tests for both directions: generated/missing-source snapshots
   may omit customer wording, while explicit customer-wording or unknown-source
   questions must still provide customer wording.

### Review Contract

- Acceptance criteria:
  - Statement-shaped snapshots with `question_source: source_policy` and empty
    `customer_wording` pass the smoke validator.
  - `topic_fallback` receives the same allowance for defensive compatibility
    with the label resolver.
  - The current real snapshot shape, which omits `question_source`, can carry
    empty `customer_wording` without failing the smoke.
  - `question_source: customer_wording` or unknown source still fails
    when `customer_wording` is empty.
  - The smoke still rejects forbidden answer/evidence/source-id leaks.
- Affected surfaces:
  - `scripts/smoke_content_ops_deflection_submit_handoff.py`
  - `tests/test_smoke_content_ops_deflection_submit_handoff.py`
- Risk areas:
  - Weakening the smoke so it stops detecting missing customer wording for
    explicitly customer-wording-sourced questions.
  - One-sided coverage that only pins the CFPB/source-policy case and misses
    missing-real-shape or unknown-source regressions.
- Reviewer rules triggered: R1, R2, R10, R13, R14.

### Files touched

- `plans/PR-Deflection-Statement-Customer-Wording-Smoke.md`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

## Mechanism

The smoke validator keeps the existing rank/question/count/frequency checks. If
`customer_wording` is blank, it fails only when the snapshot explicitly provides
a `question_source` outside the generated-source allowlist. This accepts the
current real free snapshot shape, which does not expose `question_source`, and
still rejects explicit customer-wording or unknown-source items with blank
wording.

## Intentional

- This is a smoke-contract fix, not a paid report-generation or free-snapshot
  shape change. The earlier review-fix option of exposing `question_source` in
  the free snapshot was rejected because that is a buyer-facing product-shape
  decision.
- Missing `question_source` with blank wording is intentionally allowed because
  that is the current real emitted snapshot shape. Explicit customer-wording or
  unknown source values remain fail-closed.

## Deferred

- Buyer-visible copy for statement-shaped exports remains under the broader
  #1419/#1386 product-positioning work; this PR only fixes the smoke's contract
  enforcement.
- Existing ticket FAQ label/privacy hardening entries in `HARDENING.md` remain
  parked for their own slices.

Parked hardening: none.

## Verification

- `pytest tests/test_smoke_content_ops_deflection_submit_handoff.py -q` -- 27 passed.
- `bash` + `scripts/run_extracted_pipeline_checks.sh` -- 4112 passed,
  10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Statement-Customer-Wording-Smoke.md` | 96 |
| `scripts/smoke_content_ops_deflection_submit_handoff.py` | 16 |
| `tests/test_smoke_content_ops_deflection_submit_handoff.py` | 57 |
| **Total** | **169** |
