# PR-Deflection-Status-Pending-Customer-Open

## Why this slice exists

Issue #1518 found that waiting-on-customer statuses from real support exports
normalize to `other` instead of `open`. The 200k corpus uses `Pending Customer`
and the 29k Kaggle-style export uses `Pending Customer Response`; both are live
backlog states, so bucketing them as `other` undercounts the Tier-2 answer-gap
and churn-risk backlog before the buyer pays.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Extend the deterministic status vocabulary so waiting-on-customer statuses
   map to the existing `open` bucket.
2. Add regression coverage for the cited corpus forms plus diverse same-class
   forms, and keep unrelated customer-looking labels in `other`.

### Review Contract

- Acceptance criteria:
  - `Pending Customer`, `Pending Customer Response`, `Awaiting Customer`, and
    `Waiting on Customer` normalize to `open`.
  - Unknown or non-lifecycle customer labels still normalize to `other`; the
    fix must not over-broaden to every string containing `customer`.
  - Status summary metadata counts the newly recognized open statuses.
- Affected surfaces:
  - `extracted_content_pipeline/support_ticket_input_package.py`
  - `tests/test_extracted_support_ticket_input_package.py`
- Risk areas:
  - Over-capture from generic vocabulary changes.
  - One-sided tests that only prove the cited status and miss nearby false
    positives.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Deflection-Status-Pending-Customer-Open.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

The status normalizer already lowercases and strips non-alphanumeric
characters through `_key()` before comparing against fixed bucket vocabularies.
This PR only adds explicit waiting-on-customer normalized keys to
`_OPEN_STATUS_VALUES`; it does not add fuzzy substring logic or a generic
`customer` rule.

## Intentional

- Keep the fix deterministic and narrow. No LLM, no corpus-specific branch, and
  no catch-all `customer`/`pendingcustomer*` prefix rule, because generic
  matching would risk relabeling non-lifecycle customer fields as open backlog.

## Deferred

- The existing ticket FAQ label/privacy hardening entries in `HARDENING.md`
  are unrelated to status bucket truthfulness and remain parked for their own
  slices.

Parked hardening: none.

## Verification

- `pytest tests/test_extracted_support_ticket_input_package.py -q` -- 61 passed.
- `bash` + `scripts/run_extracted_pipeline_checks.sh` -- 4101 passed,
  10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 5 |
| `plans/PR-Deflection-Status-Pending-Customer-Open.md` | 79 |
| `tests/test_extracted_support_ticket_input_package.py` | 9 |
| **Total** | **93** |
