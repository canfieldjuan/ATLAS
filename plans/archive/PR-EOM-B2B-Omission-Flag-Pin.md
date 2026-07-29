# PR-EOM-B2B-Omission-Flag-Pin

## Why this slice exists

The operator's review of #2166 found one overstatement in its evidence:
the positive behavioral customer-context test pinned
`emails_omitted_under_scope` but not the second runtime flag,
`b2b_enrichment_omitted_under_scope`. The runtime returns both; the test
pinned one.

### Problem-derived contract

- Root cause: the #2166 positive-path test asserted a subset of the
  scoped-omission contract.
- Correct fix must: assert the B2B flag and the emptied
  `b2b_churn_signals` list in the same behavioral test.
- Must not change: production code, other tests.

## Scope (this PR)

Ownership lane: eom-crm/read-scoping
Slice phase: robust testing

1. `tests/test_crm_read_scoping.py`: two asserts appended to
   `test_customer_context_serializes_still_visible_row` pinning
   `b2b_enrichment_omitted_under_scope` and `b2b_churn_signals == []`.

### Review Contract

- Acceptance criteria: the positive scoped-context test fails if either
  omission flag or the emptied enrichment list regresses.
- Reachability proof: pytest suite.
- Affected surfaces: one test.
- Risk areas: none (test-only).
- Reviewer rules triggered: R2, R14.

### Files touched

- `plans/PR-EOM-B2B-Omission-Flag-Pin.md`
- `tests/test_crm_read_scoping.py`

## Mechanism

Two asserts in the existing behavioral test, which already executes the
real MCP path with a stub service.

## Intentional

- Test-only two-line change; no production edits.

## Deferred

- Nothing new.

Parked hardening: none new.

## Verification

- `tests/test_crm_read_scoping.py` — 57 passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-EOM-B2B-Omission-Flag-Pin.md` | 60 |
| `tests/test_crm_read_scoping.py` | 2 |
| **Total** | **~62** |
