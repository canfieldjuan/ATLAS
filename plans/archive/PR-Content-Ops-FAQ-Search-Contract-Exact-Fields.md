# PR-Content-Ops-FAQ-Search-Contract-Exact-Fields

## Why this slice exists

PR-Content-Ops-FAQ-Search-Contract-Handoff pinned the FAQ search handoff doc so
it cannot omit checker-owned fields, but review correctly noted that the pin was
one-directional: the doc could still promise an extra lean search-card field the
route does not return. That is the more dangerous frontend failure mode because
a demo mapper may build UI against a non-existent field.

This slice closes that consumer-contract gap while the live hosted SaaS FAQ
route proof remains blocked on deployed API URL, bearer token, account id, and
database target inputs.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Functional validation

1. Parse the handoff doc's `results[0]` search-card field table in the focused
   route contract test.
2. Assert that the documented lean search-card fields exactly match
   `RESULT_FIELDS ∪ {faq_id}`.
3. Leave the broader detail-section coverage as completeness-only because that
   section includes prose lists and generated FAQ field groups.

### Files touched

| File | Purpose |
|---|---|
| `tests/test_check_content_ops_faq_search_route_contract.py` | Adds the exact lean search-card field assertion. |
| `plans/PR-Content-Ops-FAQ-Search-Contract-Exact-Fields.md` | Slice contract. |

## Mechanism

The test extracts only the Markdown table immediately following:

```md
When matches exist, `results[0]` is the lean card/list shape:
```

It then reads the backticked first-column values and compares them to:

```python
{"faq_id", *RESULT_FIELDS}
```

This catches both omission and over-promising in the frontend-facing search
card section.

## Intentional

- No route, checker, or handoff prose changes are needed; the current handoff
  doc already lists the correct fields.
- The exact-set assertion is scoped to the lean search-card table because that
  table maps directly to `results[0]`. Detail prose stays completeness-pinned to
  avoid brittle parsing of grouped generated FAQ lists.
- No hosted-route run is attempted because the required runtime inputs remain
  unavailable in this checkout.

## Deferred

- Live hosted SaaS FAQ route proof remains deferred until deployed API base URL,
  bearer token, matching account id, and database target inputs are available.
- Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ
  search entries touching this test-only handoff guard.

## Verification

- `python -m py_compile tests/test_check_content_ops_faq_search_route_contract.py`
  - passed.
- `python -m pytest tests/test_check_content_ops_faq_search_route_contract.py -q`
  - 82 passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-faq-search-contract-exact-fields-pr-body.md`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 84 |
| Test | 19 |
| **Total** | **103** |
