# PR-Content-Ops-FAQ-Search-Contract-Handoff

## Why this slice exists

The FAQ search route exists and the seeded route checkers validate the response
shape, but the landing-page/demo session still needs a stable handoff document
that explains what to map from search results versus the hydrated FAQ detail
route. Without that, callers can mislabel `score` as an opportunity percentage,
try to render full answer steps from the lean search result, or treat no-match
responses as errors.

The live hosted SaaS FAQ route proof remains blocked on deployed API URL,
bearer token, account id, and database target inputs. This slice is the
smallest contract slice that can land while those secrets are unavailable.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Functional validation

1. Add a demo/consumer-facing FAQ search route contract handoff document.
2. Explicitly document the search envelope, no-match envelope, `score`
   semantics, lean result fields, and full generated FAQ detail payload.
3. Add a focused regression test that keeps the handoff doc aligned with the
   route checker constants.

### Files touched

| File | Purpose |
|---|---|
| `docs/extraction/validation/content_ops_faq_search_route_contract_handoff.md` | Stable route contract handoff for demo/frontend consumers. |
| `tests/test_check_content_ops_faq_search_route_contract.py` | Pins the handoff doc to the checker-owned required fields and semantics. |
| `plans/PR-Content-Ops-FAQ-Search-Contract-Handoff.md` | Slice contract. |

## Mechanism

The existing `check_content_ops_faq_search_route_contract.py` script already
owns the machine-enforced field lists for search results and hydrated detail
items. The new test reads the handoff doc and asserts it names those fields and
the semantic caveats that are easy for the demo mapper to get wrong:

```python
for field in RESULT_FIELDS:
    assert f"`{field}`" in doc
assert "not a percentage" in doc
assert '{"query": "<query>", "results": [], "count": 0}' in doc
```

That makes the document a maintained consumer contract instead of an unpinned
note.

## Intentional

- No API response shape changes. This is a contract handoff for the shape that
  already exists.
- No hosted-route run is attempted because the required secret/runtime inputs
  are not present in this checkout.
- No generated FAQ detail fields are removed from the checker. The full detail
  route remains the source for steps, term mappings, evidence, and Markdown.

## Deferred

- Live hosted SaaS FAQ route proof remains deferred until deployed API base URL,
  bearer token, matching account id, and database target inputs are available.
- Parked hardening: none. `HARDENING.md` was scanned and has no active FAQ
  search entries touching this doc/test handoff.

## Verification

- `python -m py_compile tests/test_check_content_ops_faq_search_route_contract.py`
  - passed.
- `python -m pytest tests/test_check_content_ops_faq_search_route_contract.py -q`
  - 82 passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-faq-search-contract-handoff-pr-body.md`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 85 |
| Handoff doc | 139 |
| Test | 36 |
| **Total** | **260** |
