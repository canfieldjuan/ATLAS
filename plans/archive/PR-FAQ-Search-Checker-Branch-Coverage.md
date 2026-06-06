# FAQ Search Checker Branch Coverage

## Why this slice exists

The hosted FAQ search route checker is itself a detector. If its validation or
request-error branches silently stop firing, the operator can get a false pass
from the exact tool meant to catch a broken deployed route.

Current main already covers the largest stale concerns: `count != len(results)`
is pinned, and `_fetch_json` is tested through `urllib.request.urlopen` for
HTTPError, URLError, malformed JSON, and non-object JSON. The remaining gap is
smaller but real: most `main()` tests replace `_fetch_json`, so the CLI path does
not prove transport failures through `main()`, and a few preflight/result-field
branches are not pinned individually.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Functional validation

1. Add negative tests for the remaining FAQ search checker preflight branches.
2. Pin every first-result required-field branch, including missing `question`.
3. Add `main()` tests that mock `urllib.request.urlopen`, not `_fetch_json`, for
   success and request-failure paths.
4. Keep production checker code unchanged unless a test exposes a real bug.

### Files touched

- `plans/PR-FAQ-Search-Checker-Branch-Coverage.md`
- `tests/test_check_content_ops_faq_search_route_contract.py`

## Mechanism

The tests use the existing checker module and add branch-level negative cases:
blank base URL, blank query, non-positive limit, missing required first-result
fields, and `main()` transport behavior through `urllib.request.urlopen`.

The transport tests keep the real `_fetch_json` in play while monkeypatching the
standard-library `urlopen` call. That proves the CLI path exercises the same
request/JSON/error handling an operator uses against a deployed route.

## Intentional

- Tests only unless a bug is discovered.
- No live network call. The transport boundary is covered by monkeypatching
  `urllib.request.urlopen`, not by replacing `_fetch_json`.
- No route/API behavior changes. This hardens the checker around the existing
  `{query, results, count}` contract.

## Deferred

- Future PR: generic checker-testing discipline docs or a mutation-testing
  helper if this pattern recurs across more scripts.
- Parked hardening: none added by this slice.

## Verification

- `python -m pytest tests/test_check_content_ops_faq_search_route_contract.py -q`
  - 39 passed.
- Py compile for `scripts/check_content_ops_faq_search_route_contract.py` and
  `tests/test_check_content_ops_faq_search_route_contract.py`
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| Tests | ~150 |
| **Total** | **~220** |
