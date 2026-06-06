# PR-Content-Ops-FAQ-Search-Detail-Seed-Expectations

## Why this slice exists
#972 proves that a seeded hosted search hit can be dereferenced through the FAQ
detail route and that the detail route returns the full FAQ envelope. The review
noted one remaining gap: the detail contract does not assert that the returned
detail belongs to the seeded account/target beyond matching `faq_id`.
## Scope (this PR)
Ownership lane: content-ops/faq-search
Slice phase: Production hardening.
1. Add optional expected detail-field assertions to the existing FAQ search
   route contract checker.
2. Thread seeded account/target expectations from route cases into the seeded
   hosted e2e detail checker call.
3. Add negative fixtures for each new expectation branch.
### Files touched
- `plans/PR-Content-Ops-FAQ-Search-Detail-Seed-Expectations.md`
- `scripts/check_content_ops_faq_search_route_contract.py`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_check_content_ops_faq_search_route_contract.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
## Mechanism
The contract checker gains optional CLI flags for expected detail account,
target id, target mode, title, and status. When a flag is provided and
`--require-detail` runs, `_validate_detail` compares the returned detail field
against the expected value and reports a contract error on mismatch.

The seeded e2e detail case already reads the hit case's account and corpus IDs.
It now derives the seeded target id as `support-<corpus_id>` and passes the
known seeded title/status/target mode into the checker. This keeps validation
single-sourced in the checker while proving the detail row matches the seeded
search hit.
## Intentional
- No new database read is added to the e2e runner; it validates through the
  hosted route only.
- Expected detail checks are opt-in for the generic checker so existing manual
  contract probes keep their current behavior.
- The seeded title/target-mode constants mirror the existing seed smoke rows;
  broader seed customization is deferred until the seed smoke needs it.
## Deferred
- A separate slice can make the seeded smoke emit richer route-case metadata if
  the seeded FAQ shape becomes configurable.
- The #972 observation about a distinct "not run" detail state is left parked;
  it is presentation polish, not required for seeded detail correctness.
## Verification
- pytest tests/test_check_content_ops_faq_search_route_contract.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q - 108 passed in 0.18s.
- python -m py_compile scripts/check_content_ops_faq_search_route_contract.py scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_check_content_ops_faq_search_route_contract.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py - Passed.
- git diff --check - Passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-Detail-Seed-Expectations.md - Passed.
- bash scripts/local_pr_review.sh - Passed.
## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 51 |
| Contract checker | 52 |
| E2E runner | 18 |
| Tests | 107 |
| **Total** | **235** |
