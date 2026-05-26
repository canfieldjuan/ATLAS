# PR-Content-Ops-FAQ-Search-Cleanup-Rowcount

## Why this slice exists
#969 added manifest-based cleanup for seeded FAQ search e2e runs. The review
called out one remaining visibility gap: the cleanup result reports the number
of FAQ IDs it attempted to delete, not the rowcount confirmed by Postgres.
That can hide stale cleanup IDs or a delete that matched fewer rows than
expected.
## Scope (this PR)
Ownership lane: content-ops/faq-search
Slice phase: Production hardening.
1. Parse the asyncpg `DELETE N` command tag returned by seeded FAQ cleanup.
2. Report both requested FAQ IDs and actual deleted FAQ rows in cleanup output.
3. Add focused fixtures for valid tags, malformed tags, and the async cleanup
   integration point.
### Files touched
- `plans/PR-Content-Ops-FAQ-Search-Cleanup-Rowcount.md`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
## Mechanism
`_cleanup_seeded_faqs` captures the string returned by `pool.execute(...)`.
A small helper accepts only exact two-token `DELETE <integer>` tags and returns
`None` for unknown or malformed shapes. The cleanup payload keeps
`deleted_faq_ids` as the actual confirmed delete count when parsing succeeds,
adds `requested_faq_ids` for the attempted ID count, and includes the raw
`delete_status` tag for debugging.
## Intentional
- No change to cleanup scope; deletes still target explicit FAQ IDs only.
- Malformed command tags do not fail the whole smoke because the delete already
  completed. The raw tag is emitted so drift is visible in the JSON artifact.
- No broad rowcount mismatch gate yet; this slice surfaces the actual count
  without changing pass/fail semantics.
## Deferred
- A future hosted-data hygiene slice can decide whether `deleted_faq_ids` lower
  than `requested_faq_ids` should fail the e2e run.
- Hosted detail-route expectation checks remain deferred from #969.
## Verification
- pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q - 27 passed in 0.08s.
- python -m py_compile scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py - Passed.
- git diff --check - Passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-Cleanup-Rowcount.md - Passed.
- bash scripts/local_pr_review.sh - Passed.
## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 44 |
| E2E runner | 67 |
| Tests | 122 |
| **Total** | **233** |
