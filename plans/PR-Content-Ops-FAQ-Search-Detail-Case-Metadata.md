# PR-Content-Ops-FAQ-Search-Detail-Case-Metadata

## Why this slice exists
#974 added seeded detail-field assertions, but the review noted that the seed
smoke and e2e runner now duplicate the seeded FAQ title, target mode, status,
and target-id format. That creates drift risk if the seed row changes without
the e2e expectation builder changing with it.
## Scope (this PR)
Ownership lane: content-ops/faq-search
Slice phase: Production hardening.
1. Have the DB seed smoke emit expected detail fields in the route-case file it
   already writes for hosted e2e runs.
2. Keep the seed row values and route-case metadata sourced from the same
   constants/functions inside the seed smoke.
3. Have the seeded e2e consume those expected detail fields instead of
   reconstructing seed literals.
### Files touched
- `plans/PR-Content-Ops-FAQ-Search-Detail-Case-Metadata.md`
- `scripts/smoke_content_ops_faq_search_concurrency.py`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_smoke_content_ops_faq_search_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
## Mechanism
The seed smoke defines the seeded FAQ detail constants once and uses them for
both the `ticket_faq_markdown` insert and the hit-case metadata emitted to the
route case artifact. Hit cases now include expected detail account, target,
target mode, title, and status fields.

The seeded e2e detail-case parser validates those fields on the selected hit
case and forwards them to the existing route contract checker. It no longer
derives target id, title, mode, or status on its own.
## Intentional
- No new module is introduced; the duplicated values live in one producing
  script and are serialized into the existing artifact boundary.
- Miss cases still omit detail expectations because they are never dereferenced.
- The e2e parser fails closed on missing/malformed expected detail fields for
  hit cases.
## Deferred
- If more scripts need the seeded FAQ constants later, a tiny shared helper
  module can be introduced in that future slice.
- The #972 distinct detail-not-run state remains presentation polish.
## Verification
- pytest tests/test_smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q - 62 passed in 0.14s.
- python -m py_compile scripts/smoke_content_ops_faq_search_concurrency.py scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py - Passed.
- git diff --check - Passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-Detail-Case-Metadata.md - Passed.
- bash scripts/local_pr_review.sh - Passed.
## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 48 |
| Seed smoke | 27 |
| E2E runner | 17 |
| Tests | 27 |
| **Total** | **126** |
