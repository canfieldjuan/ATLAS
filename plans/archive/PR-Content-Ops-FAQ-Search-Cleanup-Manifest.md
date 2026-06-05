# PR-Content-Ops-FAQ-Search-Cleanup-Manifest

## Why this slice exists
#967 proved the one-command seeded hosted FAQ search path, but cleanup still
inferred seeded FAQ IDs from hit-case route expectations. That is narrower than
the seed data and can miss rows if a future seeded corpus lacks a hit case.
## Scope (this PR)
Ownership lane: content-ops/faq-search
Slice phase: Production hardening.
1. Have the DB seed smoke emit a cleanup manifest with every seeded FAQ ID.
2. Write that manifest before DB inserts so partial seed failures still expose
   cleanup IDs.
3. Have the e2e runner clean from the manifest instead of route-case hits.
4. Add negative fixtures for malformed cleanup manifests.
### Files touched
- `plans/PR-Content-Ops-FAQ-Search-Cleanup-Manifest.md`
- `scripts/smoke_content_ops_faq_search_concurrency.py`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_smoke_content_ops_faq_search_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
## Mechanism
The DB seed smoke accepts `--cleanup-manifest-output` and writes a compact JSON
object with all generated account, corpus, and FAQ IDs immediately after cases
are built. The e2e runner passes that path into the seed smoke and reads FAQ IDs
from the manifest for teardown.

Manifest parsing fails closed on missing JSON object shape, malformed `faq_ids`,
or non-string IDs. Cleanup still deletes only explicit FAQ IDs and relies on the
existing foreign-key cascade for search rows.
## Intentional
- No broad delete by account or corpus; cleanup remains ID-scoped.
- No CI enrollment change; this updates already-enrolled smoke tests.
## Deferred
- Hosted detail-route expectation checks.
- A live validation doc for running the seeded e2e against the hosted route.
## Verification
- pytest tests/test_smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q - 31 passed in 0.12s.
- python -m py_compile scripts/smoke_content_ops_faq_search_concurrency.py scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py - Passed.
- git diff --check - Passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-Cleanup-Manifest.md - Passed.
- bash scripts/local_pr_review.sh - Passed.
## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 49 |
| DB seed smoke | 23 |
| E2E runner | 54 |
| Tests | 136 |
| **Total** | **262** |
