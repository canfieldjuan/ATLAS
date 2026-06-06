# PR-FAQ-Search-Cleanup-Rowcount-Gate

## Why this slice exists

#971 made the seeded hosted FAQ search e2e report both requested cleanup IDs and
the actual Postgres delete rowcount, but deliberately left rowcount mismatch as
visibility-only. Now that the seeded search-to-detail e2e is the live
correctness path for the FAQ search route, cleanup must fail closed when it
cannot prove every seeded FAQ row was deleted.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening.

1. Make seeded FAQ cleanup fail when the asyncpg `DELETE N` tag is malformed.
2. Make seeded FAQ cleanup fail when `N` differs from the requested unique FAQ
   ID count.
3. Keep the cleanup result compact and explicit: requested count, deleted count,
   raw delete status, and a diagnostic error.
4. Add focused negative fixtures for malformed status and rowcount mismatch.

### Files touched

- `plans/PR-FAQ-Search-Cleanup-Rowcount-Gate.md`
- `scripts/smoke_content_ops_faq_search_seeded_route_e2e.py`
- `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`

## Mechanism

`_cleanup_seeded_faqs(...)` already executes an ID-scoped delete and parses the
asyncpg command tag through `_deleted_row_count(...)`. This slice makes that
parsed count load-bearing:

```python
row_count = _deleted_row_count(delete_status)
ok = row_count == requested_faq_ids
```

Malformed tags produce `deleted_faq_ids: None` and `ok: False`. Mismatched tags
produce `ok: False` with the actual parsed count. The top-level e2e already
includes `cleanup["ok"]` in its final status, so no separate orchestration
change is needed.

## Intentional

- No broader delete scope is added; cleanup still deletes only FAQ IDs emitted
  by the seed smoke cleanup manifest.
- No live hosted route behavior changes; this is an operator/test runner
  hardening slice.
- No retry loop is added. A mismatch means the e2e cannot prove cleanup
  completed, so the result should fail and preserve the diagnostic.
- Local review's cross-layer caller hints were noisy test-local `FakePool`
  matches; `_cleanup_seeded_faqs(...)` is only called by this runner and its
  focused tests.

## Deferred

- Parked hardening: none.
- Cleanup retry/backoff remains deferred until a real transient-delete failure
  is observed; this slice only makes existing proof strict.

## Verification

- `python -m pytest tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q`
  - 47 passed.
- `python -m py_compile scripts/smoke_content_ops_faq_search_seeded_route_e2e.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py`
  - passed.
- `git diff --check` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Search-Cleanup-Rowcount-Gate.md`
  - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `ATLAS_CURRENT_PR_BODY_FILE=/tmp/atlas-faq-search-cleanup-rowcount-gate-pr-body.md bash scripts/local_pr_review.sh`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 83 |
| Seeded E2E runner | 22 |
| Tests | 41 |
| **Total** | **146** |
