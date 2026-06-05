# PR: Content Ops Reasoning Context Delete

## Why this slice exists

The reasoning admin API can list and upsert rows, but operators still need a
deliberate scoped way to remove bad or stale rows.

## Scope

- Add a scoped `delete_context` repository method.
- Expose `DELETE /campaign-reasoning-contexts/{id}`.
- Require an account scope for destructive deletes.
- Add focused repository and API tests.

### Files touched

- `plans/PR-Content-Ops-Reasoning-Context-Delete.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/api/reasoning_contexts.py`
- `extracted_content_pipeline/campaign_reasoning_postgres.py`
- `tests/test_extracted_campaign_reasoning_context_api.py`
- `tests/test_extracted_campaign_reasoning_postgres.py`

## Mechanism

The repository deletes by `id` and `account_id` only. The API uses the host
tenant scope when present, otherwise it requires `account_id` as a query
parameter.

## Intentional

- No unscoped delete path.
- No bulk delete.
- No audit-table persistence in this slice.

## Deferred

- Admin audit-log persistence.
- Frontend delete/retire UX.

## Verification

```bash
pytest tests/test_extracted_campaign_reasoning_context_api.py tests/test_extracted_campaign_reasoning_postgres.py
python -m py_compile extracted_content_pipeline/api/reasoning_contexts.py extracted_content_pipeline/campaign_reasoning_postgres.py tests/test_extracted_campaign_reasoning_context_api.py tests/test_extracted_campaign_reasoning_postgres.py
bash scripts/local_pr_review.sh
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Reasoning-Context-Delete.md` | 55 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/README.md` | 4 |
| `extracted_content_pipeline/STATUS.md` | 4 |
| `extracted_content_pipeline/api/reasoning_contexts.py` | 35 |
| `extracted_content_pipeline/campaign_reasoning_postgres.py` | 30 |
| `tests/test_extracted_campaign_reasoning_context_api.py` | 45 |
| `tests/test_extracted_campaign_reasoning_postgres.py` | 55 |
| **Total** | **~230** |
