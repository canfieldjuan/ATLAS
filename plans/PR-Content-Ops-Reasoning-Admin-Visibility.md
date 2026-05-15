# PR: Content Ops Reasoning Admin Visibility

## Why this slice exists

Reasoning context admin upsert/delete actions are now exposed by API, but hosts
need a lightweight operator-visible audit hook without a new product table.

## Scope

- Add optional `visibility_provider` injection to `api.reasoning_contexts`.
- Emit metadata-only events for reasoning context upsert and delete.
- Keep events best-effort and omit full reasoning payloads.
- Add focused API tests.

### Files touched

- `plans/PR-Content-Ops-Reasoning-Admin-Visibility.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/api/reasoning_contexts.py`
- `tests/test_extracted_campaign_reasoning_context_api.py`

## Mechanism

The router resolves an optional `VisibilitySink` provider and emits metadata
after successful upsert/delete operations. Failures in the visibility sink do
not break the admin action.

## Intentional

- No full reasoning context payload in visibility events.
- No new database schema.

## Deferred

- Persistent admin audit table.
- Frontend audit-log display.

## Verification

```bash
pytest tests/test_extracted_campaign_reasoning_context_api.py
python -m py_compile extracted_content_pipeline/api/reasoning_contexts.py tests/test_extracted_campaign_reasoning_context_api.py
bash scripts/local_pr_review.sh
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Reasoning-Admin-Visibility.md` | 55 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/README.md` | 4 |
| `extracted_content_pipeline/STATUS.md` | 4 |
| `extracted_content_pipeline/api/reasoning_contexts.py` | 45 |
| `tests/test_extracted_campaign_reasoning_context_api.py` | 55 |
| **Total** | **~170** |
