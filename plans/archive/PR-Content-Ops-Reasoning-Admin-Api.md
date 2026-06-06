# PR: Content Ops Reasoning Admin API

## Why this slice exists

DB reasoning contexts can be listed and upserted by CLI, but hosts still need
a small mounted API for operator workflows.

## Scope

- Add `api.reasoning_contexts` with list and single-row upsert routes.
- Reuse `PostgresCampaignReasoningContextRepository`.
- Preserve host-owned pool, tenant scope, and auth dependencies.
- Add focused API tests and light docs.

### Files touched

- `plans/PR-Content-Ops-Reasoning-Admin-Api.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/api/reasoning_contexts.py`
- `extracted_content_pipeline/manifest.json`
- `tests/test_extracted_campaign_reasoning_context_api.py`

## Mechanism

The router resolves injected pool/scope providers, calls the repository list
path for browse, and calls the repository save path for create/edit. Request
bodies accept explicit `selectors` or selector fields like `target_id` and
`company_name`; the context payload must live under an explicit `context` key.

## Intentional

- Host tenant scope overrides request-body `account_id`.
- This PR does not add destructive actions.
- Auth stays host-owned; callers pass FastAPI dependencies at mount time.

## Deferred

- Scoped delete/retire contract.
- API CSV export; the existing CLI still owns export.
- Admin audit-log persistence and frontend UI.

## Verification

```bash
pytest tests/test_extracted_campaign_reasoning_context_api.py
python -m py_compile extracted_content_pipeline/api/reasoning_contexts.py tests/test_extracted_campaign_reasoning_context_api.py
bash scripts/local_pr_review.sh
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Reasoning-Admin-Api.md` | 50 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/README.md` | 7 |
| `extracted_content_pipeline/STATUS.md` | 4 |
| `extracted_content_pipeline/api/reasoning_contexts.py` | 190 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `tests/test_extracted_campaign_reasoning_context_api.py` | 150 |
| **Total** | **~400** |
