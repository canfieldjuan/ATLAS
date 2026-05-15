# Content Ops Reasoning Upsert Validation

## Why this slice exists

The active Content Ops backlog called out a safety gap in the reasoning context
upsert CLI: it validates row shape, selectors, and context payloads, but it does
not verify those selectors against live customer opportunities before saving.
That is fine for trusted ETL and risky for larger host imports where a typo can
create unreachable or mis-scoped context rows.

## Scope (this PR)

1. Add an opt-in `--validate-opportunities` mode to the reasoning context upsert
   CLI.
2. Validate each prepared context row against active `campaign_opportunities`
   before any write.
3. Let dry-runs use the same validation path without saving context rows.
4. Document the new flag and retire this backlog item.

### Files touched

- `plans/PR-Content-Ops-Reasoning-Upsert-Validation.md`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `scripts/upsert_extracted_campaign_reasoning_contexts.py`
- `tests/test_extracted_campaign_reasoning_upsert_cli.py`

## Mechanism

`--validate-opportunities` checks each prepared row against the opportunity
table before saving. A row passes if any selector matches an active opportunity
by `target_id`, `company_name`, `vendor_name`, or `contact_email` within the
same account and compatible target mode. The check is opt-in so custom hosts
with nonstandard opportunity storage can keep using the existing trusted-ETL
path.

Dry-run with validation opens the database, validates, closes the pool, and
reports `validated_opportunities`; it still does not call `save_context`.

## Intentional

- No default behavior change.
- No new database schema.
- No validation against arbitrary `raw_payload` keys; this slice covers the
  stable opportunity columns used by the generation path.

## Deferred

- Admin UI/API for browsing, editing, and deleting reasoning context rows.
- Custom host validation adapters for nonstandard opportunity storage.
- Batched opportunity validation query if hosts start importing hundreds or
  thousands of reasoning rows at a time.

## Verification

```bash
pytest tests/test_extracted_campaign_reasoning_upsert_cli.py
python -m py_compile scripts/upsert_extracted_campaign_reasoning_contexts.py tests/test_extracted_campaign_reasoning_upsert_cli.py
bash scripts/local_pr_review.sh
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Reasoning-Upsert-Validation.md` | 60 |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | 18 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/README.md` | 4 |
| `extracted_content_pipeline/STATUS.md` | 4 |
| `extracted_content_pipeline/docs/host_install_runbook.md` | 5 |
| `scripts/upsert_extracted_campaign_reasoning_contexts.py` | 90 |
| `tests/test_extracted_campaign_reasoning_upsert_cli.py` | 130 |
| **Total** | **~315** |
