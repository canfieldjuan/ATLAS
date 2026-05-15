# Content Ops Reasoning Upsert Audit Log

## Why this slice exists

The DB reasoning admin workflow can now list/export, dry-run upsert, apply
upserts, and clean stale rows. The remaining operator safety gap is edit
traceability: hosts need a lightweight record of manual reasoning-row changes
without adding a product schema or admin UI.

## Scope (this PR)

1. Add optional `--audit-log` support to the reasoning context upsert CLI.
2. Append one metadata-only JSONL entry per saved row after successful writes.
3. Keep dry-run read-only even when `--audit-log` is supplied.
4. Add focused tests for audit logging and dry-run non-write behavior.
5. Document the audit option in the README and host runbook.
6. Update status and coordination docs.

### Files touched

- `plans/PR-Content-Ops-Reasoning-Upsert-Audit-Log.md`
- `docs/extraction/coordination/inflight.md`
- `scripts/upsert_extracted_campaign_reasoning_contexts.py`
- `tests/test_extracted_campaign_reasoning_upsert_cli.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/STATUS.md`

## Mechanism

`--audit-log` accepts a JSONL path. After each successful `save_context(...)`
call, the CLI appends an entry with action, timestamp, saved context id,
account id, target mode, selectors, row index, and sorted context keys. The
reasoning context payload itself is intentionally not written to the audit log.

## Intentional

- No repository or schema changes.
- No audit write on failed upserts.
- No audit write in dry-run mode.
- No full reasoning payload in the audit file.

## Deferred

- Full admin UI for browsing and editing context rows.
- Database-backed audit table for hosts that need centralized audit storage.
- Bulk validation against live opportunity ids before saving.

## Verification

```bash
pytest tests/test_extracted_campaign_reasoning_upsert_cli.py
python -m py_compile scripts/upsert_extracted_campaign_reasoning_contexts.py tests/test_extracted_campaign_reasoning_upsert_cli.py
bash scripts/local_pr_review.sh
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Reasoning-Upsert-Audit-Log.md` | 60 |
| `docs/extraction/coordination/inflight.md` | 2 |
| `scripts/upsert_extracted_campaign_reasoning_contexts.py` | 45 |
| `tests/test_extracted_campaign_reasoning_upsert_cli.py` | 175 |
| `extracted_content_pipeline/README.md` | 5 |
| `extracted_content_pipeline/docs/host_install_runbook.md` | 5 |
| `extracted_content_pipeline/STATUS.md` | 3 |
| **Total** | **~300** |
