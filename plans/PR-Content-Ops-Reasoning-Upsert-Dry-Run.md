# Content Ops Reasoning Upsert Dry Run

## Why this slice exists

The DB-backed reasoning context admin path now has list/export, upsert, and
cleanup CLIs. The upsert CLI still writes immediately after parsing. Operators
need the same safe preflight behavior the other host data-loading scripts
provide before they edit durable reasoning rows.

## Scope (this PR)

1. Add `--dry-run` to `scripts/upsert_extracted_campaign_reasoning_contexts.py`.
2. Reuse the same row validation path for dry-run and real upsert.
3. Skip database pool creation when `--dry-run` is set.
4. Add focused tests for dry-run validation and CLI behavior.
5. Document the safer default operator flow in the README and host runbook.
6. Replace the stale merged list/export coordination row with this slice.

### Files touched

- `plans/PR-Content-Ops-Reasoning-Upsert-Dry-Run.md`
- `docs/extraction/coordination/inflight.md`
- `scripts/upsert_extracted_campaign_reasoning_contexts.py`
- `tests/test_extracted_campaign_reasoning_upsert_cli.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/STATUS.md`

## Mechanism

The CLI factors the existing pre-write validation into `_prepare_contexts(...)`.
Dry-run calls that helper and returns `{"status": "dry_run", "would_upsert": N}`
without opening a database pool. Real upsert calls the same helper before
writing, preserving the no-partial-write validation added in the previous
upsert slice.

## Intentional

- No repository or schema changes.
- No change to real upsert semantics.
- No hidden database connection in dry-run mode.

## Deferred

- Full admin UI for browsing and editing context rows.
- Row-level audit logging around admin edits.
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
| `plans/PR-Content-Ops-Reasoning-Upsert-Dry-Run.md` | 60 |
| `docs/extraction/coordination/inflight.md` | 2 |
| `scripts/upsert_extracted_campaign_reasoning_contexts.py` | 55 |
| `tests/test_extracted_campaign_reasoning_upsert_cli.py` | 65 |
| `extracted_content_pipeline/README.md` | 5 |
| `extracted_content_pipeline/docs/host_install_runbook.md` | 5 |
| `extracted_content_pipeline/STATUS.md` | 3 |
| **Total** | **~195** |
