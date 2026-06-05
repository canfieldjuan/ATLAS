# Content Ops Reasoning Admin Upsert CLI

## Why this slice exists

The DB-backed Content Ops reasoning provider now supports read checks, replay
upserts through the repository, and stale-row cleanup. The remaining operator
gap is a simple way to seed or edit durable reasoning contexts without writing
SQL directly.

This is intentionally one slice even though the estimated diff is slightly over
the 400 LOC target: the CLI, its focused tests, and the two host-facing docs are
the smallest reviewable unit that gives operators a usable edit path without an
undocumented script.

## Scope (this PR)

1. Add a host-facing upsert CLI for `campaign_reasoning_contexts`.
2. Add focused CLI parsing/upsert tests.
3. Add the new test file to the extracted pipeline check runner.
4. Document the admin upsert command in the README and host runbook.
5. Replace the stale merged sweeper coordination row with this slice.

### Files touched

- `plans/PR-Content-Ops-Reasoning-Admin-Upsert-CLI.md`
- `docs/extraction/coordination/inflight.md`
- `scripts/upsert_extracted_campaign_reasoning_contexts.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_campaign_reasoning_upsert_cli.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`

## Mechanism

`scripts/upsert_extracted_campaign_reasoning_contexts.py` reads a JSON object,
array, or wrapper such as `{"contexts": [...]}`. Each row can provide selectors
directly or selector fields such as `target_id`, `company_name`,
`contact_email`, and `vendor_name`. The script calls the existing
`PostgresCampaignReasoningContextRepository.save_context(...)` method so replay
semantics stay centralized in the repository.

## Intentional

- No generation/runtime path changes.
- No new table or migration.
- No web admin UI. This is the smallest useful operator edit path.
- Rows without selectors fail loudly because they would never be readable.

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
| `plans/PR-Content-Ops-Reasoning-Admin-Upsert-CLI.md` | 60 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `scripts/upsert_extracted_campaign_reasoning_contexts.py` | 180 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_campaign_reasoning_upsert_cli.py` | 130 |
| `extracted_content_pipeline/README.md` | 20 |
| `extracted_content_pipeline/docs/host_install_runbook.md` | 15 |
| **Total** | **~410** |
