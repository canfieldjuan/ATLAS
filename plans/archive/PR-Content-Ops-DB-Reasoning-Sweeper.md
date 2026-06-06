# Content Ops DB Reasoning Sweeper

## Why this slice exists

The DB-backed Content Ops reasoning provider can now read, upsert, and smoke
test durable reasoning contexts, but hosts still need a safe operational way to
remove stale rows after repeated ETL runs. Without a cleanup seam, old global or
mode-specific contexts can accumulate indefinitely.

## Scope (this PR)

1. Add a repository-level stale-context cleanup method to
   `extracted_content_pipeline/campaign_reasoning_postgres.py`.
2. Add `scripts/cleanup_extracted_campaign_reasoning_contexts.py` as the
   host-facing dry-run/apply command.
3. Add focused tests for repository SQL shape and CLI dry-run/apply behavior.
4. Add the new CLI test to `scripts/run_extracted_pipeline_checks.sh`.
5. Claim the slice in `docs/extraction/coordination/inflight.md`.

### Files touched

- `plans/PR-Content-Ops-DB-Reasoning-Sweeper.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/campaign_reasoning_postgres.py`
- `scripts/cleanup_extracted_campaign_reasoning_contexts.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_campaign_reasoning_cleanup_cli.py`
- `tests/test_extracted_campaign_reasoning_postgres.py`

## Mechanism

`PostgresCampaignReasoningContextRepository.delete_stale_contexts(...)` accepts
an explicit `older_than_days` threshold plus optional account and target-mode
filters. It counts matching rows in dry-run mode and deletes only when callers
request apply mode.

The CLI requires `--older-than-days`, defaults to dry-run, and only deletes when
`--apply` is passed. It uses the same DSN resolution as the existing Postgres
reasoning check script: `--database-url`, `EXTRACTED_DATABASE_URL`, or
`DATABASE_URL`.

## Intentional

- No generation/runtime path changes.
- No scheduler is added; hosts decide when to run the sweeper.
- No hidden age threshold default. Operators must provide `--older-than-days`.
- `target_mode=None` means all modes; `--target-mode ""` can still target the
  blank global-fallback mode.

## Deferred

- Admin UI for viewing/editing context rows.
- Scheduled cleanup wiring in Atlas.
- Per-row audit logging for deleted context ids.

## Verification

```bash
pytest tests/test_extracted_campaign_reasoning_postgres.py tests/test_extracted_campaign_reasoning_cleanup_cli.py
python -m py_compile extracted_content_pipeline/campaign_reasoning_postgres.py scripts/cleanup_extracted_campaign_reasoning_contexts.py tests/test_extracted_campaign_reasoning_postgres.py tests/test_extracted_campaign_reasoning_cleanup_cli.py
bash scripts/local_pr_review.sh
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-DB-Reasoning-Sweeper.md` | 65 |
| `docs/extraction/coordination/inflight.md` | 5 |
| `extracted_content_pipeline/campaign_reasoning_postgres.py` | 65 |
| `scripts/cleanup_extracted_campaign_reasoning_contexts.py` | 120 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_campaign_reasoning_cleanup_cli.py` | 80 |
| `tests/test_extracted_campaign_reasoning_postgres.py` | 65 |
| **Total** | **~401** |
