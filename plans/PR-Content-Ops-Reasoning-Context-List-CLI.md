# Content Ops Reasoning Context List CLI

## Why this slice exists

The DB-backed Content Ops reasoning workflow can now check one target, upsert
operator-provided contexts, and sweep stale rows. Operators still need a
read-only inventory/export path before and after those edits so they can audit
stored context rows without hand-writing SQL.

## Scope (this PR)

1. Add a repository list/export helper for `campaign_reasoning_contexts`.
2. Add a host-facing list CLI with JSON and CSV output.
3. Add focused repository/CLI tests.
4. Add the new test file to the extracted pipeline check runner.
5. Document the list/export command in the README and host runbook.
6. Replace the stale merged upsert coordination row with this slice.

### Files touched

- `plans/PR-Content-Ops-Reasoning-Context-List-CLI.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/campaign_reasoning_postgres.py`
- `scripts/list_extracted_campaign_reasoning_contexts.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_campaign_reasoning_list_cli.py`
- `tests/test_extracted_campaign_reasoning_postgres.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/STATUS.md`

## Mechanism

`PostgresCampaignReasoningContextRepository.list_contexts(...)` reads rows from
the configured reasoning context table with optional account, target-mode, and
selector filters. It returns a small result object that renders JSON or CSV.
The CLI wires that helper to `EXTRACTED_DATABASE_URL` / `DATABASE_URL`.

## Intentional

- Read-only only; no generation/runtime behavior changes.
- No new table or migration.
- No web admin UI. This is the operator inventory companion to the existing
  check, upsert, and cleanup CLIs.

## Deferred

- Full admin UI for browsing/editing reasoning rows.
- Row-level audit logging around admin edits.
- Live opportunity-id validation before upserts.

## Verification

```bash
pytest tests/test_extracted_campaign_reasoning_list_cli.py
python -m py_compile extracted_content_pipeline/campaign_reasoning_postgres.py scripts/list_extracted_campaign_reasoning_contexts.py tests/test_extracted_campaign_reasoning_list_cli.py
bash scripts/local_pr_review.sh
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `plans/PR-Content-Ops-Reasoning-Context-List-CLI.md` | 55 |
| `docs/extraction/coordination/inflight.md` | 2 |
| `extracted_content_pipeline/campaign_reasoning_postgres.py` | 85 |
| `scripts/list_extracted_campaign_reasoning_contexts.py` | 120 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_campaign_reasoning_list_cli.py` | 150 |
| `tests/test_extracted_campaign_reasoning_postgres.py` | 5 |
| `extracted_content_pipeline/README.md` | 15 |
| `extracted_content_pipeline/docs/host_install_runbook.md` | 15 |
| `extracted_content_pipeline/STATUS.md` | 5 |
| **Total** | **~448** |

This is intentionally one slice even though the estimate is above the 400 LOC
target: repository helper, CLI, tests, and host-facing docs are the smallest
usable read-only inventory feature.
