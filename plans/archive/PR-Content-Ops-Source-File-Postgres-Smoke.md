# PR-Content-Ops-Source-File-Postgres-Smoke

## Why this slice exists

PR-Content-Ops-Support-Ticket-Examples added packaged support-ticket CSV and
JSON examples and proved they work through offline conversion/generation. The
deferred gap is the Postgres-backed path: hosts still need one command that
imports a source-row file into `campaign_opportunities`, generates offline
drafts through the DB runner, and verifies persisted campaign target metadata.

This slice adds that host-facing source-file Postgres smoke using the packaged
support-ticket CSV as the default input.

## Scope (this PR)

1. Add a source-file Postgres smoke command that accepts JSON, JSONL, or CSV
   source rows.
2. Reuse the shared source Postgres smoke helpers for schema readiness,
   generation, saved-draft fetch, and target-id validation.
3. Add tests for the packaged support-ticket CSV path, schema fail-closed
   behavior, and persisted-draft validation.
4. Document the new command in README and the host install runbook.
5. Remove the merged support-ticket examples coordination row while claiming
   this slice.

### Files touched

- `scripts/smoke_content_ops_source_file_postgres.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_smoke_content_ops_source_file_postgres.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Source-File-Postgres-Smoke.md`

## Mechanism

The new smoke does:

```text
source file -> inspect_ingestion_file -> load_source_campaign_opportunities_from_file
-> import_campaign_opportunities -> generate_imported_target_drafts
-> fetch_saved_drafts -> target-id + draft-shape validation
```

It defaults to `extracted_content_pipeline/examples/support_ticket_sources.csv`
with `--source-format csv`, but callers can point it at any source-row JSON,
JSONL, or CSV file.

## Intentional

- This is a source-file smoke, not a Zendesk/Intercom/Freshdesk connector.
- The command keeps deterministic/offline draft generation; no provider call is
  introduced.
- The script reuses `content_ops_source_postgres_smoke_helpers.py` instead of
  copying the DB readiness and persisted-draft checks again.

## Deferred

- Live help desk API connectors remain future slices.
- Live DB execution is still operator-run only; CI uses fake pools.

## Verification

- `pytest tests/test_smoke_content_ops_source_file_postgres.py -q` -> 4 passed.
- `python -m py_compile scripts/smoke_content_ops_source_file_postgres.py tests/test_smoke_content_ops_source_file_postgres.py` -> passed.
- `grep -nP '[^\x00-\x7F]' scripts/smoke_content_ops_source_file_postgres.py tests/test_smoke_content_ops_source_file_postgres.py` -> no matches.
- `rg -n "TODO|FIXME|pass$|campaign_opportunities|b2b_campaigns|EXTRACTED_DATABASE_URL|appears to be weighing|acct_123|customer_support_tickets" scripts/smoke_content_ops_source_file_postgres.py tests/test_smoke_content_ops_source_file_postgres.py extracted_content_pipeline/README.md extracted_content_pipeline/docs/host_install_runbook.md` -> only expected table/env/default/fixture/doc references and no TODO/FIXME/pass placeholders.
- `git diff --check` -> passed.
- `bash scripts/local_pr_review.sh` -> passed.
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1440 passed, 1 existing torch/pynvml warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Smoke script | ~290 |
| Tests | ~220 |
| Docs + runner + plan + coordination | ~120 |
| **Total** | **~630** |

This exceeds the soft 400 LOC budget because the slice adds a host-facing CLI
and its fake-pool integration tests together. Splitting tests from the command
would leave the new Postgres smoke under-reviewed.
