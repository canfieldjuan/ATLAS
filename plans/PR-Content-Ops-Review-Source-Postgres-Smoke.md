# PR: Content Ops Review Source Postgres Smoke

## Why this slice exists

PR #595 proved that quote-grade G2 review rows can be exported, inspected, and turned into offline campaign drafts. It stops before the database-backed product loop. The next operator proof is that the same review-source rows can be imported into `campaign_opportunities` and used by the Postgres generation runner to persist drafts.

This exceeds the 400 LOC soft cap because the smoke is only useful as one route through readiness, export, ingestion inspection, import, and DB-backed generation. Splitting the script, tests, and docs would leave a partial smoke that cannot prove the persistence contract.

## Scope (this PR)

1. Add a host/operator smoke script for review-source export -> source-row import -> Postgres draft generation.
2. Require an explicit `--account-id` so repeated smoke runs stay tenant-scoped.
3. Replace matching imported opportunities by default so review rows do not duplicate in `campaign_opportunities`.
4. Document the Postgres smoke in the Content Ops README, host runbook, and status page.
5. Add tests for pass/fail contracts without hitting a live database.

### Files touched

- `scripts/smoke_content_ops_review_source_postgres.py`
- `tests/test_smoke_content_ops_review_source_postgres.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Review-Source-Postgres-Smoke.md`

## Mechanism

- Reuse the review-source exporter functions for readiness and source-row export.
- Reuse `inspect_ingestion_file` and the source-row adapter before import.
- Reuse `import_campaign_opportunities` with scoped replace-existing semantics.
- Reuse `generate_campaign_drafts_from_postgres` with deterministic offline LLM and packaged skills so the smoke does not need provider credentials.
- Fail if generated review-source drafts regress to target-account intent phrasing.

## Intentional

- This does not add a new importer or generator.
- This does not use a live LLM provider.
- This does not delete generated `b2b_campaigns` rows; the smoke reports saved ids for operator cleanup/review.
- This does not make Trustpilot usable without v4 phrase metadata.

## Deferred

- A cleanup command for smoke-generated campaign drafts.
- Live provider generation over imported review-source rows.
- Hosted API route smoke for the same review-source import/generation loop.

## Verification

- Focused Postgres smoke tests -> `7 passed`.
- Python compile check for smoke script/tests -> passed.
- Live G2/Slack Postgres smoke -> reached readiness/export/ingestion, then failed cleanly with `UndefinedTableError: relation "campaign_opportunities" does not exist`; host database needs extracted migrations before the smoke can pass live.
- `git diff --check` -> passed.
- `scripts/local_pr_review.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Smoke script | 480 |
| Tests | 381 |
| Docs/status | 36 |
| Coordination and plan | 72 |
| Total | 969 |
