# PR: Content Ops Review Source Generation Smoke

## Why this slice exists

The live G2 run proved AI Content Ops can take Atlas review rows through readiness, source-row export, ingestion inspection, and offline campaign draft generation. Today that proof is a manual chain of commands. This slice turns it into one repeatable operator smoke so future sessions can verify "feed it data and see output" without reconstructing the sequence.

This exceeds the 400 LOC soft cap because the operator smoke is only useful as one route through readiness, export, ingestion inspection, and draft validation. Splitting the script, tests, and docs would leave a partial smoke that cannot prove the end-to-end operator contract.

## Scope (this PR)

1. Add a host/operator smoke script that checks review-source readiness, exports quote-grade rows, validates ingestion, and generates offline drafts.
2. Require source-row ingestion warnings to be resolved by default, so anonymous review exports need explicit `--default-field` account/contact bindings.
3. Document the smoke in the Content Ops README, host runbook, and status page.
4. Add tests for the smoke's pass/fail contracts without hitting a live database.

### Files touched

- `scripts/smoke_content_ops_review_source_generation.py`
- `tests/test_smoke_content_ops_review_source_generation.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Review-Source-Generation-Smoke.md`

## Mechanism

- Reuse `scripts/export_content_ops_review_sources.py` functions for readiness and source-row export.
- Reuse `inspect_ingestion_file` for source-row readiness.
- Reuse `generate_campaign_drafts_from_payload` for offline draft generation.
- Fail if generated review-source drafts regress to target-account intent phrasing.

## Intentional

- This does not add a second exporter.
- This does not use a live LLM provider.
- This does not write generated drafts to Postgres.
- This does not make Trustpilot usable without v4 phrase metadata.

## Deferred

- Live provider generation smoke with `EXTRACTED_CAMPAIGN_LLM_*`.
- Postgres import/persistence smoke over exported review rows.
- Real customer export fixture coverage.

## Verification

- Focused smoke tests -> `4 passed`.
- Python compile check for smoke script/tests -> passed.
- Live G2/Slack smoke against local Atlas database -> passed; generated 2 offline drafts from 1 exported review-source row.
- `git diff --check` -> passed.
- `scripts/local_pr_review.sh` -> pending.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Smoke script | 331 |
| Tests | 174 |
| Docs/status | 38 |
| Coordination and plan | 61 |
| Total | 604 |
