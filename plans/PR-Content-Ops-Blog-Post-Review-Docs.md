# PR-Content-Ops-Blog-Post-Review-Docs

## Goal

Close the documentation, manifest, and extracted-check wiring for the
`blog_post` generated-asset review path after the repository, export helper, API,
and CLI slices merged.

## Plan

1. Mark the blog post Postgres/export modules and account-scope migration as
   product-owned in `extracted_content_pipeline/manifest.json`.
2. Add the blog post Postgres/export suites to
   `scripts/run_extracted_pipeline_checks.sh`.
3. Update manifest tests so the new product-owned files and migration stay
   tracked.
4. Update status and host runbook docs so generated-asset export/review/API
   guidance lists `blog_post` beside report, landing page, and sales brief.

## Non-Goals

- No runtime behavior changes.
- No new generated-asset routes or CLI flags.
- No competitive-intelligence files.

## Verification

- `pytest tests/test_extracted_campaign_manifest.py`
- `bash scripts/run_extracted_pipeline_checks.sh`
- `bash scripts/check_ascii_python.sh`
- `git diff --check`
