# PR-Content-Ops-Export-API-Reasoning-Fields

## Why this slice exists

PR #429 added generation-usage and reasoning summary fields to the shared
campaign draft export helper. The host-mounted B2B and seller FastAPI export
routes delegate to that helper, but their route tests still only assert generic
CSV/JSON shape. This slice locks the API surfaces to the same export contract.

## Scope (this PR)

1. Add B2B router assertions that JSON and CSV exports include derived
   generation/reasoning summary fields.
2. Add seller router assertions that CSV exports include those same fields.
3. Clarify host docs that the mounted API export routes expose the same
   summary fields as the CLI.

### Files touched

- `tests/test_extracted_campaign_api_b2b_campaigns.py`
- `tests/test_extracted_campaign_api_seller_campaigns.py`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

No production code changes. The existing routers call `list_campaign_drafts()`;
these tests seed rows with metadata containing `generation_usage`,
`generation_parse_attempts`, and `reasoning_context`, then assert the router
responses expose the derived fields added by the shared export helper.

## Intentional

- This is test/docs only because the runtime path already delegates to the
  shared helper.
- The seller route is CSV-only in this slice because that route's existing
  export test covers CSV; JSON behavior is already covered through the shared
  B2B/export helper path.

## Deferred

- Asset exports outside campaign drafts remain separate work.

## Verification

- `python -m pytest tests/test_extracted_campaign_api_b2b_campaigns.py tests/test_extracted_campaign_api_seller_campaigns.py` (53 passed)
- `bash scripts/run_extracted_pipeline_checks.sh` (1374 passed, 1 existing torch/pynvml warning)
- `git diff --check`
- Non-ASCII byte check for edited Python files (clean)

## Estimated diff size

About 4 files, under 100 changed lines.
