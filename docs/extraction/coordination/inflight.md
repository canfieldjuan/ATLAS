# In-Flight PRs

Last updated: 2026-05-05T14:40Z by codex-2026-05-05-d11

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| TBD | D11 hosted campaign workflow smoke | `tests/test_extracted_campaign_api_hosted_workflow.py`, `scripts/run_extracted_pipeline_checks.sh`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/host_install_runbook.md`, `extracted_content_pipeline/STATUS.md` | codex-2026-05-05-d11 | Avoid editing the hosted campaign API workflow smoke/runbook sequence until this PR lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
