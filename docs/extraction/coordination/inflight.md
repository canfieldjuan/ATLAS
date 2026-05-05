# In-Flight PRs

Last updated: 2026-05-05T15:11Z by codex-2026-05-05-d13

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| TBD | D13 single-pass reasoning CLI wiring | `scripts/run_extracted_campaign_generation_example.py`, `scripts/run_extracted_campaign_generation_postgres.py`, campaign generation CLI tests, content pipeline README/runbook/status | codex-2026-05-05-d13 | Avoid adding host-facing campaign reasoning CLI flags until this PR lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
