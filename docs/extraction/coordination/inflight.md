# In-Flight PRs

Last updated: 2026-05-05T15:05Z by codex-2026-05-05-d12

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| TBD | D12 single-pass campaign reasoning provider | `extracted_content_pipeline/services/single_pass_reasoning_provider.py`, `extracted_content_pipeline/skills/digest/b2b_campaign_reasoning_context.md`, `tests/test_extracted_campaign_single_pass_reasoning.py`, `scripts/run_extracted_pipeline_checks.sh`, content pipeline reasoning docs/status | codex-2026-05-05-d12 | Avoid adding campaign reasoning provider implementations or editing the reasoning handoff docs until this PR lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
