# In-Flight PRs

Last updated: 2026-05-18T21:55Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| pending | Content Ops review-evidence asset framing | `extracted_content_pipeline/skills/digest/report_generation.md`, `extracted_content_pipeline/skills/digest/sales_brief_generation.md`, `tests/test_extracted_campaign_skill_registry.py`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Content-Ops-Review-Evidence-Asset-Framing.md` | codex-2026-05-18 | Avoid editing report/sales-brief generated-asset prompt framing or packaged skill registry prompt-contract tests |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
