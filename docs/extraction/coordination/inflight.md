# In-Flight PRs

Last updated: 2026-05-18T21:42Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| pending | Content Ops review-evidence copy framing | `extracted_content_pipeline/campaign_example.py`, `atlas_brain/skills/digest/b2b_campaign_generation.md`, `extracted_content_pipeline/skills/digest/b2b_campaign_generation.md`, `tests/test_extracted_campaign_generation_example.py`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/state.md`, `plans/PR-Content-Ops-Review-Evidence-Copy-Framing.md` | codex-2026-05-18 | Avoid editing the offline campaign example, campaign generation skill copy rules, or source-row campaign example tests |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
