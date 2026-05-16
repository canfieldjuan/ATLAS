# In-Flight PRs

Last updated: 2026-05-16T00:15Z by codex-2026-05-15

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| draft | Content Ops file reasoning target-mode parity | `extracted_content_pipeline/campaign_reasoning_data.py`, `tests/test_extracted_campaign_reasoning_data.py`, `extracted_content_pipeline/README.md`, `extracted_content_pipeline/docs/reasoning_handoff_contract.md`, `docs/extraction/coordination/inflight.md`, `plans/PR-Content-Ops-File-Reasoning-Target-Mode.md` | codex-2026-05-15 | Avoid file-backed reasoning provider edits |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
