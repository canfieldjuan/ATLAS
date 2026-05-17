# In-Flight PRs

Last updated: 2026-05-17T15:35Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #564 | Split atlas review enrichment pack | `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/queue.md`, `plans/PR-Reasoning-Enrichment-Pack-Split.md`, `atlas_brain/reasoning/evidence_engine.py`, `atlas_brain/reasoning/review_enrichment.py`, `extracted_reasoning_core/evidence_engine.py`, `tests/test_atlas_reasoning_evidence_engine_aliases.py` | codex-2026-05-17 | Avoid editing the atlas-side evidence enrichment split while this moves the per-review methods into an explicit product pack module. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
