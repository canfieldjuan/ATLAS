# In-Flight PRs

Last updated: 2026-05-17T16:34Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #565 | Promote phrase metadata helpers to reasoning utility | `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/queue.md`, `plans/PR-Reasoning-Phrase-Metadata-Utility.md`, `atlas_brain/reasoning/phrase_metadata.py`, `atlas_brain/reasoning/review_enrichment.py`, `atlas_brain/autonomous/tasks/_b2b_phrase_metadata.py`, `atlas_brain/services/b2b/enrichment_contract.py`, `tests/test_b2b_phrase_metadata.py` | codex-2026-05-17 | Avoid moving phrase metadata helpers or changing review enrichment phrase gating while this promotes the canonical helper import path. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
