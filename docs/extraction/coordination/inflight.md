# In-Flight PRs

Last updated: 2026-05-03T23:55Z by claude-2026-05-03-b

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1g, in flight) | PR-C1g: Wire `score_archetypes` + `build_temporal_evidence` stubs in api.py | EDIT: `extracted_reasoning_core/api.py` (impl 2 of 3 stubs; `evaluate_evidence` waits for PR-C1d's slim engine). EDIT: `tests/test_extracted_reasoning_core_api.py` (drop the 2 now-implemented stubs from the fail-closed list; add behavioral tests for the wired entry points). | claude-2026-05-03 | `extracted_reasoning_core/api.py`; `tests/test_extracted_reasoning_core_api.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
