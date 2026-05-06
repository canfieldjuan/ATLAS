# In-Flight PRs

Last updated: 2026-05-06T18:50Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (in flight) | Promote `autonomous/visibility.py` from Phase 1 bridge to manifest-synced mirror (Phase 3 progress, 1/7 active bridges) | EDIT: `extracted_competitive_intelligence/manifest.json` (+ source/target mapping for `autonomous/visibility.py`). REPLACE: `extracted_competitive_intelligence/autonomous/visibility.py` (23-LOC bridge -> 344-LOC mirror copy of atlas_brain peer via `sync_extracted.sh`). atlas_brain count 12->11. | claude-2026-05-03 | `extracted_competitive_intelligence/manifest.json`; `extracted_competitive_intelligence/autonomous/visibility.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
