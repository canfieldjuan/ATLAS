# In-Flight PRs

Last updated: 2026-05-04T10:30Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C3c, in flight) | PR-C3c: Register cross-vendor battle prompts as packs | EDIT: `atlas_brain/reasoning/single_pass_prompts/cross_vendor_battle.py` (add `register_pack` for `cross_vendor_battle_single_pass`). EDIT: `atlas_brain/reasoning/single_pass_prompts/cross_vendor_battle_synthesis.py` (add `register_pack` for `cross_vendor_battle_synthesis`). EDIT: `extracted_competitive_intelligence/reasoning/single_pass_prompts/cross_vendor_battle.py` (parallel registration -- file is owned by the package per manifest, not synced from atlas; both calls idempotent against the registry). NEW: `tests/test_extracted_reasoning_core_pack_registry_cross_vendor_battle.py` (7 atlas-side integration tests). | claude-2026-05-03 | `atlas_brain/reasoning/single_pass_prompts/cross_vendor_battle.py`; `atlas_brain/reasoning/single_pass_prompts/cross_vendor_battle_synthesis.py`; `extracted_competitive_intelligence/reasoning/single_pass_prompts/cross_vendor_battle.py`; `tests/test_extracted_reasoning_core_pack_registry_cross_vendor_battle.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
