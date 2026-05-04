# In-Flight PRs

Last updated: 2026-05-04T10:12Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C3b, in flight) | PR-C3b: Register battle_card_reasoning prompt as a pack | EDIT: `atlas_brain/reasoning/single_pass_prompts/battle_card_reasoning.py` (add module-bottom `register_pack(...)` call against the PR-C3a registry; existing `BATTLE_CARD_REASONING_PROMPT` / `BATTLE_CARD_REASONING_PROMPT_VERSION` / `VALID_WEDGE_TYPES` exports unchanged). NEW: `tests/test_extracted_reasoning_core_pack_registry_battle_card.py` (4 tests: registration on import, owner metadata, list_packs surface, idempotent re-registration). EDIT: `scripts/run_extracted_pipeline_checks.sh` (wire the new test). The pack file stays atlas-side until PR 7 (Product Migration) moves it to `extracted_competitive_intelligence`. | claude-2026-05-03 | `atlas_brain/reasoning/single_pass_prompts/battle_card_reasoning.py`; `tests/test_extracted_reasoning_core_pack_registry_battle_card.py`; `scripts/run_extracted_pipeline_checks.sh` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
