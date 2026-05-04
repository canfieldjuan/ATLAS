# In-Flight PRs

Last updated: 2026-05-04T19:30Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C3f, in flight) | PR-C3f: Register remaining synthesis prompts as packs (closes PR-C3 sequence) | EDIT: `atlas_brain/reasoning/single_pass_prompts/category_council_synthesis.py` (add `register_pack` for `category_council_synthesis`). EDIT: `atlas_brain/reasoning/single_pass_prompts/resource_asymmetry_synthesis.py` (add `register_pack` for `resource_asymmetry_synthesis`). NEW: `tests/test_extracted_reasoning_core_pack_registry_remaining_synthesis.py` (5 atlas-side integration tests). The audit's "content/campaign placeholder packs" entry has no atlas-side prompt files; deferred to PR 7 along with the migration of campaign autonomous tasks. | claude-2026-05-03 | `atlas_brain/reasoning/single_pass_prompts/category_council_synthesis.py`; `atlas_brain/reasoning/single_pass_prompts/resource_asymmetry_synthesis.py`; `tests/test_extracted_reasoning_core_pack_registry_remaining_synthesis.py` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
