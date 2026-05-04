# In-Flight PRs

Last updated: 2026-05-04T10:50Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C3d, in flight) | PR-C3d: Register vendor_classify prompt as a pack | EDIT: `atlas_brain/reasoning/single_pass_prompts/vendor_classify.py` (add module-bottom `register_pack(...)` call against the PR-C3a registry; existing `VENDOR_CLASSIFY_SINGLE_PASS` / `VENDOR_CLASSIFY_PROMPT_VERSION` exports unchanged). NEW: `tests/test_extracted_reasoning_core_pack_registry_vendor_classify.py` (3 atlas-side integration tests: registration on import, owner metadata, list_packs surface). | claude-2026-05-03 | `atlas_brain/reasoning/single_pass_prompts/vendor_classify.py`; `tests/test_extracted_reasoning_core_pack_registry_vendor_classify.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
