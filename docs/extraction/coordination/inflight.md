# In-Flight PRs

Last updated: 2026-05-04T19:38Z by codex-2026-05-04

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #170 | Own competitive-set planner support ports | EDIT: `extracted_competitive_intelligence/services/b2b_competitive_sets.py`; `extracted_competitive_intelligence/manifest.json`; `extracted_competitive_intelligence/README.md`; `extracted_competitive_intelligence/STATUS.md`; `scripts/run_extracted_competitive_intelligence_checks.sh`; `scripts/smoke_extracted_competitive_intelligence_standalone.py`; `tests/test_extracted_competitive_manifest.py`. NEW: `extracted_competitive_intelligence/services/b2b/competitive_set_ports.py`; `tests/test_extracted_competitive_sets.py`. | codex-2026-05-04 | Competitive Intelligence extraction surface; avoid these files until PR lands. |
| (PR-C3e, in flight) | PR-C3e: Register reasoning_synthesis prompt as a pack | EDIT: `atlas_brain/reasoning/single_pass_prompts/reasoning_synthesis.py` (add module-bottom `register_pack(...)` call against the PR-C3a registry; existing `REASONING_SYNTHESIS_PROMPT` / `REASONING_SYNTHESIS_PROMPT_VERSION` exports unchanged). NEW: `tests/test_extracted_reasoning_core_pack_registry_reasoning_synthesis.py` (3 atlas-side integration tests: registration on import, owner metadata + valid_wedges, list_packs surface). Atlas-side only -- not in any extracted mirror. | claude-2026-05-03 | `atlas_brain/reasoning/single_pass_prompts/reasoning_synthesis.py`; `tests/test_extracted_reasoning_core_pack_registry_reasoning_synthesis.py` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
