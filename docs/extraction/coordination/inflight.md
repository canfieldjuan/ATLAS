# In-Flight PRs

Last updated: 2026-05-05T05:16Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-D7a, in flight) | PR-D7a: import-guard + bridge removal (PR 7 first slice) | NEW: `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` (AST + importlib.import_module string scan that fails closed on any `atlas_brain.reasoning.*` import or `import_module("atlas_brain.reasoning...")` call inside an extracted product, regardless of try/except or env-gated wrappers -- core now provides everything reasoning-side, so even gated escape hatches are out of contract). EDIT: `extracted_competitive_intelligence/reasoning/__init__.py` (drop the dead Phase-1 `__getattr__` bridge that lazy-delegated to `atlas_brain.reasoning`; nothing fires it -- submodule imports go through Python native machinery). EDIT: `scripts/run_extracted_pipeline_checks.sh`, `scripts/run_extracted_competitive_intelligence_checks.sh`, `scripts/run_extracted_llm_infrastructure_checks.sh` (wire the new guard for all three products). NEW: `tests/test_forbid_atlas_reasoning_imports.py` (positive + negative scan cases). EDIT: `.github/workflows/extracted_*_checks.yml` (paths-filter + script wiring). First PR-D7 sub-slice -- closes the audit's "no runtime atlas_brain.reasoning imports in extracted products" criterion. PR-D7b/c/d follow with atlas-side fork migration, api-only tightening, and content-pipeline long-form opt-in. | claude-2026-05-03 | `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py`; `extracted_competitive_intelligence/reasoning/__init__.py`; `scripts/run_extracted_pipeline_checks.sh`; `scripts/run_extracted_competitive_intelligence_checks.sh`; `scripts/run_extracted_llm_infrastructure_checks.sh`; `tests/test_forbid_atlas_reasoning_imports.py`; `.github/workflows/extracted_*_checks.yml` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
