# In-Flight PRs

Last updated: 2026-05-05T13:25Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-D7b2, in flight) | PR-D7b2: wedge_registry.py atlas wrapper (PR 7 third slice, 2/5 of fork migration) | EDIT: `atlas_brain/reasoning/wedge_registry.py` (159-LOC fork becomes a ~30-LOC re-export from `extracted_reasoning_core.wedge_registry`; preserves the eight public symbols Wedge / WedgeMeta / WEDGE_ENUM_VALUES / wedge_from_archetype / validate_wedge / get_wedge_meta / get_sales_motion / get_required_pools). NEW: `tests/test_atlas_reasoning_wedge_registry_aliases.py` (alias-identity pin mirroring PR-D7b1's tiers test). EDIT: `scripts/run_extracted_pipeline_checks.sh` + `.github/workflows/extracted_pipeline_checks.yml` (wire new test + atlas wedge_registry.py path). Drift analysis: cosmetic only -- atlas's richer module docstring preserved in wrapper, core gained `__all__` (more disciplined), no behavioral diffs across the 8 symbols. Production callers verified: `_b2b_synthesis_validation.py`, `_b2b_reasoning_contracts.py`, `_b2b_synthesis_reader.py`, `b2b_blog_post_generation.py` + `tests/test_reasoning_synthesis_v2.py`. PR-D7b3-b5 follow with evidence_engine / temporal / archetypes wrappers. | claude-2026-05-03 | `atlas_brain/reasoning/wedge_registry.py`; `tests/test_atlas_reasoning_wedge_registry_aliases.py`; `scripts/run_extracted_pipeline_checks.sh`; `.github/workflows/extracted_pipeline_checks.yml` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
