# In-Flight PRs

Last updated: 2026-05-05T05:56Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-D7b1, in flight) | PR-D7b1: tiers.py atlas wrapper (PR 7 second slice, 1/5 of fork migration) | EDIT: `atlas_brain/reasoning/tiers.py` (190-LOC fork becomes a ~25-LOC re-export from `extracted_reasoning_core.tiers`; preserves the six public symbols Tier / TierConfig / TIER_CONFIGS / get_tier_config / build_tiered_pattern_sig / needs_refresh / gather_tier_context). NEW: `tests/test_atlas_reasoning_tiers_aliases.py` (alias-identity pin similar to test_atlas_reasoning_graph_aliases.py). EDIT: `scripts/run_extracted_pipeline_checks.sh` + `.github/workflows/extracted_pipeline_checks.yml` (wire new test + atlas tiers.py path). Behavior preserved per drift analysis: core's TierConfig(frozen=True) and inherits_from=tuple are stricter than atlas's; no atlas caller mutates either; debug-log loses one detail string acceptable; logger namespace migrates to extracted_reasoning_core.tiers (correct — code is core's). PR-D7b2-b5 follow with wedge_registry / evidence_engine / temporal / archetypes wrappers. | claude-2026-05-03 | `atlas_brain/reasoning/tiers.py`; `tests/test_atlas_reasoning_tiers_aliases.py`; `scripts/run_extracted_pipeline_checks.sh`; `.github/workflows/extracted_pipeline_checks.yml` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
