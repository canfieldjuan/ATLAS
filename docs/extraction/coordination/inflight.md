# In-Flight PRs

Last updated: 2026-05-05T18:35Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-D7b3, in flight) | PR-D7b3: evidence_engine.py atlas wrapper (PR 7 final slice -- closes PR-D7 5/5) | EDIT: `atlas_brain/reasoning/evidence_engine.py` (548-LOC fork becomes a ~280-LOC subclass wrapper around `extracted_reasoning_core.evidence_engine.EvidenceEngine`). Subclass pattern (NOT pure re-export like b1/b2/b4/b5) because atlas's six per-review enrichment methods (`compute_urgency`, `override_pain`, `derive_recommend`, `derive_price_complaint`, `derive_budget_authority`, plus the `_check_derivation_rule` helper) stay atlas-side per PR-C1's slim-core split -- `derive_price_complaint` depends on atlas-only `_b2b_phrase_metadata`. Re-exports `ConclusionResult` / `SuppressionResult` from `extracted_reasoning_core.types` (shape-identical to atlas's old local dataclasses). Atlas's factory still consults `settings.b2b_churn.evidence_map_path`. Combines what queue.md called PR-C1e + PR-D7b3 into one atomic PR -- the subclass pattern collapses them safely because callers keep writing `engine.compute_urgency(...)` against a single object. NEW: `tests/test_atlas_reasoning_evidence_engine_aliases.py`. EDIT: `scripts/run_extracted_pipeline_checks.sh` + `.github/workflows/extracted_pipeline_checks.yml`. No caller / test-stub updates required. PR-D7 closes after this with 5/5 forks wrapped. | claude-2026-05-03 | `atlas_brain/reasoning/evidence_engine.py`; `tests/test_atlas_reasoning_evidence_engine_aliases.py`; `scripts/run_extracted_pipeline_checks.sh`; `.github/workflows/extracted_pipeline_checks.yml` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
