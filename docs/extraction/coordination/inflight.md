# In-Flight PRs

Last updated: 2026-05-05T13:48Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-D7b4, in flight) | PR-D7b4: temporal.py atlas wrapper (PR 7 fourth slice, 3/5 of fork migration -- evidence_engine deferred per PR-D7b3 drift analysis) | EDIT: `atlas_brain/reasoning/temporal.py` (490-LOC fork becomes a ~40-LOC re-export from `extracted_reasoning_core.temporal` + `extracted_reasoning_core.types`; preserves the 12 public symbols TemporalEngine + 5 dataclasses + 5 constants). NEW: `tests/test_atlas_reasoning_temporal_aliases.py` (alias-identity pin mirroring PR-D7b1/b2). EDIT: `scripts/run_extracted_pipeline_checks.sh` + `.github/workflows/extracted_pipeline_checks.yml` (wire test + atlas temporal.py path). Drift analysis: dataclasses (TemporalEvidence + 4 sub-types) moved to core.types in PR-C1c (frozen+slots), no atlas caller mutates them; core's __init__ adds backward-compatible `min_days_for_percentiles` kwarg; MIN_DAYS_FOR_PERCENTILES canonicalized to atlas's actual runtime value (3, was constant=7 / hardcoded gate=3). Atlas callers verified: b2b_churn_intelligence + _b2b_shared + market_pulse + tests. PR-D7b3 (evidence_engine.py) DEFERRED -- atlas has 6 methods absent from core (the per-review enrichment surface PR-C1e was supposed to carve out into review_enrichment.py but didn't); needs PR-C1e backfill before wrapper. PR-D7b5 (archetypes.py) follows. | claude-2026-05-03 | `atlas_brain/reasoning/temporal.py`; `tests/test_atlas_reasoning_temporal_aliases.py`; `scripts/run_extracted_pipeline_checks.sh`; `.github/workflows/extracted_pipeline_checks.yml` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
