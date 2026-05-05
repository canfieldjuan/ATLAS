# In-Flight PRs

Last updated: 2026-05-05T14:56Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-D7b5, in flight) | PR-D7b5: archetypes.py atlas wrapper (PR 7 fifth slice, 4/5 of fork migration -- evidence_engine still deferred) | EDIT: `atlas_brain/reasoning/archetypes.py` (592-LOC fork becomes a ~50-LOC re-export from `extracted_reasoning_core.archetypes`; preserves ARCHETYPES + ArchetypeProfile + SignalRule + MATCH_THRESHOLD + score_evidence + best_match + top_matches + get_archetype + get_falsification_conditions + enrich_evidence_with_archetypes). Atlas's `ArchetypeMatch` aliases core's `_ArchetypeMatchInternal` -- core deliberately preserved atlas's field names (archetype / score / matched_signals / missing_signals / risk_level) on the rich internal type per PR-C1a's design ("Atlas-side callers continue to consume this richer shape directly"). The canonical public `ArchetypeMatch` in `core.types` (with archetype_id / label / evidence_hits / etc.) stays separate -- only external products through `core.api.score_archetypes` see that shape. NEW: `tests/test_atlas_reasoning_archetypes_aliases.py` (alias-identity pin mirroring PR-D7b1/b2/b4). EDIT: `scripts/run_extracted_pipeline_checks.sh` + `.github/workflows/extracted_pipeline_checks.yml`. No test or caller updates required: atlas's existing field reads (m.archetype, m.matched_signals, m.risk_level in test_reasoning_live.py) keep working because core's _ArchetypeMatchInternal has the same field names. PR-D7 closes after this with 4/5 forks wrapped; b3/evidence_engine remains deferred behind PR-C1e. | claude-2026-05-03 | `atlas_brain/reasoning/archetypes.py`; `tests/test_atlas_reasoning_archetypes_aliases.py`; `scripts/run_extracted_pipeline_checks.sh`; `.github/workflows/extracted_pipeline_checks.yml` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
