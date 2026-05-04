# Upcoming Queue

Last updated: 2026-05-04T04:39Z by codex-2026-05-03

Sequence reflects dependencies. Claim a slice (set Owner) before starting code so a parallel session does not pick the same one. See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

A-series (cost-closure, `extracted_llm_infrastructure`) is fully merged: PR-A1 #87, PR-A1.5 #107, PR-A2 #89, PR-A3 #92, PR-A4a #95, PR-A4b #106, PR-A4c #98.
B-series progress: PR-B2 #85 (product-claim core), PR-B3 #114 (safety-gate split), PR-B4a #118 (blog quality pack), PR-B4b #120 (campaign quality pack), PR-B5b #125 (witness specificity pack), PR-B5a #130 (evidence coverage gate). PR-B5c (source-quality pack) remains.

| Slice | Product | Owner | Dependencies | Notes |
|---|---|---|---|---|
| PR-B5c | `extracted_quality_gate` | claude-2026-05-03-b | PR-B5b / #125 (merged) | Source-quality pack. Composes `source_impact.py` + `witness_render_gate.py`. Async DB + settings dependency for source capability registry. ~500 LOC including tests. |
| PR-C1 | `extracted_reasoning_core` | claude-2026-05-03 | PR #80, PR #82 (both merged) | Consolidate evidence/temporal/archetypes per merged PR #82 audit. NEW in core: `archetypes.py`, `evidence_engine.py` (slim conclusions+suppression surface), `evidence_map.yaml`, `temporal.py` (with `_numeric_value` / `_row_get` helpers + parameterized `MIN_DAYS_FOR_PERCENTILES`). Atlas-side: NEW `atlas_brain/reasoning/review_enrichment.py`; slim `atlas_brain/reasoning/evidence_engine.py`. Convert `extracted_content_pipeline/reasoning/{archetypes,evidence_engine,temporal}.py` to re-export wrappers. EDIT `extracted_reasoning_core/api.py` (impl 3 stubs) and `extracted_reasoning_core/types.py` (rich `TemporalEvidence` + 4 sub-types + `ConclusionResult` + `SuppressionResult`). Rename + redirect `tests/test_extracted_reasoning_*.py`. PR #79 contract amendment lands in the same commit. |
