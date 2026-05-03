# Upcoming Queue

Last updated: 2026-05-04T00:51Z by codex-2026-05-03

Sequence reflects dependencies. Claim a slice (set Owner) before starting code so a parallel session does not pick the same one. See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

A-series (cost-closure, `extracted_llm_infrastructure`) is fully merged: PR-A1 #87, PR-A1.5 #107, PR-A2 #89, PR-A3 #92, PR-A4a #95, PR-A4b #106, PR-A4c #98.
B-series progress: PR-B2 #85 (product-claim core), PR-B3 #114 (safety-gate split). PR-B4 / PR-B5 remain.

| Slice | Product | Owner | Dependencies | Notes |
|---|---|---|---|---|
| PR-B4 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | Blog + campaign quality packs over the core gate contract. |
| PR-B5 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | B2B evidence + witness + source-quality packs. |
| PR-C1 | `extracted_reasoning_core` | claude-2026-05-03 | PR #80, PR #82 (both merged) | Consolidate evidence/temporal/archetypes per merged PR #82 audit. NEW in core: `archetypes.py`, `evidence_engine.py` (slim conclusions+suppression surface), `evidence_map.yaml`, `temporal.py` (with `_numeric_value` / `_row_get` helpers + parameterized `MIN_DAYS_FOR_PERCENTILES`). Atlas-side: NEW `atlas_brain/reasoning/review_enrichment.py`; slim `atlas_brain/reasoning/evidence_engine.py`. Convert `extracted_content_pipeline/reasoning/{archetypes,evidence_engine,temporal}.py` to re-export wrappers. EDIT `extracted_reasoning_core/api.py` (impl 3 stubs) and `extracted_reasoning_core/types.py` (rich `TemporalEvidence` + 4 sub-types + `ConclusionResult` + `SuppressionResult`). Rename + redirect `tests/test_extracted_reasoning_*.py`. PR #79 contract amendment lands in the same commit. |
| PR-D8 | `extracted_content_pipeline` | codex-2026-05-03 | PR-D7 / #112 (merged); avoid PR-C1 files | Add product-owned draft review/export path so hosts can inspect generated `b2b_campaigns` rows without handwritten SQL. |
