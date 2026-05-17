# Upcoming Queue

Last updated: 2026-05-17T19:50Z by codex-2026-05-17

Sequence reflects dependencies. Claim a slice (set Owner) before starting code so a parallel session does not pick the same one. See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

A-series (cost-closure, `extracted_llm_infrastructure`) is fully merged: PR-A1 #87, PR-A1.5 #107, PR-A2 #89, PR-A3 #92, PR-A4a #95, PR-A4b #106, PR-A4c #98.
**B-series (extracted_quality_gate) is fully merged**: PR-B2 #85 (product-claim core), PR-B3 #114 (safety-gate split), PR-B4a #118 (blog quality pack), PR-B4b #120 (campaign quality pack), PR-B5b #125 (witness specificity pack), PR-B5a #130 (evidence coverage gate), PR-B5c #132 (source-quality pack).

| Slice | Product | Owner | Dependencies | Notes |
|---|---|---|---|---|
| PR-C2-current-state-reset | `extracted_reasoning_core` | codex-2026-05-16 | Latest merged reasoning-core work: #163 | Reconcile current main against the 2026-05-03 reasoning boundary audit. Current main already has public APIs, semantic cache keys/ports, pack registry, graph/state ports, content-pipeline wrappers, atlas wrappers, and import guards. Next code slice should target a remaining gap, not repeat PR-C1/PR-C4 work. |
| PR-C3-enrichment-pack-split | `extracted_reasoning_core` | merged #564 | PR-C2-current-state-reset | Atlas-side per-review enrichment now lives in an explicit product pack. |
| PR-C4-phrase-metadata-utility | `extracted_reasoning_core` | merged #565 | PR-C3-enrichment-pack-split | Pure phrase metadata readers now live in `atlas_brain.reasoning.phrase_metadata`; the task module is a compatibility wrapper. |
| PR-C5-manifest-standalone-smoke | `extracted_reasoning_core` | merged #569 | PR-C4-phrase-metadata-utility | Added reasoning-core manifest ownership, standalone smoke, and shared validation wiring so the product is covered by the same extraction checks as the other extracted packages. |
| PR-C6-node-reason-core | `extracted_reasoning_core` | merged #570 | PR-C5-manifest-standalone-smoke | Promoted the graph reason-node LLM call / parse / fallback contract into core while Atlas keeps its host-specific prompt builder. |
| PR-C7-graph-routing-test-seam | `extracted_reasoning_core` | merged #571 | PR-C6-node-reason-core | Repaired stale Atlas graph routing tests so they exercise the current `AtlasLLMClient` port-adapter seam instead of the old `_llm_generate` helper seam. |
| PR-C8-reflection-port-adapter | `extracted_reasoning_core` | merged #572 | PR-C7-graph-routing-test-seam | Routed reflection LLM analysis through `AtlasLLMClient` + `complete_with_json`, then removed the legacy `_llm_generate` helper from `atlas_brain.reasoning.graph`. |
| PR-C9-graph-boundary-closeout | `extracted_reasoning_core` | merged #573 | PR-C8-reflection-port-adapter | Documented the post-#570/#571/#572 graph boundary so follow-up work does not keep extracting Atlas orchestration wrappers by inertia. |
| PR-D23-landing-reasoning-parity | `extracted_content_pipeline` | merged #566 | Content Ops packaged reasoning runtime | Added `landing_page` to the packaged structured reasoning runtime allowlist so catalog support and execution behavior match. |
| PR-D24-email-campaign-reasoning-parity | `extracted_content_pipeline` | merged #567 | PR-D23-landing-reasoning-parity | Added `email_campaign` to the packaged structured reasoning runtime allowlist so the core campaign output can use packaged multi-pass reasoning. |
