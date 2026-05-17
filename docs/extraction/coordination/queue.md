# Upcoming Queue

Last updated: 2026-05-17T15:35Z by codex-2026-05-17

Sequence reflects dependencies. Claim a slice (set Owner) before starting code so a parallel session does not pick the same one. See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

A-series (cost-closure, `extracted_llm_infrastructure`) is fully merged: PR-A1 #87, PR-A1.5 #107, PR-A2 #89, PR-A3 #92, PR-A4a #95, PR-A4b #106, PR-A4c #98.
**B-series (extracted_quality_gate) is fully merged**: PR-B2 #85 (product-claim core), PR-B3 #114 (safety-gate split), PR-B4a #118 (blog quality pack), PR-B4b #120 (campaign quality pack), PR-B5b #125 (witness specificity pack), PR-B5a #130 (evidence coverage gate), PR-B5c #132 (source-quality pack).

| Slice | Product | Owner | Dependencies | Notes |
|---|---|---|---|---|
| PR-C2-current-state-reset | `extracted_reasoning_core` | codex-2026-05-16 | Latest merged reasoning-core work: #163 | Reconcile current main against the 2026-05-03 reasoning boundary audit. Current main already has public APIs, semantic cache keys/ports, pack registry, graph/state ports, content-pipeline wrappers, atlas wrappers, and import guards. Next code slice should target a remaining gap, not repeat PR-C1/PR-C4 work. |
| PR-C3-enrichment-pack-split | `extracted_reasoning_core` | codex-2026-05-17 | PR-C2-current-state-reset | Recommended next code slice: carve atlas-side per-review enrichment into an explicit product pack. |
