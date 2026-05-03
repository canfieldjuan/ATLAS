# Decisions Log

Last updated: 2026-05-03T20:03Z by claude-2026-05-03

Append-only chronological log. Never edit historical entries; supersede with a newer entry instead. See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

- **2026-05-01** — Folders stay siblings of `atlas_brain/`, not relocated under `extracted/`. Only `extracted/_shared/` lives in the umbrella. Path moves would touch hundreds of references in manifests, READMEs, and CI; not worth the disruption.
- **2026-05-01** — Wrapper-script pattern for shared tooling rollout: keep existing entry-point script names as thin wrappers that delegate to `extracted/_shared/scripts/`. Preserves CI references. Settled by PRs #48-50.
- **2026-05-03** — Reasoning is its own extracted product (`extracted_reasoning_core`), not a leaf duplicated into each consumer. Boundary doc + skeleton + compat-wrapper migration. Settled by PRs #79, #80.
- **2026-05-03** — Cost-closure additions (`llm_exact_cache.py`, `provider_cost_sync.py`, migrations 251 + 258, plus new code: cache-savings, drift report, budget gate, OpenAI adapter) go INTO `extracted_llm_infrastructure`. No separate `llm-spend-py` package.
- **2026-05-03** — `docs/extraction/COORDINATION.md` is the canonical state-of-the-world doc for extraction work. Read at session start, update at session end.
- **2026-05-03** — Coordination protocol refinements: ISO 8601 UTC timestamps; alphabetical suffix scheme (`-b`, `-c`, ...) for AI sessions colliding on a date, claimed in the same commit; unknown-owner fallback (treat as locked); tie-breaker on simultaneous claims (last write wins, loser negotiates); forgive-and-claim for missed-step recovery. CI enforcement deferred to PR-Coord-2.
- **2026-05-03** — Active session letter aliases (A/B/C) added as conversational shorthand alongside canonical agent-date IDs (Option 2 over replacement). Aliases re-anchor each calendar day; agent-date IDs remain canonical in tables and decisions log.
- **2026-05-03** — Post-merge cleanup: PRs #77, #78, #79, #80, #81, #82, #83, #84, #85 merged into main. Per-product state advanced for `extracted_competitive_intelligence` (#80), `extracted_content_pipeline` (#78), `extracted_reasoning_core` (#82, now Phase 1 with wedge consolidated), `extracted_quality_gate` (#85, now Phase 1 with product_claim core landed). `PR-Coord` and `PR-A0` slices retired. PR-B1 retired (merged as #84). PR-B2 retired (merged as #85).
- **2026-05-03** — PR-C1 claimed by `claude-2026-05-03` for the reasoning evidence/temporal/archetypes consolidation. Hot zone recorded in per-product state and Upcoming queue.
- **2026-05-03** — PR-A1 merged as #87. `llm_exact_cache.py` and migration 251 are now in the LLM-infrastructure manifest; PR-A2 and PR-A3 are unblocked.
- **2026-05-03** — Coordination timestamp protocol tightened: stamps must be monotonic using `max(now, last_stamp + 1 minute)` so future edits cannot regress the audit log.
- **2026-05-03** — Coordination doc split into per-section files under `docs/extraction/coordination/` to reduce merge-conflict contention. COORDINATION.md is now an index + protocol + conventions; the four high-frequency sections (state, inflight, queue, decisions) live in separate files. Sessions touching different sections no longer conflict at the file level. PR-Coord-2 (CI enforcement of COORDINATION updates) remains deferred.
