# Extraction Coordination

Last updated: 2026-05-03T22:30Z by claude-2026-05-03-b

State-of-the-world for the multi-product extraction effort. Read this end-to-end at session start before doing substantive work. Update before opening a PR, after merging one, or when a decision lands.

The team is one human (`@canfieldjuan`) plus AI sessions. Owner column uses GitHub usernames for human work and agent-stamped session IDs for AI work (`{agent}-YYYY-MM-DD[-suffix]`, e.g. `claude-2026-05-03`, `codex-2026-05-03`). The first session for an agent on a calendar day is unsuffixed; subsequent same-agent sessions claim alphabetical suffixes from `-b` (`claude-2026-05-03-b`, `codex-2026-05-03-b`, …) in the same commit that claims a slice. Timestamps in this doc use ISO 8601 UTC (`YYYY-MM-DDTHH:MMZ`).

**Active session aliases (2026-05-03)** — for conversational shorthand: `A` = `claude-2026-05-03-b` (PR #81 authoring / PR-A0 claim), `B` = `codex-2026-05-03` (PR #81 review, PR #82 coordination update, PR-B1 quality-gate audit), `C` = `claude-2026-05-03` (PRs #79, #80, #82). Aliases re-anchor each calendar day. Agent-date IDs remain canonical in all tables; aliases are for in-conversation reference only.

---

## Per-product state

| Product | Phase | Most recent merged PR | Active PRs | Next milestone | Active hot zone |
|---|---|---|---|---|---|
| `extracted_llm_infrastructure` | 2 (standalone toggle landed; Phase 3 decoupling pending) | #87 | #89 (PR-A2), #90 (PR-A1.5), (PR-A3 opening) | Cost-closure additions (PR-A2 lift, PR-A1.5 cleanup, PR-A3 cache-savings new code, PR-A4 next) | `extracted_llm_infrastructure/services/cost/`, `extracted_llm_infrastructure/storage/migrations/259_*` |
| `extracted_competitive_intelligence` | 1 (scaffold) | #80 | — | Phase 2 standalone toggle | none |
| `extracted_content_pipeline` | 1 -> 2 (productization seams) | #78 | — | Standalone runner without `atlas_brain` on path | none |
| `extracted_reasoning_core` | 1 (scaffold + wedge consolidated; PR-C1 claimed) | #82 | — (PR-C1 claimed by claude-2026-05-03) | Evidence/temporal/archetypes consolidation per merged PR #82 audit | `extracted_reasoning_core/**` (api/types/archetypes/evidence_engine/evidence_map.yaml/temporal); `atlas_brain/reasoning/{evidence_engine.py, review_enrichment.py}`; `extracted_content_pipeline/reasoning/{archetypes,evidence_engine,temporal}.py`; `tests/test_extracted_reasoning_*.py` |
| `extracted_quality_gate` | 1 (scaffold + product_claim core landed via #85) | #85 | — | Safety-gate split (PR-B3); blog + campaign packs (PR-B4) | none |

Phase legend: 0 = pre-extraction (audit doc only). 1 = byte-for-byte scaffold, still imports from `atlas_brain`. 2 = standalone toggle loads local substrate (per-product env var: `EXTRACTED_LLM_INFRA_STANDALONE`, `EXTRACTED_COMP_INTEL_STANDALONE`, `EXTRACTED_PIPELINE_STANDALONE`, etc.; see `extracted/METHODOLOGY.md` for the canonical list). 3 = full Protocol-based decoupling, no `atlas_brain` runtime imports.

---

## In-flight PRs (claim before opening, update when state changes)

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #89 | Add `provider_cost_sync` to LLM-infrastructure manifest (PR-A2) | `extracted_llm_infrastructure/{manifest.json, services/provider_cost_sync.py, storage/migrations/258_provider_cost_reconciliation.sql, README.md, STATUS.md}`; `docs/extraction/COORDINATION.md` | claude-2026-05-03-b | manifest edits or files synced into `extracted_llm_infrastructure/services/` and `storage/migrations/258_*` |
| #90 | Re-apply Copilot fixes that missed PR #87 merge (PR-A1.5) | `extracted_llm_infrastructure/{skills/__init__.py, _standalone/config.py, STATUS.md}`; `scripts/smoke_extracted_llm_infrastructure_imports.py`; `scripts/smoke_extracted_llm_infrastructure_standalone.py` | claude-2026-05-03-b | the 5 listed files |
| (PR-A3, opening) | Add cache-savings persistence (NEW CODE) | `extracted_llm_infrastructure/{manifest.json, services/cost/__init__.py, services/cost/cache_savings.py, storage/migrations/259_llm_cache_savings.sql, README.md, STATUS.md}`; `tests/test_extracted_llm_infrastructure_cache_savings.py`; `docs/extraction/COORDINATION.md` | claude-2026-05-03-b | `services/cost/` (any file) or `storage/migrations/259_*` |
_(Rows for merged PRs #77, #78, #79, #80, #81, #82, #83, #84, #85, #86, #87 dropped per session protocol step 4. Outcomes preserved in Decisions log and per-product state.)_

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

---

## Upcoming queue (claim before starting; sequence reflects dependencies)

| Slice | Product | Owner | Dependencies | Notes |
|---|---|---|---|---|
| PR-A3 | `extracted_llm_infrastructure` | claude-2026-05-03-b | PR-A1 / #87 (merged) | New code: cache-savings persistence layer + migration 259. Closes the "$ saved by cache" telemetry gap. **In flight.** |
| PR-A4 | `extracted_llm_infrastructure` | unclaimed | PR-A2, PR-A3 | New code: drift report (local vs invoiced), budget gate, OpenAI provider adapter. May split if too large. |
| PR-B3 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | Safety-gate split: deterministic content/risk scan to core; approvals + audit log + DB to ports + Atlas adapter wrapper. |
| PR-B4 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | Blog + campaign quality packs over the core gate contract. |
| PR-B5 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | B2B evidence + witness + source-quality packs. |
| PR-C1 | `extracted_reasoning_core` | claude-2026-05-03 | PR #80, PR #82 (both merged) | Consolidate evidence/temporal/archetypes per merged PR #82 audit. NEW in core: `archetypes.py`, `evidence_engine.py` (slim conclusions+suppression surface), `evidence_map.yaml`, `temporal.py` (with `_numeric_value` / `_row_get` helpers + parameterized `MIN_DAYS_FOR_PERCENTILES`). Atlas-side: NEW `atlas_brain/reasoning/review_enrichment.py`; slim `atlas_brain/reasoning/evidence_engine.py`. Convert `extracted_content_pipeline/reasoning/{archetypes,evidence_engine,temporal}.py` to re-export wrappers. EDIT `extracted_reasoning_core/api.py` (impl 3 stubs) and `extracted_reasoning_core/types.py` (rich `TemporalEvidence` + 4 sub-types + `ConclusionResult` + `SuppressionResult`). Rename + redirect `tests/test_extracted_reasoning_*.py`. PR #79 contract amendment lands in the same commit. |

---

## Decisions log (chronological, append-only)

- **2026-05-01** — Folders stay siblings of `atlas_brain/`, not relocated under `extracted/`. Only `extracted/_shared/` lives in the umbrella. Path moves would touch hundreds of references in manifests, READMEs, and CI; not worth the disruption.
- **2026-05-01** — Wrapper-script pattern for shared tooling rollout: keep existing entry-point script names as thin wrappers that delegate to `extracted/_shared/scripts/`. Preserves CI references. Settled by PRs #48–50.
- **2026-05-03** — Reasoning is its own extracted product (`extracted_reasoning_core`), not a leaf duplicated into each consumer. Boundary doc + skeleton + compat-wrapper migration. Settled by PRs #79, #80.
- **2026-05-03** — Cost-closure additions (`llm_exact_cache.py`, `provider_cost_sync.py`, migrations 251 + 258, plus new code: cache-savings, drift report, budget gate, OpenAI adapter) go INTO `extracted_llm_infrastructure`. No separate `llm-spend-py` package.
- **2026-05-03** — `docs/extraction/COORDINATION.md` is the canonical state-of-the-world doc for extraction work. Read at session start, update at session end.
- **2026-05-03** — Coordination protocol refinements: ISO 8601 UTC timestamps; alphabetical suffix scheme (`-b`, `-c`, …) for AI sessions colliding on a date, claimed in the same commit; unknown-owner fallback (treat as locked); tie-breaker on simultaneous claims (last write wins, loser negotiates); forgive-and-claim for missed-step recovery. CI enforcement deferred to PR-Coord-2.
- **2026-05-03** — Active session letter aliases (A/B/C) added as conversational shorthand alongside canonical agent-date IDs (Option 2 over replacement). Aliases re-anchor each calendar day; agent-date IDs remain canonical in tables and decisions log.
- **2026-05-03** — Post-merge cleanup: PRs #77, #78, #79, #80, #81, #82, #83, #84, #85 merged into main. Per-product state advanced for `extracted_competitive_intelligence` (#80), `extracted_content_pipeline` (#78), `extracted_reasoning_core` (#82, now Phase 1 with wedge consolidated), `extracted_quality_gate` (#85, now Phase 1 with product_claim core landed). `PR-Coord` and `PR-A0` slices retired. PR-B1 retired (merged as #84). PR-B2 retired (merged as #85).
- **2026-05-03** — PR-C1 claimed by `claude-2026-05-03` for the reasoning evidence/temporal/archetypes consolidation. Hot zone recorded in per-product state and Upcoming queue.
- **2026-05-03** — PR-A1 merged as #87. `llm_exact_cache.py` and migration 251 are now in the LLM-infrastructure manifest; PR-A2 and PR-A3 are unblocked.
- **2026-05-03** — Coordination timestamp protocol tightened: stamps must be monotonic using `max(now, last_stamp + 1 minute)` so future edits cannot regress the audit log.

---

## Open questions / blockers

- **Future hardening (deferred)**: a CI check that requires any merged PR touching `extracted_*/` to also modify `COORDINATION.md`. Forces the protocol mechanically instead of relying on convention. Land as a follow-up PR-Coord-2 once the doc has hit real friction (i.e. someone has demonstrably forgotten to update).

---

## Session protocol

1. **At session start**: read this doc end-to-end before opening files.
2. **Before opening a PR**: add a row to *In-flight PRs* with your owner ID and the files you'll touch.
3. **Before starting code on a queued slice**: claim it in *Upcoming queue* (set Owner) so a parallel session doesn't pick the same one.
4. **After a PR merges**: update *Per-product state* (most recent PR, next milestone), drop the row from *In-flight PRs*, log any decisions made during review.
5. **When a decision lands**: append to *Decisions log* with the date. Never edit historical entries; supersede with a newer entry instead.
6. **Update the "Last updated" stamp** every time you touch this file. ISO 8601 UTC: `YYYY-MM-DDTHH:MMZ`. Stamps must be monotonic relative to the previous value: write `max(now, last_stamp + 1 minute)`. If the prior stamp is in the future relative to your real clock (clock drift, estimation), still bump past it -- the audit log must never regress.
7. **Tie-breaker on simultaneous claims**: if two sessions claim the same slice within minutes, last commit to this file wins; the loser pivots to a different slice or negotiates in PR comments before opening a competing PR.
8. **Forgive-and-claim**: if you opened a PR without first adding a row, add the row before requesting review. Skipping the claim once is not punishable; abandoning the protocol is.

---

## Conventions

- **Owner format** — GitHub username (`@canfieldjuan`) for human work; `{agent}-YYYY-MM-DD[-suffix]` for AI session work, e.g. `claude-2026-05-03`, `codex-2026-05-03-b`.
- **Unknown-owner fallback** — if an in-flight PR's Owner is `(unknown — confirm)`, treat its listed file paths as locked until the owner is filled in. Safer default than racing on an unattributed PR.
- **PR title verbs** — match the established pattern: `Add X`, `Own X`, `Route X through Y`, `Document X`, `Harden X`, `Refresh X`. The verb signals intent (Phase 1 add vs Phase 2 ownership vs Phase 3 decoupling vs docs).
- **Boundary / consolidation audit docs** — land in `docs/extraction/<slug>_audit_<date>.md` (with optional `_boundary` infix for first-PR boundary audits) BEFORE the first scaffold PR. `<slug>` is the slice or topic, not the full product name (examples: `reasoning_boundary_audit_2026-05-03.md`, `quality_gate_boundary_audit_2026-05-03.md`, `cost_closure_audit_2026-05-03.md`, `evidence_temporal_archetypes_audit_2026-05-03.md`). PR #79 is the template.
- **Per-product status** — STATUS.md inside each `extracted_*/` folder is the product-internal state. This doc is the cross-product state. Don't duplicate detail; link.

## What this doc is NOT for

- Detailed product roadmaps — those live in each product's `STATUS.md` or boundary audit doc.
- Architecture decisions specific to one product — capture those in the relevant boundary audit or README.
- A real-time PR mirror — `gh pr list` is the source of truth for what's open. This doc tracks intent and ownership for in-flight work we're coordinating around.
- Long discussion threads — keep this scannable. Conversations belong in PR descriptions and review comments; only the *outcome* lands in *Decisions log*.
