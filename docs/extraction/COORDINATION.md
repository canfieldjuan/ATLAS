# Extraction Coordination

Last updated: 2026-05-03T21:00Z by claude-2026-05-03-b

State-of-the-world for the multi-product extraction effort. Read this end-to-end at session start before doing substantive work. Update before opening a PR, after merging one, or when a decision lands.

The team is one human (`@canfieldjuan`) plus AI sessions. Owner column uses GitHub usernames for human work and agent-stamped session IDs for AI work (`{agent}-YYYY-MM-DD[-suffix]`, e.g. `claude-2026-05-03`, `codex-2026-05-03`). The first session for an agent on a calendar day is unsuffixed; subsequent same-agent sessions claim alphabetical suffixes from `-b` (`claude-2026-05-03-b`, `codex-2026-05-03-b`, …) in the same commit that claims a slice. Timestamps in this doc use ISO 8601 UTC (`YYYY-MM-DDTHH:MMZ`).

**Active session aliases (2026-05-03)** — for conversational shorthand: `A` = `claude-2026-05-03-b` (PR #81 authoring / PR-A0 claim), `B` = `codex-2026-05-03` (PR #81 review, PR #82 coordination update, PR-B1 quality-gate audit), `C` = `claude-2026-05-03` (PRs #79, #80, #82). Aliases re-anchor each calendar day. Agent-date IDs remain canonical in all tables; aliases are for in-conversation reference only.

---

## Per-product state

| Product | Phase | Most recent merged PR | Active PRs | Next milestone | Active hot zone |
|---|---|---|---|---|---|
| `extracted_llm_infrastructure` | 2 (standalone toggle landed; Phase 3 decoupling pending) | #49 | #87 | Cost-closure additions (PR-A1 -> A4); A1 adds `llm_exact_cache.py` + migration 251 | `extracted_llm_infrastructure/services/b2b/llm_exact_cache.py`, `extracted_llm_infrastructure/storage/migrations/251_b2b_llm_exact_cache.sql`, manifest, README, STATUS |
| `extracted_competitive_intelligence` | 1 (scaffold) | #80 | — | Phase 2 standalone toggle | none |
| `extracted_content_pipeline` | 1 -> 2 (productization seams) | #78 | — | Standalone runner without `atlas_brain` on path | none |
| `extracted_reasoning_core` | 1 (scaffold + wedge consolidated) | #82 | — | Evidence/temporal/archetypes consolidation per merged #82 audit (PR-C1 queued) | none |
| `extracted_quality_gate` | 1 (scaffold + product_claim core landed via #85) | #85 | — | Safety-gate split (PR-B3); blog + campaign packs (PR-B4) | none |

Phase legend: 0 = pre-extraction (audit doc only). 1 = byte-for-byte scaffold, still imports from `atlas_brain`. 2 = standalone toggle loads local substrate (per-product env var: `EXTRACTED_LLM_INFRA_STANDALONE`, `EXTRACTED_COMP_INTEL_STANDALONE`, `EXTRACTED_PIPELINE_STANDALONE`, etc.; see `extracted/METHODOLOGY.md` for the canonical list). 3 = full Protocol-based decoupling, no `atlas_brain` runtime imports.

---

## In-flight PRs (claim before opening, update when state changes)

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #87 | Add `llm_exact_cache` to LLM-infrastructure manifest | `extracted_llm_infrastructure/{manifest.json, services/b2b/llm_exact_cache.py, storage/migrations/251_b2b_llm_exact_cache.sql, README.md, STATUS.md}`; `docs/extraction/COORDINATION.md` | claude-2026-05-03-b | manifest edits or files synced into `extracted_llm_infrastructure/services/b2b/` and `storage/migrations/` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

---

## Upcoming queue (claim before starting; sequence reflects dependencies)

| Slice | Product | Owner | Dependencies | Notes |
|---|---|---|---|---|
| PR-A1 | `extracted_llm_infrastructure` | claude-2026-05-03-b | none (PR-A0 / #83 merged) | Add `services/b2b/llm_exact_cache.py` + migration `251_b2b_llm_exact_cache.sql` to manifest. Update README + STATUS. **In flight.** |
| PR-A2 | `extracted_llm_infrastructure` | unclaimed | PR-A1 | Add `services/provider_cost_sync.py` + migration `258_provider_cost_reconciliation.sql`. Sync orchestration. |
| PR-A3 | `extracted_llm_infrastructure` | unclaimed | PR-A1 | New code: cache-savings persistence layer + migration. Closes the "$ saved by cache" telemetry gap. |
| PR-A4 | `extracted_llm_infrastructure` | unclaimed | PR-A2, PR-A3 | New code: drift report (local vs invoiced), budget gate, OpenAI provider adapter. May split if too large. |
| PR-B3 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | Safety-gate split: deterministic content/risk scan to core; approvals + audit log + DB to ports + Atlas adapter wrapper. |
| PR-B4 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | Blog + campaign quality packs over the core gate contract. |
| PR-B5 | `extracted_quality_gate` | unclaimed | PR-B2 / #85 (merged) | B2B evidence + witness + source-quality packs. |
| PR-C1 | `extracted_reasoning_core` | unclaimed | PR #80, PR #82 (both merged) | Consolidate evidence/temporal/archetypes per merged #82 audit: `archetypes.py`, `evidence_engine.py`, `temporal.py`, `evidence_map.yaml`, plus PR #79 contract amendment. |

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

---

## Open questions / blockers

- Owners for in-flight PRs #77, #78 — separate AI sessions; pending session-ID confirmation from `@canfieldjuan`. PRs #79, #80, and #82 confirmed as `claude-2026-05-03`.
- **Future hardening (deferred)**: a CI check that requires any merged PR touching `extracted_*/` to also modify `COORDINATION.md`. Forces the protocol mechanically instead of relying on convention. Land as a follow-up PR-Coord-2 once the doc has hit real friction (i.e. someone has demonstrably forgotten to update).

---

## Session protocol

1. **At session start**: read this doc end-to-end before opening files.
2. **Before opening a PR**: add a row to *In-flight PRs* with your owner ID and the files you'll touch.
3. **Before starting code on a queued slice**: claim it in *Upcoming queue* (set Owner) so a parallel session doesn't pick the same one.
4. **After a PR merges**: update *Per-product state* (most recent PR, next milestone), drop the row from *In-flight PRs*, log any decisions made during review.
5. **When a decision lands**: append to *Decisions log* with the date. Never edit historical entries; supersede with a newer entry instead.
6. **Update the "Last updated" stamp** every time you touch this file. ISO 8601 UTC: `YYYY-MM-DDTHH:MMZ`.
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
