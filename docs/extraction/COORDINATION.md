# Extraction Coordination

Last updated: 2026-05-03T18:42Z by codex-2026-05-03

State-of-the-world for the multi-product extraction effort. Read this end-to-end at session start before doing substantive work. Update before opening a PR, after merging one, or when a decision lands.

The team is one human (`@canfieldjuan`) plus AI sessions. Owner column uses GitHub usernames for human work and agent-stamped session IDs for AI work (`{agent}-YYYY-MM-DD[-suffix]`, e.g. `claude-2026-05-03`, `codex-2026-05-03`). The first session for an agent on a calendar day is unsuffixed; subsequent same-agent sessions claim alphabetical suffixes from `-b` (`claude-2026-05-03-b`, `codex-2026-05-03-b`, …) in the same commit that claims a slice. Timestamps in this doc use ISO 8601 UTC (`YYYY-MM-DDTHH:MMZ`).

**Active session aliases (2026-05-03)** — for conversational shorthand: `A` = `claude-2026-05-03-b` (PR #81 authoring / PR-A0 claim), `B` = `codex-2026-05-03` (PR #81 review + PR #82 coordination update), `C` = `claude-2026-05-03` (PRs #79, #80, #82). Aliases re-anchor each calendar day. Agent-date IDs remain canonical in all tables; aliases are for in-conversation reference only.

---

## Per-product state

| Product | Phase | Most recent merged PR | Active PRs | Next milestone | Active hot zone |
|---|---|---|---|---|---|
| `extracted_llm_infrastructure` | 2 (standalone toggle landed; Phase 3 decoupling pending) | #49 | — | Cost-closure additions (PR-A1 → A4) | none |
| `extracted_competitive_intelligence` | 1 (scaffold) | #48 | #80 | Stabilize after #80 wedge migration | `reasoning/wedge_registry.py` |
| `extracted_content_pipeline` | 1 → 2 (productization seams in flight) | #76 | #77, #78 | Standalone runner without `atlas_brain` on path | `campaign_generation.py`, `*_postgres_*`, `README.md`, `STATUS.md`, `docs/remaining_productization_audit.md` |
| `extracted_reasoning_core` | 0 → 1 (kickoff) | — | #79, #80, #82 | First scaffold + wedge registry land; evidence/temporal/archetypes audit queued via #82 | `extracted_reasoning_core/**`, `docs/extraction/evidence_temporal_archetypes_audit_2026-05-03.md` |
| `extracted_quality_gate` | not started | — | — | Boundary audit (deferred behind cost-closure) | — |

Phase legend: 0 = pre-extraction (audit doc only). 1 = byte-for-byte scaffold, still imports from `atlas_brain`. 2 = standalone toggle (`EXTRACTED_X_STANDALONE=1`) loads local substrate. 3 = full Protocol-based decoupling, no `atlas_brain` runtime imports.

---

## In-flight PRs (claim before opening, update when state changes)

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #77 | docs: park product strategy notes | `extracted_content_pipeline/docs/long_form_creative_backlog.md`, `extracted_content_pipeline/docs/podcast_repurposing_landing_page_strategy.md`, `extracted_content_pipeline/docs/remaining_productization_audit.md` | (unknown — confirm) | #78 on `extracted_content_pipeline/docs/remaining_productization_audit.md` |
| #78 | Add multi-channel campaign generation flow | `extracted_content_pipeline/campaign_generation.py`, postgres runners, README, STATUS, tests | (unknown — confirm) | `extracted_content_pipeline/{campaign_generation.py, campaign_postgres_generation.py, campaign_example.py, README.md, STATUS.md, docs/remaining_productization_audit.md, docs/standalone_productization.md}`; `scripts/run_extracted_campaign_generation_*.py`; `tests/test_extracted_campaign_*.py` |
| #79 | Document reasoning core extraction boundary | `docs/extraction/reasoning_boundary_audit_2026-05-03.md` | claude-2026-05-03 | (docs only) |
| #80 | Add shared reasoning core wedge registry | `extracted_reasoning_core/**`, `extracted_competitive_intelligence/reasoning/wedge_registry.py`, `extracted_content_pipeline/reasoning/wedge_registry.py`, tests | claude-2026-05-03 | `extracted_reasoning_core/**`, the migrated `wedge_registry.py` files |
| #81 | Add extraction coordination doc | `docs/extraction/COORDINATION.md` | claude-2026-05-03-b (drafted), codex-2026-05-03 (reviewed + #82 coordination update) | coordination-doc edits; claim before touching `docs/extraction/COORDINATION.md` |
| #82 | Document reasoning evidence-temporal-archetypes consolidation | `docs/extraction/evidence_temporal_archetypes_audit_2026-05-03.md` | claude-2026-05-03 | (docs only) |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.

---

## Upcoming queue (claim before starting; sequence reflects dependencies)

| Slice | Product | Owner | Dependencies | Notes |
|---|---|---|---|---|
| PR-Coord | meta | claude-2026-05-03-b | none | This doc. Establishes the mechanism. |
| PR-A0 | `extracted_llm_infrastructure` | claude-2026-05-03-b | none | Boundary audit doc: `docs/extraction/cost_closure_audit_2026-05-04.md`. Mirrors PR #79's structure. |
| PR-A1 | `extracted_llm_infrastructure` | unclaimed | PR-A0 | Add `services/b2b/llm_exact_cache.py` + migration `251_b2b_llm_exact_cache.sql` (rename target: `llm_exact_cache.sql`) to manifest. Update README "What's in scope" table. |
| PR-A2 | `extracted_llm_infrastructure` | unclaimed | PR-A1 | Add `services/provider_cost_sync.py` + migration `258_provider_cost_reconciliation.sql`. Sync orchestration. |
| PR-A3 | `extracted_llm_infrastructure` | unclaimed | PR-A1 | New code: cache-savings persistence layer + migration. Closes the "$ saved by cache" telemetry gap. |
| PR-A4 | `extracted_llm_infrastructure` | unclaimed | PR-A2, PR-A3 | New code: drift report (local vs invoiced), budget gate, OpenAI provider adapter. May split if too large. |
| PR-B1 | `extracted_quality_gate` | unclaimed | (independent of A) | Boundary audit doc. Mirrors PR #79. Can run in parallel with cost-closure if a second session opens. |
| PR-C1 | `extracted_reasoning_core` | unclaimed | PR #80, PR #82 | Consolidate evidence/temporal/archetypes after #82 lands: `archetypes.py`, `evidence_engine.py`, `temporal.py`, `evidence_map.yaml`, plus PR #79 contract amendment. |

---

## Decisions log (chronological, append-only)

- **2026-05-01** — Folders stay siblings of `atlas_brain/`, not relocated under `extracted/`. Only `extracted/_shared/` lives in the umbrella. Path moves would touch hundreds of references in manifests, READMEs, and CI; not worth the disruption.
- **2026-05-01** — Wrapper-script pattern for shared tooling rollout: keep existing entry-point script names as thin wrappers that delegate to `extracted/_shared/scripts/`. Preserves CI references. Settled by PRs #48–50.
- **2026-05-03** — Reasoning is its own extracted product (`extracted_reasoning_core`), not a leaf duplicated into each consumer. Boundary doc + skeleton + compat-wrapper migration. Settled by PRs #79, #80.
- **2026-05-03** — Cost-closure additions (`llm_exact_cache.py`, `provider_cost_sync.py`, migrations 251 + 258, plus new code: cache-savings, drift report, budget gate, OpenAI adapter) go INTO `extracted_llm_infrastructure`. No separate `llm-spend-py` package.
- **2026-05-03** — `docs/extraction/COORDINATION.md` is the canonical state-of-the-world doc for extraction work. Read at session start, update at session end.
- **2026-05-03** — Coordination protocol refinements: ISO 8601 UTC timestamps; alphabetical suffix scheme (`-b`, `-c`, …) for AI sessions colliding on a date, claimed in the same commit; unknown-owner fallback (treat as locked); tie-breaker on simultaneous claims (last write wins, loser negotiates); forgive-and-claim for missed-step recovery. CI enforcement deferred to PR-Coord-2.
- **2026-05-03** — Active session letter aliases (A/B/C) added as conversational shorthand alongside canonical agent-date IDs (Option 2 over replacement). Aliases re-anchor each calendar day; agent-date IDs remain canonical in tables and decisions log.

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
- **Boundary audit docs** — land in `docs/extraction/<product>_boundary_audit_<date>.md` BEFORE the first scaffold PR. PR #79 is the template.
- **Per-product status** — STATUS.md inside each `extracted_*/` folder is the product-internal state. This doc is the cross-product state. Don't duplicate detail; link.

## What this doc is NOT for

- Detailed product roadmaps — those live in each product's `STATUS.md` or boundary audit doc.
- Architecture decisions specific to one product — capture those in the relevant boundary audit or README.
- A real-time PR mirror — `gh pr list` is the source of truth for what's open. This doc tracks intent and ownership for in-flight work we're coordinating around.
- Long discussion threads — keep this scannable. Conversations belong in PR descriptions and review comments; only the *outcome* lands in *Decisions log*.
