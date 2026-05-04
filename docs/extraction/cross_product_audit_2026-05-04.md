# Cross-Product Standalone Audit — Content Ops vs Competitive Intelligence

Date: 2026-05-04

Side-by-side audit of the two largest extracted product packages so the
two numbers are comparable at a glance, and so the user can decide
whether to ship single-pass v0 of each or invest in multi-pass
reasoning first.

Mirrors the methodology of `reasoning_boundary_audit_2026-05-03.md` and
the per-package audit on PR #122. Different scope: this one covers
both packages in one pass and answers the specific differentiator
question "does competitive intelligence already have multi-pass /
cross-vendor reasoning that AI Content Ops doesn't?"

## TL;DR

|  | extracted_content_pipeline | extracted_competitive_intelligence |
|---|---|---|
| **Standalone-usable today** | **~85%** | **~32%** |
| Total Python LOC | 64,500 across 103 files | 16,684 across 75 files |
| Manifest-mapped (Atlas snapshots) | 40 files / ~53,500 LOC | 21 files / ~14,300 LOC |
| Product-owned | 40 files / ~8,100 LOC (12.5%) | 3 files / ~864 LOC (5.2%) |
| Multi-pass LLM reasoning | None (Tier 2 stubs `NotImplementedError`) | None (architecturally single-pass) |
| What's blocking 100% | Reasoning *producer* (architectural choice, not debt) | Phase 2 runtime substrate (config, DB pool, LLM, auth, service bridges) |

## What "standalone-usable" means

Both packages are measured against the same bar:

1. The extracted package imports clean with no `atlas_brain` runtime
   dependency (`audit_extracted_standalone.py --fail-on-debt` returns
   `Atlas runtime import findings: 0`).
2. The buyer can run a documented happy-path runbook end-to-end on a
   host with Postgres + Python + the package's `requirements.txt`
   installed, no atlas_brain on the path.
3. Where reasoning is needed, the package accepts pre-baked reasoning
   from the host (via a Protocol port) or operates with a defensive
   fallback at known-lower quality.

## AI Content Ops — 85%

- Standalone debt audit: **0 atlas_brain runtime imports**.
- Buyer happy path (`migrate / import / generate / export`) works today
  via 5 CLI commands. Documented in
  `extracted_content_pipeline/docs/host_install_runbook.md` and on
  PR #122.
- The reasoning *consumer* surface is complete and clean:
  `CampaignReasoningContextProvider` Protocol +
  `FileCampaignReasoningContextProvider` reference adapter.
- The 15% gap is the reasoning *producer*. The package consumes
  reasoning but doesn't generate it.
- `extracted_reasoning_core` exposes three working evaluators
  (`score_archetypes`, `evaluate_evidence`, `build_temporal_evidence`)
  but its four producer-shaped functions (`run_reasoning`,
  `continue_reasoning`, `check_falsification`,
  `build_narrative_plan`) all raise `NotImplementedError`.
- Per `remaining_productization_audit.md:338-343`, this is the
  documented Option B decision: reasoning is host-owned. The 15% is
  *architectural choice*, not extraction debt.

## Competitive Intelligence — 32%

The package is still Phase 1 scaffold (per
`extracted_competitive_intelligence/STATUS.md`):

- 21 of 24 manifest-mapped files are byte-snapshots of `atlas_brain`.
- Only 3 files are product-owned: the read-only MCP tools
  (`vendor_registry.py`, `displacement.py`, `cross_vendor.py`).
- Phase 2 (standalone substrate: config, DB pool, LLM bridge, auth,
  campaign sender, suppression policy) is in progress — the
  `b2b_battle_cards.py` LLM call still routes through atlas.
- Per `STATUS.md` Phase 2 column, only 2 of 15 reasoning-adjacent
  files are Phase 2 ready, both pure prompt strings:
  - `reasoning/single_pass_prompts/cross_vendor_battle.py` (55 LOC)
  - `reasoning/single_pass_prompts/battle_card_reasoning.py` (327 LOC)
- The other 13 files (battle cards task, vendor briefing task, cross-
  vendor synthesis helper, b2b shared, etc.) are still Atlas-shaped.

## The reasoning depth comparison

User-asked: does competitive intelligence already include multi-pass
or cross-vendor reasoning that AI Content Ops doesn't?

**No. Both packages use the same single-pass-with-rich-context
pattern.** The differentiator between them is *what* they reason
over, not *how deeply*.

| Aspect | AI Content Ops | Competitive Intelligence |
|---|---|---|
| Reasoning execution | Single-pass LLM call per opportunity | Single-pass LLM call per battle / briefing |
| Multi-pass loops | Architected via Tier 2 producer stubs in `extracted_reasoning_core/api.py`; **all four raise `NotImplementedError`** | Not architected at all; no producer stubs, no orchestration scaffolding |
| Self-check pattern | Embedded in single-pass prompt; no follow-up call | Embedded in single-pass prompt (`SELF-CHECK` block in `reasoning/single_pass_prompts/cross_vendor_battle.py:36-43`); no follow-up call |
| Falsification | Mentioned in output contracts (`falsification_conditions` field); no multi-pass check | Mentioned in prompts; no follow-up call to verify or refute |
| Cross-vendor reasoning | N/A (single-vendor scope) | **Selection logic is deterministic** (`reasoning/cross_vendor_selection.py`, 404 LOC: scores edges by overlap, displacement, segment divergence); pairwise battles are then **single-pass LLM calls** with no back-and-forth refinement |
| Knowledge graph traversal | None | None (the `atlas_brain/reasoning/` knowledge graph is *not* extracted; comp intel reasons over flat edge tables) |

## What this means for product strategy

The user's plan ("ship single-pass v0 of both, extract reasoning
layer separately, sell as upsell") is **directionally right and
well-aligned with what's actually built**:

- AI Content Ops is single-pass today and ready to ship as such.
- Competitive Intelligence is single-pass today (with deterministic
  cross-vendor selection feeding the prompts) and ready to ship as
  such once Phase 2 substrate work lands.
- Multi-pass reasoning — what the user described as the
  differentiator versus most apps — does not exist in either package
  today. It would be a separate engineering build:
  - For AI Content Ops: implement the four `extracted_reasoning_core`
    producer stubs + adapter (~2-3 weeks, ~2,000 LOC).
  - For Competitive Intelligence: add a falsification / refinement
    loop after each battle card generation (new architecture, not in
    scope today).

The reasoning upsell is therefore a **separate sellable**, not an
integration tax on the v0 launches.

## Honest claim-vs-code flag for marketing

If any landing page copy positions cross-vendor reasoning as
"multi-pass" or "iterative refinement", that claim does not match
the code today. The cross-vendor *domain* is real (deterministic edge
selection + per-pair LLM analysis) — but the reasoning *style* is
single-pass. Worth checking landing-page copy before either product
goes live.

## References

- `extracted_content_pipeline/docs/reasoning_state_audit.md` — per-
  package detail and tier sizings (PR #122)
- `extracted_content_pipeline/docs/remaining_productization_audit.md` —
  Option B decision, reasoning host-owned
- `extracted_competitive_intelligence/STATUS.md` — phase tracking
- `docs/extraction/reasoning_boundary_audit_2026-05-03.md` —
  earlier reasoning boundary audit
- `extracted_reasoning_core/api.py` — evaluator + producer-stub surface
