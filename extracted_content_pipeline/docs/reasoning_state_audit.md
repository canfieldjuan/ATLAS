# AI Content Ops — Reasoning State Audit

Date: 2026-05-04

This audit answers a single concrete question: **how far along is the
extracted AI Content Ops product before a buyer can add a source and
start reasoning over it end-to-end without atlas_brain on the path?**

Scope: the primary AI Content Ops product (B2B campaign generation
spine inside `extracted_content_pipeline/`). Not the podcast
repurposing offer that shipped in PR #121 — that is a separate product
with its own data flow.

## TL;DR

- The product is **shippable today** for buyers who bring their own reasoning
  input as JSON, configure the packaged single-pass reasoning provider, or
  accept a known quality penalty from running without reasoning.
- The standalone debt audit reports **0 atlas_brain runtime imports**.
- The reasoning *consumer* surface is complete and clean.
- The extracted package now includes a Tier 1 opportunity-level reasoning
  producer: `SinglePassCampaignReasoningProvider`. It does not replace the
  heavier reasoning-core producer stubs; `extracted_reasoning_core` still has
  four producer-shaped functions (`run_reasoning`, `continue_reasoning`,
  `check_falsification`, `build_narrative_plan`) that raise
  `NotImplementedError`.
- No branch in flight wires `extracted_reasoning_core` into the
  content-pipeline generator.

## What works today end-to-end without atlas_brain

The standalone audit:

```bash
python scripts/audit_extracted_standalone.py --fail-on-debt
# Atlas runtime import findings: 0
```

The buyer-facing happy path. Prerequisites: Python 3.11+, Postgres,
plus `pip install -r requirements.txt` (the DB-backed CLIs need
`asyncpg`). Step 4 below uses `--llm offline` so the path runs without
external LLM credentials. To run against a real LLM, drop `--llm
offline` and set `EXTRACTED_CAMPAIGN_LLM_*` per the host install
runbook.

```bash
# 1. Migrations
export EXTRACTED_DATABASE_URL="postgres://user:pass@localhost/content_ops"
python scripts/run_extracted_content_pipeline_migrations.py

# 2. Validate offline (optional but recommended; no DB required)
python scripts/run_extracted_campaign_generation_example.py \
  customer_opportunities.csv --format csv

# 3. Load opportunities
python scripts/load_extracted_campaign_opportunities.py \
  customer_opportunities.csv --format csv --account-id acct_123

# 4. Generate drafts (offline LLM; swap to real LLM by removing --llm offline
#    and configuring EXTRACTED_CAMPAIGN_LLM_*)
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 --limit 10 --llm offline

# 5. Export
python scripts/export_extracted_campaign_drafts.py \
  --account-id acct_123 --format csv --output drafts.csv
```

CLI inventory:

| Runner | Purpose | Standalone? |
|---|---|---|
| `run_extracted_content_pipeline_migrations.py` | Apply schema | Yes |
| `load_extracted_campaign_opportunities.py` | Import opportunities (JSON/CSV) | Yes |
| `run_extracted_campaign_generation_example.py` | Offline file-driven generation | Yes |
| `run_extracted_campaign_generation_postgres.py` | DB-backed generation | Yes (with `--llm offline` or real LLM) |
| `export_extracted_campaign_drafts.py` | Read-only draft export | Yes |

## The three reasoning layers

| Layer | Purpose | Status |
|---|---|---|
| `CampaignReasoningContextProvider` port (`campaign_ports.py:171-180`) | Optional Protocol the host implements to inject reasoning into generation | Defined and working; reference adapter `FileCampaignReasoningContextProvider` covers the file-backed case |
| `extracted_reasoning_core` evaluators (`api.py`) | Score / gate / compute over pre-existing reasoning state. `score_archetypes`, `evaluate_evidence`, `build_temporal_evidence` | Implemented and exported |
| `extracted_reasoning_core` producers (`api.py`) | Generate reasoning from raw inputs. `run_reasoning`, `continue_reasoning`, `check_falsification`, `build_narrative_plan` | All four raise `NotImplementedError` |

And the wiring between the layers:

| Wiring | Status |
|---|---|
| `extracted_content_pipeline.reasoning.archetypes` re-exports `extracted_reasoning_core.archetypes` | Done |
| `extracted_content_pipeline.reasoning.wedge_registry` re-exports `extracted_reasoning_core.api` | Done |
| `campaign_generation.py` actually calls those evaluators | Zero call sites |
| Adapter that packages `extracted_reasoning_core` output into a `CampaignReasoningContextProvider` instance | Does not exist |
| Branch in flight building any of the above | None — `git branch -a \| grep -i reason` returns nothing |

## What the generator expects (the input shape)

`CampaignReasoningContext` (defined in `campaign_ports.py:53-100`):

| Field | Cardinality | Semantic |
|---|---|---|
| `anchor_examples` | `Mapping[label, rows]` | Labeled evidence rows for proof anchors |
| `witness_highlights` | up to ~5 | Quote/witness rows that ground copy |
| `reference_ids` | `Mapping[type, ids]` | Source ids for auditability |
| `top_theses` | up to 2 | Top strategic theses for the opportunity |
| `account_signals` | up to 2 | Account-level buying / churn signals |
| `timing_windows` | up to 2 | Why-now windows (renewal, anchor, urgency) |
| `proof_points` | up to 2 | Compact metrics (label, value, interpretation) |
| `coverage_limits` | up to 3 | Explicit caveats |
| `canonical_reasoning` | dict | Raw reasoning fields: wedge, confidence, summary, why_now, primary_driver, recommended_action |
| `scope_summary` | dict | Metadata about evidence scope |
| `delta_summary` | dict | What changed since prior snapshot |

Any reasoning producer that wants to plug in must produce values that
normalize into this shape. Reference example payload:
`extracted_content_pipeline/examples/campaign_reasoning_context.json`.

## Runtime behavior under different reasoning conditions

| Configuration | Behavior | Quality |
|---|---|---|
| `reasoning_context = None` (default) | Generator falls back to `normalize_campaign_reasoning_context` over the opportunity row itself; only fields embedded in the source data flow into the prompt | Lowest. Drafts are generic. |
| `FileCampaignReasoningContextProvider` (file-backed) | Loads pre-baked reasoning JSON keyed by target_id / company / email / vendor; normalized and threaded into the prompt | High when the source file is well-built. Quality scales with effort the host put into producing the JSON. |
| `SinglePassCampaignReasoningProvider` | Calls the configured LLM once per opportunity with the packaged reasoning prompt; normalized and threaded into the campaign prompt | Medium. Better than no reasoning, but no multi-hop planning, cache, or falsification. |
| Custom `CampaignReasoningContextProvider` (e.g. real producer) | Provider is called once per opportunity; output normalized and threaded in | High. This is the architecturally intended path. |

## What "add a source and reason over it" costs, by tier

The architecture supports three honestly distinct tiers of reasoning.
Each lands the buyer at "source in, drafts out":

### Tier 1: Single-pass prompt reasoning

- Implemented in `extracted_content_pipeline/services/single_pass_reasoning_provider.py`
- One LLM call per opportunity that takes the source row + context and
  produces the `CampaignReasoningContext` shape directly via a
  structured-output prompt
- Implements the `CampaignReasoningContextProvider` Protocol
- Matches the single-pass pattern used elsewhere in the package
- Quality matches what most current "AI content ops" tools actually do

Trade-off: no multi-step planning, no falsification, no cache. If the
LLM gets it wrong on the first call, the campaign draft inherits the
mistake.

### Tier 2: Implement `extracted_reasoning_core` producer stubs

- ~2-3 weeks of work, ~2,000 LOC
- Fill in `run_reasoning`, `continue_reasoning`, `check_falsification`,
  `build_narrative_plan` in `extracted_reasoning_core/api.py`
- Plus an adapter that packages those outputs into a
  `CampaignReasoningContextProvider`
- Multi-step LLM with state, falsification gating, semantic cache
- This is what the architecture intended when the producer stubs were
  defined

Trade-off: real reasoning, but you're committing to ~3 weeks of work
before the primary product ships meaningfully better drafts than the
single-pass version.

### Tier 3: Extract the atlas_brain producer

- ~3-4 weeks of work, ~16,000 LOC + Neo4j
- Bring `b2b_reasoning_synthesis.py` (3,903 LOC) +
  `b2b_churn_intelligence.py` (5,779 LOC) +
  `atlas_brain/reasoning/` (~5,000 LOC across 17 files) into the
  extraction
- Includes Neo4j knowledge graph, event bus orchestration, entity
  locks, semantic cache
- Contradicts the documented Option B decision in
  `remaining_productization_audit.md:338-343` (reasoning is
  host-owned)

Not recommended. Captured here for completeness; the cost-benefit on
Tier 3 only makes sense if the goal is to fold the reasoning engine
into the content product as a single SKU.

## Decision points after Tier 1

Tier 1 is now implemented for B2B campaign opportunities. Remaining choices are
about deeper reasoning, not basic "source row in, reasoned draft out":

1. **Tier 2 scope.** Decide whether to fill the four
   `extracted_reasoning_core` producer stubs for multi-pass planning,
   falsification, and cache-aware reasoning.
2. **Additional source formats.** CRM rows are covered through
   `campaign_opportunities`; review/complaint data, episode transcripts, and
   sales call transcripts need their own schema-aware adapters.
3. **Product promise.** If single-pass meets the sold promise, AI Content Ops is
   operational. If the promise is "multi-pass refinement over your data," Tier
   2 is the floor.

## Status verdict

| Capability | State |
|---|---|
| Buyer can install the product on a fresh box | Yes |
| Buyer can apply migrations and load opportunities without atlas_brain | Yes |
| Buyer can generate drafts with a real LLM | Yes |
| Buyer can generate drafts with the offline deterministic LLM | Yes |
| Buyer can review and export drafts | Yes |
| Buyer can supply pre-baked reasoning as JSON | Yes |
| Buyer can supply a custom Python reasoning provider | Yes (port is defined) |
| The product itself produces reasoning from a source | Yes, single-pass opportunity-level reasoning only |
| `extracted_reasoning_core` produces reasoning from a source | No (4 stubs raise NotImplementedError) |

The remaining structural gap is multi-step reasoning. Tier 1 now lands the
buyer at "source in, reasoned drafts out"; Tier 2 is still required for
multi-pass planning, falsification, cache, and deeper reasoning state.

## References

- `extracted_content_pipeline/docs/host_install_runbook.md` — buyer-
  facing install runbook
- `extracted_content_pipeline/docs/reasoning_handoff_contract.md` —
  reasoning Protocol contract for hosts
- `extracted_content_pipeline/docs/remaining_productization_audit.md` —
  decision Log including the Option B decision (reasoning host-owned)
- `extracted_content_pipeline/campaign_ports.py:171-180` —
  `CampaignReasoningContextProvider` Protocol definition
- `extracted_content_pipeline/campaign_generation.py:262-297` —
  `_opportunity_with_reasoning_context` (the runtime call site)
- `extracted_reasoning_core/api.py` — evaluator + producer-stub
  surface
- `extracted_content_pipeline/campaign_reasoning_data.py` — file-backed
  reference adapter
