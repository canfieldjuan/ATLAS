# Reasoning Product Extraction Decision

Date: 2026-05-06

## Decision

Extract deeper reasoning as a separate product/provider, then wire it into
AI Content Ops and other products through explicit product adapters. Do not
embed Atlas reasoning producer internals directly inside AI Content Ops.

This supersedes any ambiguity from the current host-owned wording without
reversing the boundary: reasoning can become a packaged product, but consumer
products still consume it through ports/read models rather than importing its
internal pool, graph, synthesis, or orchestration modules.

## What pre-compressed reasoning means

Pre-compressed reasoning is a campaign-ready reasoning payload that has already
been generated, selected, summarized, and normalized before AI Content Ops sees
it. It is not raw source data and it is not the full reasoning state machine.

For AI Content Ops, the normalized target shape is
`CampaignReasoningContext`, which carries:

- `anchor_examples`: labeled evidence rows for proof anchors.
- `witness_highlights`: quote or witness rows that can ground generated copy.
- `reference_ids`: source ids for auditability.
- `top_theses`: the strongest strategic theses for the opportunity.
- `account_signals`: account-level buying or churn signals.
- `timing_windows`: why-now windows such as renewal, budget, or migration
  triggers.
- `proof_points`: compact metrics or facts the prompt can cite.
- `coverage_limits`: explicit caveats or missing-evidence warnings.
- `canonical_reasoning`: raw compact fields such as wedge, confidence,
  summary, why_now, primary_driver, and recommended_action.
- `scope_summary`: metadata about the evidence considered.
- `delta_summary`: what changed since the prior reasoning snapshot.

In other words, pre-compressed reasoning is the output of a reasoning producer,
not the producer itself.

## Verified current state

1. `extracted_content_pipeline` can consume reasoning today through
   `CampaignReasoningContextProvider` and the file-backed reference adapter.
2. `extracted_reasoning_core` exists as a separate package and already owns
   evaluator/helper surfaces such as archetypes, evidence evaluation, temporal
   evidence, tiers, state, ports, and wedge registry.
3. `extracted_reasoning_core` does not yet own end-to-end producer behavior.
   Its producer-shaped public functions (`run_reasoning`,
   `continue_reasoning`, `check_falsification`, and `build_narrative_plan`)
   are still fail-closed placeholders.
4. The older extraction decision is still directionally correct: reasoning is
   its own extracted product, not logic duplicated into every consuming product.
5. The AI Content Ops decision is also still correct: AI Content Ops should
   consume reasoning through a port and should not import Atlas producer
   internals directly.

## Product wiring model

The intended product topology is:

```text
Raw source data
  -> extracted_reasoning_product/provider
  -> product-specific adapter
  -> AI Content Ops / Competitive Intelligence / other consumers
```

AI Content Ops adapter:

```text
Reasoning output -> CampaignReasoningContextProvider -> CampaignReasoningContext
```

Competitive Intelligence adapter:

```text
Reasoning output -> typed reasoning reader/DTO -> battle card or signal views
```

UI/API adapter:

```text
Reasoning output -> explainability read model -> evidence, confidence, caveats,
proof points, and recommended actions
```

The reasoning provider may be implemented by Atlas, by the extracted reasoning
product, or by a buyer-owned engine. The consumer product should not care which
implementation is behind the adapter.

## Extraction tiers

### Tier 1: Single-pass reasoning provider

Goal: fastest UI-first product value.

Build a provider that takes one normalized source/opportunity payload, calls an
LLM once with structured output, and returns `CampaignReasoningContext`.

- Estimated lift: small.
- Best for: MVP demos, upload-a-source-and-generate flows, buyer validation.
- Trade-off: no multi-step state, graph, falsification loop, or semantic cache.

### Tier 2: Finish `extracted_reasoning_core` producers

Goal: real standalone reasoning product without directly transplanting the full
Atlas churn pipeline.

Implement the producer-shaped APIs in `extracted_reasoning_core/api.py`, then add
adapters that package `ReasoningResult` into product-specific handoff shapes.

- Estimated lift: medium.
- Best for: durable reusable reasoning product shared by multiple products.
- Trade-off: requires product-neutral state, narrative planning, falsification,
  and adapter contracts before it materially improves customer output.

### Tier 3: Extract Atlas churn reasoning producers

Goal: preserve the deepest existing churn reasoning behavior.

Bring over the Atlas producer stack, including churn intelligence pooling,
reasoning synthesis, witness/pool compression, reader contracts, prompt packs,
and supporting graph/state/runtime dependencies.

- Estimated lift: large.
- Best for: a churn/B2B reasoning SKU where the existing domain ontology is the
  product.
- Trade-off: high coupling risk, likely requires graph/runtime dependencies, and
  should remain separate from AI Content Ops even if packaged alongside it.

## Recommended sequence

1. Keep AI Content Ops UI/API work moving first. The customer-facing workflow
   should not wait for Tier 3.
2. Add a Tier 1 reasoning provider if the UI needs immediate source-to-reasoned
   draft behavior.
3. In parallel, plan Tier 2 as the reusable extracted reasoning product path.
4. Only pursue Tier 3 if product positioning requires the existing churn/B2B
   reasoning depth as a bundled SKU.
5. Wire all consuming products through adapters, not internal imports.

## UI-first implication

A UI-first AI Content Ops product should expose reasoning as a visible capability
and state, not as an invisible backend assumption.

The UI should show:

- whether reasoning is absent, uploaded, generated-lite, or generated-deep;
- the proof points and witness highlights used by the draft;
- confidence and coverage limits;
- source references or ids;
- editable draft copy with the reasoning context visible beside it.

This lets the product sell immediately with optional or lightweight reasoning,
while leaving room to upgrade customers to a deeper extracted reasoning provider.

## Next implementation options

Choose exactly one next implementation slice:

1. **UI/API first**: define the AI Content Ops API/router and UI data contracts,
   including a reasoning status/read model. This is best if buyer usability is
   the top concern.
2. **Tier 1 provider first**: add a single-pass `CampaignReasoningContextProvider`
   implementation so uploaded opportunity data can produce reasoning without an
   external engine. This is best if first demos need source-to-reasoned-draft
   behavior immediately.
3. **Tier 2 core first**: implement the first non-stub
   `extracted_reasoning_core.run_reasoning(...)` path and a campaign adapter.
   This is best if the company wants reusable reasoning to be the platform
   foundation before UI polish.

Default recommendation: take option 1 and option 2 in parallel if staffing
allows. Do not take Tier 3 Atlas extraction work before UI/customer
validation confirms that depth is a purchase driver.

## Scope guard

In scope for the next planning/implementation wave:

- Product adapters from reasoning output to consumer ports/read models.
- UI/API contracts that expose reasoning status and explainability.
- Tier 1 or Tier 2 producer work in a separate reasoning boundary.

Out of scope unless explicitly re-approved:

- Importing Atlas producer internals directly into AI Content Ops runtime code.
- Making AI Content Ops depend on Neo4j, graph state, entity locks, or event bus
  internals.
- Replacing existing campaign generation contracts while adding reasoning.
- Duplicating reasoning producer logic independently in each product.
