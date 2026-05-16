# AI Content Ops Reasoning Policy Audit

Date: 2026-05-16

## Purpose

AI Content Ops can already consume reasoning context across the generated
asset services. The remaining question is not "can it use reasoning?" It can.
The question is which reasoning depth and which policy knobs should be exposed
per content type without turning the content product into a second reasoning
monolith.

This audit is the handoff from source-adapter work into reasoning-depth work.
It is docs-only and does not change runtime behavior.

## Scope Boundary

In scope:

- AI Content Ops generated assets:
  - `email_campaign`
  - `blog_post`
  - `report`
  - `landing_page`
  - `sales_brief`
  - `signal_extraction`
- Host-facing reasoning policy for:
  - single-pass vs. multi-pass reasoning
  - continuation depth
  - falsification
  - narrative planning
  - output validation
  - per-content-type opt-in

Out of scope:

- Evidence-to-Story orchestration.
- Podcast or story-repurposing workflows.
- Extracting the full Atlas reasoning synthesis producer.
- New source-adapter breadth.

AI Content Ops and Evidence-to-Story should stay separate. Content Ops uses
reasoning to improve business content assets; Evidence-to-Story can own
story-specific state, arcs, and long-form narrative continuity.

## Current Shipped Reasoning Seams

| Surface | Current behavior | Policy implication |
|---|---|---|
| `CampaignReasoningContextProvider` | Stable port consumed by campaign, blog, report, landing page, and sales brief services. | Keep this as the product boundary. Generated assets should not import reasoning internals directly. |
| `FileCampaignReasoningContextProvider` | Hosts can provide precomputed JSON reasoning context. | Useful for buyers with their own analysis pipeline. No new policy needed. |
| DB-backed reasoning context | Hosted admin APIs can list/upsert/delete contexts and generated assets can consume them. | Operational seam is present; policy should decide when a run requires context vs. treats it as optional. |
| `SinglePassCampaignReasoningProvider` | One LLM call creates normalized campaign reasoning context. | Good default for light assets and low-budget runs. |
| `MultiPassCampaignReasoningProvider` | Calls `run_reasoning`, can chain `continue_reasoning`, and can optionally run falsification, narrative planning, and validation when constructed with typed policy objects. | Core capability exists, but the host-facing preset/config layer is still missing. |
| Control-surface catalog | LLM-backed assets are marked `optional_host_context`; `signal_extraction` is `absent`. | Catalog is honest, but too coarse for depth selection. |

## Current Config Gap

The campaign operations API exposes simple multi-pass knobs:

- `generation_multi_pass_reasoning`
- `generation_multi_pass_pack_name`
- `generation_multi_pass_top_thesis_limit`
- `generation_multi_pass_enable_chain`
- `generation_multi_pass_max_continuations`

It intentionally does not expose the richer typed objects:

- `FalsificationPolicy`
- `ReasoningPack` for narrative planning
- `OutputPolicy`
- `block_on_validation_failure`

That is the right default for now. Those richer objects are policy decisions,
not simple booleans. They need a product-level preset model before they are
safe as host-facing API fields.

## Per-Asset Policy Recommendation

| Asset | Current reasoning support | Recommended default | Rich reasoning should expose |
|---|---|---|---|
| `signal_extraction` | None; deterministic source normalization. | No reasoning. | Nothing. Keep this cheap and predictable. |
| `email_campaign` | Uses optional context in short-form prompts. | Single-pass or precomputed context by default, especially because hosted multi-pass has no cache today. | Multi-pass only when the host asks for account-specific sequence strategy. Falsification should be informational, not blocking, unless the host opts in. |
| `blog_post` | Can consume context on blueprints. | Multi-pass optional for long-form posts. | Narrative planning is the first useful rich knob. Output validation should be soft until a blog-specific policy exists. |
| `report` | Can consume context and report sections mirror `NarrativePlan.sections`. | Multi-pass preferred for reports. | Narrative planning plus output validation. Falsification can block only when the report is customer-facing and policy is explicit. |
| `landing_page` | Can consume campaign-level context. | Single-pass or precomputed context by default. | Validation should focus on quality-pack gates; narrative planning only for host-judged evidence-heavy pages, such as pages built around multiple references or proof sections. No automatic evidence-heavy gate in v1. |
| `sales_brief` | Can consume context and sections mirror report sections. | Multi-pass preferred for sales/account briefs. | Narrative planning plus optional falsification. Blocking validation is useful for high-stakes account briefs. |

## Depth Presets

Use named presets instead of exposing every low-level field directly.

| Preset | Intended use | Suggested behavior |
|---|---|---|
| `none` | Deterministic extraction or cheapest generation. | No provider. |
| `context_only` | Host already has reasoning JSON or DB context. | Use provided context; no generated reasoning. |
| `single_pass` | Fast enrichment for short-form assets. | Use `SinglePassCampaignReasoningProvider`. |
| `multi_pass_light` | Better reasoning without heavy policy. | Use `MultiPassCampaignReasoningProvider` with continuation chain and bounded max continuations. |
| `multi_pass_structured` | Reports, briefs, and long-form blog posts. | Multi-pass plus narrative planning pack. Validation report is surfaced but not blocking by default. |
| `multi_pass_strict` | High-stakes reports or account briefs. | Multi-pass plus narrative planning, falsification, output validation, and optional blocking on validation failure. |

The first runtime slice should implement only the preset vocabulary and
resolution rules. It should not wire every policy object at once.
Presets should stay fixed in the product API; hosts that need fine-grained
overrides can keep constructing a custom `MultiPassCampaignReasoningProvider`
and pass it through the existing provider port.

## Falsification Policy

Falsification is valuable, but it is also expensive and can be too aggressive
for short copy.

Recommendation:

- Do not enable falsification by default for `email_campaign` or
  `landing_page`.
- Surface falsification results informationally for `blog_post`, `report`, and
  `sales_brief` before blocking output.
- Treat informational falsification as unavailable in product presets until the
  validation-metadata surfacing slice gives operators a place to see the
  result.
- Allow strict blocking only through an explicit `multi_pass_strict` preset or
  host-constructed provider.
- Preserve the current fail-soft per-claim behavior in
  `MultiPassCampaignReasoningProvider`; one failed falsification call should
  not collapse the entire generation run.

## Narrative Planning Policy

Narrative planning is the clearest immediate value for Content Ops.

Recommendation:

- First rich-policy wiring should target `report` and `sales_brief`, because
  their section shapes already mirror `NarrativePlan.sections`.
- `blog_post` should follow after reports/briefs. It needs a blog-specific pack
  so sections do not become generic report sections.
- `landing_page` should stay opt-in. Landing pages often need conversion
  hierarchy more than reasoning hierarchy.
- `email_campaign` should not use narrative planning by default.

## Output Validation Policy

Quality packs already validate generated assets after the LLM call. Reasoning
core `OutputPolicy` is a different layer: it validates the reasoning result
before it becomes prompt context.

Recommendation:

- Keep asset quality packs as the primary output guard.
- Use `OutputPolicy` to validate reasoning claims before they are handed to
  long-form assets.
- Default to surfacing validation metadata, not blocking, until each content
  type has a concrete policy.
- Enable blocking first for `report` and `sales_brief`, not for short-form
  email copy.

## Cache And State Policy

Semantic cache and state store are ports on `ReasoningPorts`, but the hosted
campaign operations helper currently constructs `ReasoningPorts(llm=llm)` for
its simple multi-pass path. That means no semantic cache and no state store on
the default hosted toggle.

Recommendation:

- Keep cache/state host-owned for now.
- Do not add hidden storage behind the Content Ops router.
- Add capability/status reporting before adding cache/state configuration.
- Treat stateful reasoning as a separate host wiring step, not a default
  behavior of every generated asset.
- Operators should expect multi-pass cost to repeat per run until cache/state
  are wired host-side.

## Recommended Next Slices

| # | Slice | Estimated LOC | Notes |
|---|---|---:|---|
| 1 | Reasoning preset catalog (docs + pure config) | ~150 | Add a small `reasoning_policy` module or config table that maps asset type + preset to allowed depth and policy intent. No runtime provider construction beyond the current behavior yet. |
| 2 | Report and sales brief structured reasoning | ~300-400 | Use the preset catalog to construct a narrative-plan-enabled multi-pass provider for report and sales brief execution when explicitly requested. |
| 3 | Validation metadata surfacing | ~200-300 | Ensure generated asset metadata exposes reasoning validation reports consistently for report and sales brief. |
| 4 | Strict mode only after metadata is visible | ~150-250 | Add blocking validation only after operators can see why a draft was skipped. |
| 5 | Blog-specific narrative pack | ~250-400 | Add a blog pack later, after the report and sales brief path proves the policy shape. |

## Non-Goals

- Do not move `atlas_brain` reasoning synthesis wholesale into AI Content Ops.
- Do not make every content type use multi-pass reasoning.
- Do not add source adapters speculatively.
- Do not collapse Evidence-to-Story into Content Ops.

## Verdict

AI Content Ops is operational outside Atlas for generated-asset production and
can consume reasoning today. The next value step is not more source breadth or
another custom adapter. It is a small, explicit reasoning policy layer that
lets hosts choose depth per asset while keeping heavy reasoning opt-in.
