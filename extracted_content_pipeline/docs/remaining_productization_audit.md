# Remaining Productization Audit

Date: 2026-05-02
Current-state refresh: 2026-05-17

This audit follows the helper ownership pass that moved `_b2b_batch_utils`,
`_blog_matching`, `_campaign_sequence_context`, and `campaign_audit` into the
extracted content pipeline boundary.

## Current State

Standalone import debt is clean:

```bash
EXTRACTED_PIPELINE_STANDALONE=1 python scripts/audit_extracted_standalone.py --fail-on-debt
# Atlas runtime import findings: 0
```

The extracted runner was green after the helper pass:

```bash
EXTRACTED_PIPELINE_STANDALONE=1 bash scripts/run_extracted_pipeline_checks.sh
# 177 passed
```

The current smoke import script is manifest-driven: it imports every Python
target listed in `manifest.json`, excluding migrations and package
`__init__.py` files. The copied task modules that were originally blockers now
import in standalone mode through product-owned seams.

| Module | Standalone import | First blocker |
| --- | --- | --- |
| `autonomous.tasks.b2b_campaign_generation` | Passes | PR 2 seams present |
| `autonomous.tasks.b2b_vendor_briefing` | Passes | PR 1 seams present |
| `autonomous.tasks._b2b_pool_compression` | Passes | Product-owned witness/compatibility seams present |
| `autonomous.tasks.competitive_intelligence` | Passes | Product-owned brand registry seam present |

Many manifest-mapped files remain Atlas-shaped references. Passing import does
not make those whole files product-owned; production paths should continue
moving behavior into native extracted modules instead of expanding ownership of
copied task files.

## Remaining Mapped Python Surface

| File | Lines | Classification |
| --- | ---: | --- |
| `_b2b_shared.py` | 19,875 | Monolith; split by required product seams |
| `b2b_blog_post_generation.py` | 9,613 | Blog product surface, not campaign-core |
| `b2b_campaign_generation.py` | 6,043 | Campaign-core copied task; importable through product seams |
| `b2b_vendor_briefing.py` | 3,222 | Campaign/email adjacent; importable through product seams |
| `_b2b_pool_compression.py` | 2,319 | Upstream reasoning pool; deliberately outside campaign-core |
| `_b2b_reasoning_contracts.py` | 1,773 | Reasoning policy; importable but large |
| `_b2b_synthesis_reader.py` | 1,767 | Reasoning read model; importable but large |
| `blog_post_generation.py` | 1,758 | Consumer/blog sidecar |
| `competitive_intelligence.py` | 1,455 | Consumer intelligence sidecar; importable through compatibility seams |
| `_b2b_cross_vendor_synthesis.py` | 1,063 | Reasoning/synthesis helper; importable |
| `_b2b_specificity.py` | 772 | Specificity policy; importable |
| `complaint_analysis.py` | 527 | Consumer/complaint sidecar |
| `complaint_enrichment.py` | 493 | Consumer/complaint sidecar |
| `article_enrichment.py` | 431 | Consumer/blog sidecar |
| `complaint_content_generation.py` | 348 | Consumer/complaint sidecar |

## Import Seams Closed

`b2b_campaign_generation.py` had these missing top-level dependencies, now
covered by product-owned compatibility seams:

- `services.b2b.account_opportunity_claims`
- `services.campaign_reasoning_context`
- `services.campaign_quality`
- `services.vendor_target_selection`
- `autonomous.visibility`
- `autonomous.tasks.campaign_suppression`

`b2b_vendor_briefing.py` had these missing top-level dependencies, now covered
by product-owned compatibility seams:

- `services.campaign_sender`
- `services.vendor_target_selection`
- `templates.email.vendor_briefing`
- `autonomous.tasks.campaign_suppression`

`_b2b_pool_compression.py` was originally left outside campaign-core ownership.
It now imports in standalone mode through product-owned compatibility seams,
but remains upstream reasoning infrastructure rather than a campaign-core
runtime dependency.

- `autonomous.tasks._b2b_witnesses`

`competitive_intelligence.py` now imports in standalone mode through the
product-owned brand registry seam, but remains a consumer-intelligence sidecar
rather than part of the email/campaign runtime.

- `services.brand_registry`

## Productization Interpretation

The sellable campaign product already has a cleaner product-owned spine:

- `campaign_ports.py`
- `campaign_generation.py`
- `campaign_send.py`
- `campaign_sequence_progression.py`
- `campaign_suppression.py`
- `campaign_sender.py`
- `campaign_webhooks.py`
- `campaign_analytics.py`
- `campaign_postgres.py`
- `campaign_llm_client.py`

The copied Atlas task files are still useful as extraction references, but the
next work should not make `b2b_campaign_generation.py` or `_b2b_shared.py`
product-owned in one move. The correct next move is to add narrow compatibility
seams that let the copied campaign task modules import, then migrate behavior
into the product-owned spine in smaller slices.

## Recommended Sequence

### PR 1: Vendor Briefing Import Seams

Status: implemented. `b2b_vendor_briefing.py` imports in standalone mode, and
the smoke script now covers it.

Goal: make `b2b_vendor_briefing.py` import in standalone mode without claiming
the copied 3,222-line task is product-owned.

Add minimal product-owned or compatibility modules:

- `services/vendor_target_selection.py`
  - Start with `dedupe_vendor_target_rows(...)`.
  - Keep it deterministic and data-only.
- `autonomous/tasks/campaign_suppression.py`
  - Compatibility wrapper over product `campaign_suppression.py`.
  - Provide copied task names `is_suppressed(...)`,
    `assign_recipient_to_sequence(...)`.
- `services/campaign_sender.py`
  - Compatibility wrapper around product `campaign_sender.create_campaign_sender`.
- `templates/email/vendor_briefing.py`
  - Extract the email renderer or provide a compatibility shim if the full
    template is not required for import-only readiness.

Acceptance criteria:

- `EXTRACTED_PIPELINE_STANDALONE=1 python -c "import extracted_content_pipeline.autonomous.tasks.b2b_vendor_briefing"`
- Local tests for each compatibility shim.

This slice may expand the smoke script to include `b2b_vendor_briefing.py`
once the direct import passes. Keep `b2b_campaign_generation.py` out of the
smoke list until PR 2 handles its separate missing imports.

### PR 2: Campaign Generation Import Seams

Status: implemented. `b2b_campaign_generation.py` imports in standalone mode,
and the smoke script now covers both campaign-core copied tasks.

Goal: make `b2b_campaign_generation.py` import in standalone mode without
claiming the copied 6,043-line task is product-owned.

Add:

- `services/b2b/account_opportunity_claims.py`
  - `account_opportunity_source_review_count(...)`
  - `build_account_opportunity_claim(...)`
  - `serialize_product_claim(...)`
- `services/campaign_reasoning_context.py`
  - `campaign_reasoning_scope_summary(...)`
  - `campaign_reasoning_atom_context(...)`
  - `campaign_reasoning_delta_summary(...)`
- `services/campaign_quality.py`
  - `campaign_quality_revalidation(...)`
- `autonomous/visibility.py`
  - Route `emit_event(...)` / `record_attempt(...)` through
    `pipelines.notify` or safe no-op defaults.

After this PR, expand `scripts/smoke_extracted_pipeline_imports.py` to include:

- `extracted_content_pipeline.autonomous.tasks.b2b_campaign_generation`
- `extracted_content_pipeline.autonomous.tasks.b2b_vendor_briefing`

Acceptance criteria:

- `EXTRACTED_PIPELINE_STANDALONE=1 python -c "import extracted_content_pipeline.autonomous.tasks.b2b_campaign_generation"`
- `EXTRACTED_PIPELINE_STANDALONE=1 python -c "import extracted_content_pipeline.autonomous.tasks.b2b_vendor_briefing"`
- New tests for each seam's local behavior.
- Smoke import script expanded to include both modules.

### PR 3: Pool Compression Decision

Status: implemented. `_b2b_pool_compression.py` remains outside the campaign
product boundary. The product-owned generator now accepts normalized
host-provided reasoning context through `CampaignReasoningContextProvider`
instead.

Goal: decide whether `_b2b_pool_compression.py` is part of the sellable campaign
product or a reasoning-product dependency.

The smoke script includes campaign-core modules after PR 2, but intentionally
does not add `_b2b_pool_compression.py`:

- `_b2b_pool_compression` is upstream/host-owned reasoning infrastructure.
- Campaign-core accepts already-compressed `anchor_examples`,
  `witness_highlights`, `reference_ids`, `account_signals`, `timing_windows`,
  and `proof_points` instead of importing Atlas compression internals.

Options:

1. Extract a minimal `_b2b_witnesses.py` compatibility helper and keep
   `_b2b_pool_compression.py` importable as a copied reasoning dependency.
2. Keep pool compression out of campaign-core and add a product-owned interface
   that accepts already-compressed reasoning/witness context from the host.

Recommendation: option 2 unless a campaign-core test requires pool compression
directly. It keeps the campaign product from owning the whole B2B reasoning
stack.

Implemented contract:

- `campaign_ports.CampaignReasoningContext`
- `campaign_ports.CampaignReasoningContextProvider`
- `services.campaign_reasoning_context.normalize_campaign_reasoning_context(...)`
- `services.campaign_reasoning_context.campaign_reasoning_context_metadata(...)`

`CampaignGenerationService` can now accept an optional
`reasoning_context=CampaignReasoningContextProvider` dependency. The provider
receives `(scope, target_id, target_mode, opportunity)` and returns compressed
context for the prompt and draft metadata. With no provider configured, embedded
context already present on the opportunity row is normalized defensively.

## Explicit Deferrals

These remain outside the immediate campaign-core import path:

- `competitive_intelligence.py`
  - Consumer intelligence sidecar, not required for the email/campaign product
    runtime even though it is now importable.
- Consumer/blog generation tasks
  - `blog_post_generation.py`, `article_enrichment.py`,
    `complaint_*` modules.
- Whole-file ownership of `_b2b_shared.py`
  - Too large to own as a unit; split only the helpers needed by product tests.
- Whole-file ownership of `b2b_blog_post_generation.py`
  - Blog/content product, separate from the campaign delivery product.

## Current Next Slice Guidance

With the campaign-core import path and reasoning-context boundary settled, the
next slice should be driven by a concrete runtime gap: a real host export, a
failed install/readiness check, or a product-owned generator behavior that still
requires the copied Atlas task path. Do not keep expanding import coverage
sideways just because a copied task exists.

Previously recommended producer-flow migration has landed: the native
`CampaignGenerationService` expands one normalized opportunity into configured
channels such as `email_cold` and `email_followup`, passes generated cold-email
context into the follow-up prompt, and saves drafts through
`CampaignRepository`.

Future behavior migrations should satisfy the same boundary:

- Use `IntelligenceRepository.read_campaign_opportunities(...)` as the source.
- Optionally enrich with `CampaignReasoningContextProvider`.
- Generate and persist drafts through `CampaignRepository`.
- Keep `_b2b_pool_compression.py`, `_b2b_witnesses.py`, and `_b2b_shared.py`
  out of the product-owned path.

## Reasoning Producer Gap (logged 2026-05-03)

The extraction has the *consumers* of reasoning (`reasoning/archetypes.py`,
`evidence_engine.py`, `temporal.py`, plus `services/campaign_reasoning_context.py`
which normalises pre-baked input). The *producers* - the LLM-driven engines
that actually generate the reasoning context - never made it across. The
decision to keep `_b2b_pool_compression.py` and `_b2b_witnesses.py` outside the
product boundary is documented above; the larger producer surface was
implicitly left behind without a documented rationale. The multi-channel
producer slice above improves the product-owned campaign flow, but it does not
change this reasoning boundary.

### What's absent

| Source file | Lines | Role | In extracted? |
| --- | --- | --- | --- |
| `atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py` | 3,903 | Main LLM-based reasoning generator. Produces vendor / displacement / category / account reasoning contracts that the campaign generator later consumes. | No |
| `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py` | 5,779 | Pools raw review / signal / temporal data and seeds the synthesis above. | No |
| `atlas_brain/reasoning/` directory | ~5,000 across 17 files | Knowledge graph (Neo4j), narrative chains, ecosystem health, multi-hop graph walks, semantic cache, falsification rules, entity locks, market patterns, event bus, agent orchestrator, trigger events, cross-vendor selection, multi-tier cache. | None of it. |

The extracted package's three reasoning files (`archetypes.py`,
`evidence_engine.py`, `temporal.py`, ~1,400 lines combined) are all evaluators
of pre-existing reasoning state, not generators of it.

### Why this matters

The extraction made an architectural choice - probably implicit - that
**reasoning is a separate sellable product, not part of the content-ops
product**. Campaign generation accepts pre-baked reasoning *as input* via
the `CampaignReasoningContextProvider` port. The host (atlas_brain or
equivalent) is expected to produce that input.

This makes the extracted package effectively unable to run as a standalone
product *for the B2B campaign use case* unless the buyer either:

1. Brings their own reasoning engine and pipes output through the provider port, OR
2. Operates with `reasoning_context = None` and accepts the
   defensive-normalisation fallback (dramatically lower output quality), OR
3. Bundles atlas_brain alongside it (which defeats the extraction).

### Implication for the parked product directions

- **Podcast repurposing** — unaffected. Episode-to-assets is a single-pass
  transformation; no reasoning engine needed. The episode itself is the
  compressed context.
- **Long-form creative stories** — needs a reasoning generator built from
  scratch (worldbuilding planner, character-arc tracker, plot coherence
  reasoner). The B2B reasoning machinery wouldn't transfer even if extracted;
  the domain shape is wrong.
- **B2B campaign generation as a sellable product** — has a structural hole.
  Either the reasoning engine joins the extraction, or the product positioning
  explicitly says "bring your own reasoning."

### Two options for closing the gap

**Option A - Extract the reasoning producer.**

Bring `b2b_reasoning_synthesis.py`, `b2b_churn_intelligence.py`, and the
`atlas_brain/reasoning/` directory into the extraction following the same
scaffold pattern (manifest + sync + validate + smoke). Multi-week effort;
probably its own PR series. Decide first whether reasoning is part of the
content-ops product or its own sellable.

Pros: standalone product becomes self-sufficient for the B2B use case.
Cons: large surface area, Neo4j dependency, event-bus orchestration, and the
producer surface has its own atlas_brain coupling that needs to be unwound.

**Option B - Document the host-owned-reasoning contract explicitly.**

Add a buyer-facing note that the standalone product accepts pre-baked
reasoning and provide a reference adapter contract spec for what shape that
input must take. Smaller scope, but it accepts the architectural choice
that reasoning is a separate concern.

Pros: keeps the extracted product narrowly scoped and shippable.
Cons: limits the addressable market to buyers who already have reasoning
infrastructure or are willing to build one.

### Decision

Use Option B for AI Content Ops. Reasoning is a separate product/host concern,
and the content package consumes already-compressed context through
`CampaignReasoningContextProvider`. The explicit handoff contract is now
documented in `reasoning_handoff_contract.md`.

This keeps AI Content Ops focused on generation, sequencing, delivery, and
customer-data adapters while allowing Atlas, an extracted reasoning product, or
a buyer-owned engine to feed the same context shape.

For the two parked product directions captured in the strategy docs in this
folder, Option B is sufficient for the podcast repurposing offer (no
reasoning needed at all) and insufficient for either creative-content or
B2B-campaign-as-product offers unless a reasoning provider is bundled,
integrated, or supplied by the buyer.

## Refresh Log

### 2026-05-17

- Updated the stale import-state table after the manifest-driven standalone
  smoke confirmed `_b2b_pool_compression.py` and `competitive_intelligence.py`
  now import successfully.
- Reframed the next-slice guidance away from sideways import coverage and
  toward concrete runtime gaps.
- Kept the host-owned reasoning decision unchanged.
