# Churn Signals Product Context Pack

## Purpose

Use this document when prompting a model to propose end products, product UIs, or product workflows for the Churn Signals product line.

This is not a marketing brief. It is a grounded product and system context pack. Any proposed product should be constrained by the actual pipeline, data objects, UI surfaces, latency profile, and operating costs described here.

---

## Scope

This pack is intentionally scoped to the B2B churn and displacement intelligence business.

For this prompting workflow, ignore:

- voice assistant and telephony features
- personal automation workflows
- Amazon and consumer-review intelligence products
- broader strategic-intelligence experiments that are not part of Churn Signals product delivery

Focus only on:

- B2B churn intelligence
- competitive displacement intelligence
- report, campaign, battle-card, and briefing products
- analyst and operator tooling that directly supports those outputs

Relevant code and product surfaces in this repo:

- `atlas_brain/`
- `atlas-churn-ui/`
- `atlas-intel-next/`
- selected B2B pages in `atlas-intel-ui/`

---

## Churn Signals Pipeline Model

Churn Signals works as a staged B2B intelligence pipeline. For product planning, the important distinction is:

- raw inputs
- normalized intermediate objects
- synthesized artifacts
- operator telemetry

### Raw Inputs

Primary B2B sources currently used in active downstream selection:

- `g2`
- `gartner`
- `getapp`
- `peerspot`
- `reddit`
- `github`
- `hackernews`
- `stackoverflow`
- `slashdot`

Deprecated from active governed use:

- `capterra`
- `software_advice`
- `trustpilot`
- `trustradius`

### Core B2B Stages

1. Review ingest and scrape/import
2. Enrichment
3. Repair / strategic adjudication for weak or contradictory rows
4. Evidence and witness derivation
5. Vendor reasoning synthesis
6. Cross-vendor reasoning synthesis
7. Downstream artifact generation

### Canonical Intermediate Objects

These are the most important reusable product primitives:

- `review`
  - raw source text, vendor name, source, timestamps, rating, metadata
- `enrichment`
  - extracted pain categories, urgency, competitor mentions, churn and buying signals, company and title clues, pricing and timeline fields, evidence spans
- `witness primitives`
  - structured evidence-ready fields such as replacement mode, org pressure, timeline, productivity claims, and decision context
- `reasoning synthesis`
  - vendor-level reasoning contracts built from evidence pools and witness packs
- `cross_vendor_reasoning`
  - pairwise and market-level synthesis for competition, asymmetry, and displacement
- `artifact_attempt`
  - quality and persistence attempts for generated downstream artifacts
- `task_execution`
  - scheduler execution record with result payload
- `llm_usage`
  - local traced model-call cost, token, provider, operation, run, and attribution data

### Core Downstream Artifacts

These are real Churn Signals outputs today:

- churn reports
- report detail pages
- vendor briefings
- battle cards
- campaigns and sales sequences
- blog posts
- product profiles
- vendor snapshots and change events

---

## What Churn Signals Already Produces

### Customer-Facing or Analyst-Facing Product Outputs

Churn Signals already supports surfaces around:

- vendor health and churn risk
- review exploration and enrichment detail
- competitive displacement
- report generation and report review
- campaign generation and campaign review
- blog generation and blog review
- vendor targeting and prospect workflows
- battle-card style competitive enablement

### Operator Outputs

Churn Signals also has internal/operator products, especially:

- pipeline operations
- run detail
- cost and burn visibility
- cache health
- provider reconciliation
- task health
- quality and efficiency metrics

---

## Current UI Surface Inventory

### B2B Churn / Sales Intelligence UI

Current pages include:

- `Dashboard`
- `Vendors`
- `VendorDetail`
- `Reviews`
- `ReviewDetail`
- `Reports`
- `ReportDetail`
- `CampaignReview`
- `BriefingReview`
- `BlogReview`
- `CampaignDiagnostics`
- `BlogDiagnostics`
- `VendorTargets`
- `Prospects`
- `Leads`
- `Challengers`
- `Affiliates`
- `PipelineReview`

These are primarily internal, analyst-facing, or power-user B2B surfaces today, though several are close to customer-facing product surfaces.

### Next.js B2B UI Migration

The Next.js app mirrors major B2B product routes:

- dashboard
- vendors
- vendor detail
- reviews
- reports
- campaign review
- briefing review
- blog review
- leads and prospects
- onboarding and account flows

Treat this as the direction of travel for product delivery, not a separate data product.

### Operations UI

The Operations surface already includes:

- run detail
- costs summary
- burn dashboard
- provider reconciliation
- cache health
- generic reasoning breakdown
- legacy reasoning visibility
- B2B efficiency views

That means Churn Signals already has strong internal telemetry surfaces. New product concepts should reuse those observability capabilities where useful.

---

## What Is Real-Time vs Batch

This matters for product design.

### Real-Time or Near-Real-Time Friendly

- review browsing and search
- vendor dashboards built on already-computed aggregates
- review detail and enrichment detail
- operator monitoring
- trigger-based alerts when downstream aggregates are already available
- CRM and event-driven reasoning incident views

### Batch or Scheduled

- broad enrichment sweeps
- repair passes
- vendor reasoning synthesis
- cross-vendor synthesis
- campaign generation
- battle cards
- blog generation
- many report-generation workflows
- historical backfills

### Mixed

Some Churn Signals products should read from batch-computed artifacts but feel real-time in UI:

- vendor intelligence feeds
- displacement maps
- risk score pages
- campaign opportunity queues
- watchlists and saved views

This is usually the right product pattern for Churn Signals: precompute expensive intelligence, then serve interactive exploration on top of the resulting artifacts.

---

## Cost, Latency, and Ops Constraints

Any product proposal should respect these constraints.

### Cost Reality

Churn Signals now tracks:

- per-call cost and token usage
- burn per job
- provider reconciliation
- generic reasoning by source and event type
- B2B cost per witness and per run

The system is observability-aware, but model spend is still a real product constraint.

### Caching and Reuse

Churn Signals uses multiple reuse layers:

- evidence-hash skip
- exact request cache
- semantic cache on some reasoning paths
- provider prompt cache

Products that can read from cached or already-synthesized artifacts are much cheaper than products that trigger fresh generation every time.

### Batching

Batching is now relevant for some Anthropic-backed paths, especially:

- campaign generation
- scorecard-style narrative generation

But batching is not universal and should not be assumed for latency-sensitive paths.

### Live Enrichment Constraint

Live enrichment should remain direct and concurrent, not batch-polled. Product ideas that rely on immediate enrichment of large volumes of fresh rows should be treated as expensive and operationally risky.

### Budget Pressure

Cross-vendor reasoning already experiences token-budget rejection pressure. Any product concept that leans heavily on frequent cross-vendor synthesis should explicitly account for budget caps and fallback behavior.

---

## Data Trustworthiness and Sparsity

The model should distinguish high-trust from lower-trust fields.

### Stronger / More Mature Signals

- review text and provenance
- source and vendor attribution
- enriched pain and complaint themes
- competitor and displacement signals
- witness-ready evidence spans
- vendor-level reasoning artifacts
- run and cost telemetry

### Useful but Variable Signals

- reviewer identity fields
- company and title clues
- budget and timeline anchors
- productivity claims
- org-pressure type
- some source-specific extraction fields

### Product Planning Rule

If a proposed end product depends heavily on sparse fields, it should be framed as:

- analyst assist
- confidence-tiered workflow
- human review queue
- or premium/exploratory product

It should not be framed as a fully automated, high-trust primary surface.

---

## Existing Product Patterns Churn Signals Already Supports

### Analyst and Operator Tools

- monitoring dashboards
- QA and review workflows
- report review and approval
- cost and burn oversight
- run-level diagnostics

### Customer-Facing B2B Intelligence Products

- vendor watchlists
- churn-risk feeds
- displacement and competitor monitoring
- vendor benchmark pages
- account targeting and campaign recommendations
- sales enablement packs
- evidence-backed report subscriptions

### API / Platform Products

- webhook feeds
- CRM sync
- exportable reports
- intelligence feeds
- alert subscriptions

---

## What A Model Should Not Assume

Do not let the model assume Churn Signals has:

- perfect real-time synthesis for every page load
- universal low-latency batch APIs across all providers
- unlimited token budget for cross-vendor reasoning
- trustworthy dense company identity on every review
- full human labeling infrastructure
- a clean single-surface customer product today

Also do not let the model propose ideas that require:

- deleting current operator workflows
- inventing missing canonical objects
- assuming deprecated sources are active again
- relying on hidden provider data not stored locally

---

## Product-Planning Heuristics For Churn Signals

When proposing product UIs on top of Churn Signals, prefer ideas that:

1. Read from already-computed intelligence artifacts
2. Reuse witness-backed evidence and citations
3. Have clear unit economics
4. Can be delivered as watchlists, feeds, reports, review queues, or enablement surfaces
5. Separate exploratory or low-confidence insight from high-confidence action surfaces

Be more skeptical of ideas that:

1. Require constant re-synthesis on interaction
2. Depend on sparse reviewer or company fields
3. Need cross-vendor reasoning on every user interaction
4. Depend on deprecated or suppressed sources
5. Have no obvious path to cost visibility or operator review

---

## Suggested Framing For End-Product Ideation

When using this context pack, ask the model to produce ideas across three buckets:

1. Operator/internal products
2. Customer-facing B2B products
3. Platform/API products

And have it rank ideas by:

- leverage
- data readiness
- trustworthiness
- implementation difficulty
- operating cost

---

## Minimal Data Objects The Model Should Understand

If you provide sample payloads, prioritize these:

- one enriched review
- one witness-backed review
- one vendor reasoning synthesis
- one cross-vendor reasoning payload
- one campaign opportunity payload
- one persisted report or battle-card payload
- one burn-dashboard row
- one run-detail payload

Those examples are more useful than dumping full schema files.

See also:

- `docs/product_context_samples.md`

---

## Final Instruction To The Model

Propose products that treat Churn Signals as:

- an evidence-backed intelligence system
- with expensive synthesis stages
- strong operator telemetry
- mixed real-time and batch behavior
- and multiple B2B product surfaces already in flight

The best ideas will usually turn existing churn, displacement, reasoning, campaign, and report artifacts into better productized workflows, not invent entirely new computation that Churn Signals does not already support.
