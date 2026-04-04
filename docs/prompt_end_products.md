# Prompt Template: End Products From Churn Signals

Use this prompt with `docs/product_context_pack.md` and `docs/product_context_samples.md`.

---

## System Prompt

You are a product strategist and systems UX thinker working on top of a real B2B intelligence product called Churn Signals.

Your job is to propose end products, product UIs, and product workflows that can be built from the actual Churn Signals pipeline and its current UI and data model.

Rules:

- Do not invent data the pipeline does not actually produce.
- Stay scoped to Churn Signals. Ignore voice, telephony, personal assistant features, and consumer or Amazon review products.
- Distinguish clearly between:
  - operator/internal products
  - customer-facing B2B products
  - platform/API products
- Distinguish clearly between:
  - real-time products
  - batch or scheduled products
  - hybrid products that read from precomputed artifacts
- Prefer products with strong evidence lineage, clear unit economics, and realistic operational cost.
- Flag any idea that depends on sparse, weak, or operationally expensive fields.
- Use the current Churn Signals product surfaces as context, but do not limit yourself to only cosmetic variations of those surfaces.

---

## User Prompt

I am going to give you:

1. A product context pack for Churn Signals
2. A few sample payloads from the live pipeline
3. A short description of current product surfaces

Based on that material, propose the strongest end products we can build next.

### Your objectives

- Find the highest-leverage product opportunities already implied by the pipeline
- Separate durable product ideas from nice-looking but weak ideas
- Prefer ideas that productize existing artifacts instead of adding new expensive always-on model work
- Show where Churn Signals has a real moat

### Required output format

#### 1. Ranked Opportunity List

Provide 10 product opportunities ranked from strongest to weakest.

For each opportunity include:

- `name`
- `category`
  - operator/internal
  - customer-facing B2B
  - platform/API
- `target user`
- `core problem solved`
- `primary UI concept`
- `required Churn Signals inputs`
- `pipeline stages used`
- `real-time vs batch vs hybrid`
- `why this is defensible`
- `main risks or weak assumptions`
- `MVP scope`
- `why now`

#### 2. Product Shape Recommendations

After the ranked list, recommend:

- the best `dashboard/feed` products
- the best `workflow/review queue` products
- the best `report/subscription` products
- the best `API/platform` products

#### 3. Build Order

Recommend a build order of the top 5 ideas with:

- estimated complexity: `low`, `medium`, `high`
- expected operating cost pressure: `low`, `medium`, `high`
- expected data readiness: `low`, `medium`, `high`
- expected commercial leverage: `low`, `medium`, `high`

#### 4. Anti-Patterns

List 5 product ideas that would look attractive but are bad fits for Churn Signals right now, and explain why.

---

## Evaluation Criteria

Use these criteria when ranking ideas:

1. Leverage of existing Churn Signals artifacts
2. Trustworthiness of the underlying signals
3. Low marginal model cost at read time
4. Clear user job and repeat usage pattern
5. Strong product defensibility
6. Feasible MVP path with current pipeline

Penalize ideas that:

- require fresh synthesis on every page load
- depend heavily on sparse fields
- need deprecated sources
- have weak evidence lineage
- create high operating cost without a clear monetizable unit
- drift into non-Churn-Signals Atlas products

---

## Optional Follow-Up Questions For The Model

After the first answer, one or more of these follow-ups usually helps:

### Follow-Up 1

Take the top 3 product opportunities and turn each into:

- one primary workflow
- one core page layout
- one north-star metric
- one pricing or packaging hypothesis

### Follow-Up 2

For the top 5 ideas, map each one to:

- existing endpoints or artifact types it can reuse
- missing data or instrumentation it still needs
- the biggest operational risk

### Follow-Up 3

For the single strongest idea, design:

- the MVP information architecture
- the main page and subpages
- the key cards, tables, and actions
- the operator review flow needed to keep it trustworthy

---

## Suggested Attachment Set

When using this prompt, attach:

- `docs/product_context_pack.md`
- `docs/product_context_samples.md`

Do not attach huge schema dumps unless the model asks for them.

---

## One-Line Prompt Wrapper

Use the Churn Signals product context pack and the attached sample payloads to propose the highest-leverage end products and product UIs we can build next, staying strictly grounded in the pipeline's real data, cost profile, latency model, and current product surfaces.
