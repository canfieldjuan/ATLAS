# Churn Signals End Products

Date: 2026-04-04

## Decision

Treat the following ranked list as the current benchmark for Churn Signals end-product planning.

This supersedes the earlier broader Atlas-wide ideation and keeps product planning scoped to:

- churn and displacement intelligence
- witness-backed evidence UX
- persisted reports, battle cards, briefings, and campaigns
- operator and API surfaces that directly support those products

Do not expand this benchmark with:

- voice or telephony
- personal assistant workflows
- Amazon or consumer-review products
- generic "AI chat over everything" ideas

See also:

- [CHURN_SIGNALS_UI_FIRST_GUARDRAILS_2026-04-10.md](/home/juan-canfield/Desktop/Atlas/docs/CHURN_SIGNALS_UI_FIRST_GUARDRAILS_2026-04-10.md)

---

## Benchmark Ranking

### 1. Vendor Intelligence Watchlists & Accounts-in-Motion Feed

- Category: customer-facing B2B
- Core shape: watchlists, ranked vendor feed, movement cards, evidence drawer, saved views, freshness markers
- Why it leads: best fit for the current precompute-then-explore pipeline model and strongest daily-use habit surface

### 2. Report & Battle Card Library with Evidence-Backed Subscriptions

- Category: customer-facing B2B
- Core shape: report library, artifact detail, witness citations, freshness state, subscribe/export actions
- Why it ranks high: low read-time cost, strong monetization surface, already backed by persisted artifacts

### 3. Campaign Opportunity Workbench

- Category: customer-facing B2B
- Core shape: opportunity queue, evidence panel, analyst review, approve-to-campaign/export-to-CRM
- Important constraint: treat as analyst-assisted because some commercially useful fields remain variable-trust

### 4. Evidence Explorer / Witness Drilldown

- Category: customer-facing B2B
- Core shape: reusable trust layer showing source text, evidence spans, witness IDs, provenance, and reasoning references
- Important note: best treated as a cross-product trust layer, not a standalone business by itself

### 5. Churn Incident Alerts API / Webhooks

- Category: platform/API
- Core shape: event subscriptions, payload previews, delivery logs, retry status, and feed integration

### 6. Displacement Map & Competitor Pressure Monitor

- Category: customer-facing B2B
- Core shape: precomputed competitor matrix with witness-backed wedges and freshness state
- Important constraint: must stay precomputed because cross-vendor reasoning is budget-constrained

### 7. CRM Sync for High-Intent Account Signals

- Category: platform/API
- Core shape: approval-gated sync of qualified opportunities into CRM custom objects

### 8. Unified Operator Review Queue

- Category: operator/internal
- Core shape: consolidated review and publish control plane across reports, campaigns, briefings, and evidence coverage

### 9. Pipeline Budget & Coverage Control Center

- Category: operator/internal
- Core shape: artifact-level cost, rejection pressure, freshness, and reprocess monitoring

### 10. Executive Briefing Subscription

- Category: customer-facing B2B
- Core shape: recurring exec digest with summary, witness anchors, and linked deeper surfaces

---

## Strategic Notes

### Strongest product pattern

The strongest product pattern for Churn Signals remains:

1. expensive intelligence computed offline or on schedule
2. durable artifacts stored with evidence lineage
3. interactive read-time exploration on top of those artifacts

That is why the top-ranked products are:

- watchlists and feeds
- report libraries
- evidence drilldowns
- API subscriptions

and not:

- live re-synthesis tools
- generic chat fronts
- on-demand cross-vendor comparison engines

### Moat

Churn Signals' moat remains:

- longitudinal review enrichment depth
- witness-backed evidence lineage
- reusable reasoning contracts
- operator-grade cost and telemetry visibility

The benchmark products should surface those directly.

### Build sequencing principle

Do not simply build in rank order.

Build order should respect:

1. trust layer first
2. precomputed read surfaces second
3. workflow products third
4. platform extensions fourth

That means the watchlist/feed product can lead, but it should be designed with embedded evidence drilldown from the start.

---

## Selected Build Target

Proceed next with:

### Vendor Intelligence Watchlists & Accounts-in-Motion Feed

This is the current product build priority.

Companion expectation:

- evidence drilldown is not a separate later idea in spirit
- it should be planned as a required trust layer inside this product

See:

- [WATCHLISTS_ACCOUNTS_IN_MOTION_BUILD_PLAN_2026-04-04.md](/home/juan-canfield/Desktop/Atlas/docs/WATCHLISTS_ACCOUNTS_IN_MOTION_BUILD_PLAN_2026-04-04.md)
