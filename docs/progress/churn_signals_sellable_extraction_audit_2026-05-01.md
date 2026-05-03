# Churn Signals Sellable Extraction Audit

Date: 2026-05-01

## Executive Readiness

The churn signals system is sellable today as an Atlas-hosted product, not yet
as a clean standalone package.

The strongest sellable unit is:

- tracked vendor setup
- public-review ingestion
- two-tier churn enrichment
- vendor and account signal surfaces
- evidence-backed reports
- webhooks / exports / CRM push as delivery channels

The system has real product assets already:

- customer-visible API surfaces under `b2b_dashboard`, `b2b_vendor_briefing`,
  `b2b_evidence`, `b2b_reviews`, and CRM/campaign endpoints
- billing and checkout flow for vendor reports
- tenant scoping through `saas_accounts` and `tracked_vendors`
- outbound webhook delivery and report subscription infrastructure
- recently centralized enrichment execution, persistence, policy, and provider
  call architecture

The main blocker is product boundary, not core capability. The codebase
currently exposes one large Atlas B2B operating system. The sellable extraction
should define one initial product lane and treat the rest as attachable modules.

## Recommended Product Boundary

### Product 1: Churn Signals Core

Sell this first.

Customer promise:

- Track one or more vendors.
- Ingest public review/community/source data.
- Enrich reviews into churn intent, pain, urgency, timeline, buyer authority,
  budget, competitor mentions, and evidence.
- Surface vendor-level and account-level churn signals.
- Deliver reports, dashboard views, exports, webhooks, and CRM pushes.

Core inputs:

- `tracked_vendors`
- scrape targets / manual review imports
- `b2b_reviews`
- source parser output
- optional CRM events

Core outputs:

- enriched `b2b_reviews.enrichment`
- `b2b_churn_signals`
- `b2b_company_signals`
- `b2b_intelligence`
- accounts-in-motion reports
- evidence witnesses / trace views
- webhook events
- CRM push records

This is the main extraction candidate.

### Product 2: Content Generation Pipeline

Treat as an adjacent product that consumes churn intelligence.

Includes:

- blog post generation
- vendor briefing generation
- challenger campaign generation
- email/report delivery templates

Do not bundle this into the first Churn Signals Core extraction unless the
commercial offer explicitly includes content production.

### Product 3: Account Motion / Watchlist Alerts

Treat as a premium module.

Includes:

- company signal candidates
- analyst approval/suppression workflow
- saved watchlist views
- scheduled alert emails
- webhooks and CRM pushes tied to account motion

This is sellable, but it depends on the core signal graph being stable.

## Current System Boundary

### Storage

Minimum core tables:

- `b2b_reviews`
- `b2b_churn_signals`
- `b2b_intelligence`
- `tracked_vendors`
- `saas_accounts`
- `saas_users`
- `b2b_scrape_targets`
- `b2b_scrape_log`
- `b2b_enrichment_stage_runs`
- `b2b_llm_exact_cache`

Likely core-plus tables:

- `b2b_company_signals`
- `b2b_review_vendor_mentions`
- `b2b_reasoning_synthesis`
- `b2b_vendor_witness_packets`
- `b2b_evidence_claims`
- `b2b_evidence_annotations`
- `b2b_webhook_subscriptions`
- `b2b_webhook_delivery_log`
- `b2b_crm_events`
- `b2b_crm_push_log`

Premium/adjacent tables:

- `b2b_company_signal_candidates`
- `b2b_company_signal_candidate_groups`
- `b2b_watchlist_views`
- `b2b_watchlist_alert_events`
- `b2b_watchlist_alert_email_log`
- `b2b_report_subscriptions`
- `b2b_report_subscription_delivery_log`
- `b2b_campaigns`
- campaign audit/sequence/suppression tables
- blog post tables

### Autonomous Jobs

Core:

- `b2b_scrape_intake`
- `b2b_enrichment`
- `b2b_enrichment_repair`
- `b2b_churn_intelligence`
- `b2b_reasoning_synthesis`
- `b2b_tenant_report`

Operational/support:

- `b2b_parser_upgrade_maintenance`
- `b2b_scrape_target_pruning`
- `b2b_score_calibration`
- `b2b_product_profiles`
- `b2b_witness_quality_maintenance`

Adjacent:

- `b2b_blog_post_generation`
- `b2b_campaign_generation`
- `b2b_challenger_brief`
- `b2b_watchlist_alert_delivery`
- `crm_event_processing`

### API Surface

Core customer API:

- `POST /b2b/reviews/import`
- `GET /b2b/dashboard/signals`
- `GET /b2b/dashboard/signals/{vendor_name}`
- `GET /b2b/dashboard/high-intent`
- `GET /b2b/dashboard/accounts-in-motion`
- `GET /b2b/dashboard/vendors/{vendor_name}`
- `GET /b2b/dashboard/reports`
- `GET /b2b/dashboard/reports/{report_id}`
- `GET /b2b/dashboard/reviews`
- `GET /b2b/dashboard/reviews/{review_id}`
- `GET /b2b/dashboard/export/signals`
- `GET /b2b/dashboard/export/reviews`
- `GET /b2b/dashboard/export/high-intent`
- `GET /b2b/evidence/witnesses`
- `GET /b2b/evidence/witnesses/{witness_id}`
- `GET /b2b/evidence/vault`
- `GET /b2b/evidence/trace`
- webhook subscription endpoints under `b2b_dashboard`
- CRM event endpoints under `b2b_crm_events`

Operator/internal API:

- vendor refresh/reasoning triggers
- parser/source health endpoints
- calibration endpoints
- correction endpoints
- source telemetry endpoints
- briefing review queues

These should not be exposed as the first customer contract without deliberate
role separation.

## Extraction Readiness

### Strong Areas

1. Enrichment execution is now productizable.

The enrichment engine has a clean internal shape:

- task runner
- row runner
- single-review runner
- stage planner
- stage controller
- stage ledger
- provider-call service
- persistence/finalization service
- validation/derivation/policy modules

This is the strongest extraction-ready subsystem.

2. The customer tenancy model already exists.

`saas_accounts`, plans, auth, and `tracked_vendors` already support scoped B2B
customers. This makes hosted-product extraction more realistic than library
extraction.

3. Delivery channels exist.

The system already has:

- report delivery
- webhook subscriptions
- CRM event ingestion
- CRM push logs
- vendor report checkout
- email templates

4. Evidence and auditability are real differentiators.

The evidence vault, witnesses, reasoning traces, and annotations can make the
product defensible. They should be treated as product features, not internal
debug screens.

### Weak Areas

1. The dashboard API is too large and mixed-purpose.

`b2b_dashboard.py` includes customer dashboards, operator controls, source
health, corrections, webhooks, account motion, exports, and operational
debugging. A sellable API needs a smaller public contract.

2. Scraping is still the largest extraction risk.

The scrape system carries source-specific operational complexity:

- source allowlists and deprecated sources
- CAPTCHA / proxy / Web Unlocker configuration
- parser maintenance
- source quality gates
- anti-bot failure modes
- source-specific parser health

For a sellable product, this should remain Atlas-operated infrastructure at
first. Do not sell it as customer-run software yet.

3. Config is broad and Atlas-shaped.

`B2BChurnConfig` contains core product controls, scraping controls, enrichment
controls, reasoning controls, content controls, report controls, and campaign
controls. The first extraction needs a smaller config facade.

4. Source onboarding is not yet productized end-to-end.

Vendor add is close. Source add still depends on parser/declaration work and
source-specific operational knowledge.

5. Public/private data boundaries need tightening.

Some endpoints use optional auth while others require `b2b_growth`. The current
shape is workable for Atlas but should be reviewed before selling.

## Sellable Today Assessment

### Sellable Now: Hosted Churn Signals Dashboard + Reports

Readiness: high.

Why:

- core data pipeline works
- tenant/vendor scoping exists
- reports and exports exist
- evidence views exist
- billing/checkout exists for vendor reports

Packaging:

- "Track vendors and receive evidence-backed churn intelligence."
- Hosted Atlas product, not downloadable software.
- Customer config is vendor list, delivery settings, CRM/webhook destinations.

### Sellable Soon: Accounts In Motion

Readiness: medium-high.

Why:

- endpoints and candidate approval flows exist
- watchlists and alerts exist
- CRM/webhook push exists

Blockers:

- needs cleaner customer-facing review workflow
- needs clear confidence/explainability language
- should depend on persisted reports by default, not ad hoc live scans

### Sellable Later: Source/Data Engine

Readiness: medium.

Why:

- valuable data acquisition system
- many source parsers and gates

Blockers:

- source operational fragility
- anti-bot/provider dependency
- too much Atlas-specific scrape governance

### Sellable Later: Content Generation

Readiness: medium.

Why:

- generation skills exist
- campaign/blog/report templates exist
- consumes churn intelligence well

Blockers:

- must be separated from lead-gen campaign ops
- needs explicit content QA and approval contracts
- should consume a stable public intelligence contract, not raw Atlas tables

## Recommended Extraction Plan

### Phase 1: Product Contract

Define the public Churn Signals contract:

- `VendorSignal`
- `ReviewSignal`
- `AccountInMotion`
- `EvidenceWitness`
- `SignalReport`
- `WebhookEvent`

This is the most important next step. Without this, extraction will keep
following internal table shapes.

### Phase 2: Public API Facade

Create a small customer-facing API layer separate from `b2b_dashboard.py`:

- list tracked vendors
- add/remove tracked vendor
- list vendor signals
- list account signals
- list reports
- get report
- list evidence witnesses
- configure webhook
- export signals/reviews

Keep source health, parser health, correction tools, and calibration in
operator APIs.

### Phase 3: Config Facade

Split product config from operations config:

- customer-visible product controls
- operator scrape/enrichment controls
- provider/cost controls
- internal experiment controls

The product should not expose the entire `B2BChurnConfig`.

### Phase 4: Data Boundary

Define which tables are product tables versus Atlas operations tables.

Customer-facing reads should go through product views/adapters, not directly
through raw internal tables.

### Phase 5: Packaging

Choose one packaging mode:

1. Atlas-hosted SaaS module.
2. Dedicated Churn Signals service inside Atlas deployment.
3. Standalone package.

Recommendation: choose 1 first, then evolve toward 2. Do not attempt 3 yet.

## Product Boundary Decisions

1. Scraping should remain internal.

Customers add vendors and select sources. They should not operate parser health,
proxy config, CAPTCHA controls, or scrape maintenance.

2. Evidence is part of the product.

The system is more sellable if every signal can point to a witness and trace.
Do not hide evidence as an internal debug artifact.

3. Content generation is a consumer, not the core.

Keep content generation downstream of the Churn Signals contract.

4. Account motion is a premium module.

It should consume the same signal contract but can have its own workflow,
approval queue, alerts, and CRM integration.

5. The first public API should be smaller than the internal dashboard API.

Expose the product. Keep operations internal.

## Key Risks

1. Overbundling.

If churn signals, accounts-in-motion, content generation, campaigns, CRM
automation, and source operations are all sold as one product immediately, the
support surface becomes too large.

2. Data-source brittleness.

Scraping sources have uneven reliability. Sell "managed intelligence" rather
than "bring your own scraper."

3. Optional-auth endpoints.

Review public/private boundaries before presenting any customer-facing API.

4. Cost controls.

The enrichment pipeline now has better execution architecture, but customer
plans need hard usage limits and budget observability.

5. Reporting freshness.

Some customer views depend on persisted reports and freshness gates. This is
good, but the UI/API needs clear stale/no-data states.

## Immediate Next Actions

1. Draft the public Churn Signals DTO contract.
2. Split a small `b2b_churn_public` API facade from `b2b_dashboard.py`.
3. Add a tenant-scoped "add vendor" happy-path audit.
4. Define product/ops tables in a schema map.
5. Decide whether accounts-in-motion is in the first SKU or a premium module.

## Bottom Line

Churn Signals is sellable today as a hosted Atlas product if the offer is
framed as managed churn intelligence for tracked vendors.

It is not ready to be extracted as standalone software because source
operations, config, and internal dashboard surfaces are still too broad.

The next architecture move should be a public product contract and facade,
not another internal refactor.
