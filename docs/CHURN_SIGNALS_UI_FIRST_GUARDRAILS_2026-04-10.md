# Churn Signals UI-First Guardrails

Date: 2026-04-10

## Decision

Treat the interactive UIs as the first-class Churn Signals products.

Treat reports, battle cards, briefings, and subscriptions as supporting or additive
artifacts rather than the default center of the product.

This note supplements:

- [CHURN_SIGNALS_END_PRODUCTS_2026-04-04.md](/home/juan-canfield/Desktop/Atlas/docs/CHURN_SIGNALS_END_PRODUCTS_2026-04-04.md)

## Product Guardrails

### Favor

- precomputed watchlists, feeds, review queues, and subscriptions
- evidence-backed drilldown inside every workflow
- explicit human review for outbound, briefing, and CRM actions
- confidence-tiered and analyst-assist UX when identity or account resolution is sparse
- bounded interactive tools that read persisted intelligence rather than free-form live synthesis

### Avoid

- "ask anything" live copilot surfaces that re-synthesize market intelligence on page load or on every interaction
- fully automated outbound sequencing with no approval step
- large-volume live enrichment triggered by opening a dashboard
- competitor maps that recompute cross-vendor reasoning on every filter change
- product claims that depend on deprecated sources or dense company identity being present everywhere

## Verified Audit

The following files were reviewed for this audit:

- `atlas-churn-ui/src/App.tsx`
- `atlas-churn-ui/src/components/Sidebar.tsx`
- `atlas-churn-ui/src/hooks/useApiData.ts`
- `atlas-churn-ui/src/pages/Dashboard.tsx`
- `atlas-churn-ui/src/pages/Watchlists.tsx`
- `atlas-churn-ui/src/pages/Opportunities.tsx`
- `atlas-churn-ui/src/pages/Prospects.tsx`
- `atlas-churn-ui/src/pages/EvidenceExplorer.tsx`
- `atlas-churn-ui/src/pages/Reports.tsx`
- `atlas-churn-ui/src/pages/ReportDetail.tsx`
- `atlas-churn-ui/src/pages/CampaignReview.tsx`
- `atlas-churn-ui/src/pages/BriefingReview.tsx`
- `atlas-churn-ui/src/pages/Challengers.tsx`
- `atlas-churn-ui/src/pages/VendorTargets.tsx`
- `atlas-churn-ui/src/pages/WinLossPredictor.tsx`
- `atlas_brain/api/b2b_tenant_dashboard.py`
- `atlas_brain/api/b2b_evidence.py`

### Strong Alignment

#### Watchlists and accounts in motion are already the best product fit

- `Watchlists.tsx` reads `fetchSlowBurnWatchlist()` and `fetchAccountsInMotionFeed()`.
- `b2b_tenant_dashboard.py` serves those from ranked signal rows and persisted
  accounts-in-motion reports, not from live synthesis on interaction.
- Saved views, alert thresholds, alert delivery, evidence drilldown, and
  competitive-set controls all fit the precompute-then-explore model.

Verdict: keep treating `/watchlists` as the primary habitual surface.

#### Opportunities, prospects, and review queues are analyst-assist products

- `Opportunities.tsx` reads `fetchHighIntent()` and uses explicit user actions for
  `generateCampaigns()`, `approveCampaign()`, and `pushToCrm()`.
- `CampaignReview.tsx` is an approval queue with draft, approve, queue-send, and
  reject actions.
- `Prospects.tsx` includes a manual queue and company overrides instead of
  pretending identity resolution is always dense or certain.
- `BriefingReview.tsx` uses explicit pending approval and bulk approve/reject flows.

Verdict: this is the right operational shape. Keep these human-reviewed.

#### Evidence is already positioned as a trust layer

- `EvidenceExplorer.tsx` is read-only over witnesses, vault, and trace.
- `b2b_evidence.py` explicitly describes the Evidence Explorer as the trust layer
  embedded across watchlists, reports, and opportunity queues.
- Witness drilldown is embedded from `Watchlists.tsx` and `ReportDetail.tsx`
  through `EvidenceDrawer`.

Verdict: aligned. Keep evidence as a reusable drilldown layer.

#### Operations surfaces are product-supporting, not customer-facing gimmicks

- `PipelineReview.tsx` is an internal control center for queue health, artifact
  attempts, validation failures, delivery ops, and cost telemetry.
- `CampaignReview.tsx` and `BriefingReview.tsx` are the active approval gates.
- Polling is limited to review/ops queues via `pollIntervalMs: 30000`; the main
  customer-facing pages are not running constant polling loops.

Verdict: aligned. This is the right place for operational live updates.

#### Win/Loss is bounded, not an open-ended copilot

- `WinLossPredictor.tsx` loads saved predictions on mount, but actual prediction
  runs happen only on explicit user action through `predictWinLoss()` or
  `compareWinLoss()`.

Verdict: acceptable as a bounded tool.

## Gaps To Keep In Mind

### 1. The default home route is still generic overview, not the strongest habitual product

- `/` still maps to `Dashboard.tsx`.
- `Dashboard.tsx` is useful, but it is a general overview page rather than the
  stronger watchlist/accounts-in-motion habit surface.

Recommendation:

- bias future product framing toward `/watchlists` as the primary day-to-day home
- keep the dashboard as summary/health context, not the core value proposition

### 2. Reports are supporting, but the UI still presents them as a top-level peer product

- `Sidebar.tsx` keeps `/reports` in the main top-level nav.
- `Reports.tsx` is both a library and a composer surface for on-demand report
  generation.

This is not a production bug, but it is a product-framing mismatch with the
current direction.

Recommendation:

- keep reports, battle cards, and subscriptions
- frame them as library/subscription/supporting artifacts rather than the main shell
- consider renaming or repositioning the surface toward "Library" or "Artifacts"

### 3. Review-safe workflows exist, but they are fragmented

- Campaign review, briefing review, prospects/manual queue, and pipeline review
  all exist as separate surfaces.

This is better than unsafe automation, but it is not yet the "Unified Operator
Review Queue" target.

Recommendation:

- keep the current approval gates
- consolidate cross-workflow operator triage over time instead of adding more
  separate review pages

### 4. Challengers is lightweight and safe, but not yet the stronger displacement monitor

- `Challengers.tsx` derives summaries from `fetchVendorTargets()` and
  `fetchHighIntent()` client-side.
- That avoids expensive live cross-vendor re-synthesis, which is good.
- It also means the surface is lighter-weight than the longer-term
  precomputed displacement monitor concept.

Recommendation:

- keep this surface read-only and cheap
- if expanded, back it with persisted pairwise/displacement artifacts rather than
  live cross-vendor recomputation

### 5. Evidence should remain a layer even if the standalone page stays

- The standalone `EvidenceExplorer.tsx` page is useful.
- The main strategic value is still the embedded witness drilldown inside
  watchlists, reports, and workflow queues.

Recommendation:

- keep the standalone explorer as a power-user utility
- prioritize embedded evidence affordances before adding more standalone evidence features

## Anti-Pattern Check

### 1. Live copilot that re-synthesizes on every load

Result: not found in the current UI shell.

Notes:

- customer-facing pages mostly read persisted feeds, signal rows, witnesses, or
  previously generated artifacts
- the hook-level refresh behavior is focus/reconnect based, not constant reasoning

### 2. Fully automated outbound with no human review

Result: not found.

Notes:

- campaigns and briefings run through explicit review and approval pages
- CRM push is user-triggered

### 3. Large-volume live enrichment when opening dashboards

Result: not found in the main UI surfaces reviewed.

Notes:

- the main pages fetch precomputed or persisted data
- the backend does expose some live/manual operational paths, but they are not
  the default customer-facing page behavior

### 4. Real-time competitor map recomputed on every filter change

Result: not found.

Notes:

- the current challenger/displacement-adjacent surface is lightweight and derived
  from fetched rows
- pairwise and battle-card logic remain persisted-artifact oriented

### 5. Pretending dense company identity exists everywhere

Result: not found.

Notes:

- `Prospects.tsx` keeps a manual queue and company override workflow
- opportunities and prospects expose evidence, titles, and status, but do not
  pretend certainty where the data is sparse

## Default Decision Rule For Future UI Work

When choosing between two product directions:

1. prefer the option that reads persisted artifacts or narrow read models
2. prefer the option that strengthens watchlists, feeds, review queues, and evidence drilldown
3. keep outbound, briefing, and CRM actions approval-gated
4. treat reports as supporting assets unless a workflow explicitly needs them
5. reject new surfaces that depend on live cross-vendor synthesis during ordinary browsing

