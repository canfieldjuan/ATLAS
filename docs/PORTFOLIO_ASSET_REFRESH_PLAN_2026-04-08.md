# Portfolio Asset Refresh Plan

Date: 2026-04-08

## Goal

Refresh the smallest set of portfolio assets that produces the biggest visible improvement in the public `atlas-portfolio` repo.

## Highest ROI Assets

### 1. `watchlists.png`

Why:
- Shows the newest ATLAS product layer clearly
- Combines saved views, competitive sets, vendor monitoring, account movement, and alerting
- Communicates a richer product surface than another dashboard screenshot

What to show:
- one saved view or competitive-set control
- vendor feed visible
- one expanded account-movement drawer if possible

### 2. `evidence-explorer.png`

Why:
- One of the most differentiated product surfaces in ATLAS
- Shows evidence-backed reasoning, witness inspection, and traceability
- Demonstrates that outputs are grounded, reviewable, and not prompt-only

What to show:
- vendor selected
- filters visible
- witness results visible
- vault or trace tab visible

### 3. `report-detail.png`

Why:
- Proves the system turns evidence and reasoning into a usable downstream artifact
- Shows executive summary, citations, structured sections, and reviewable output
- Connects the platform story from source evidence to delivered artifact

What to show:
- a battle card or reasoning-heavy report
- executive summary visible
- citation chips visible
- at least one structured section below

## Capture Order

1. `watchlists.png`
2. `evidence-explorer.png`
3. `report-detail.png`

## Practical Sequence

1. Open `/watchlists`
2. Capture `watchlists.png`
3. Open `/evidence`
4. Capture `evidence-explorer.png`
5. Open `/reports`
6. Open a strong battle-card or reasoning-heavy report at `/reports/:id`
7. Capture `report-detail.png`

## Why These Three First

- They reflect the newest ATLAS UI surfaces better than the older campaign/blog visuals
- They show product differentiation, evidence lineage, and reviewable downstream artifacts
- They improve the public portfolio materially without requiring a full media refresh

## Tactical Operator Checklist

Use this checklist during the actual capture session.

### `watchlists.png`

Route:
- `/watchlists`

Preferred state:
- a saved view selected, or a competitive set visible
- vendor feed populated
- one account-movement row expanded in the drawer if available

What to avoid:
- empty-state view
- modal overlays covering the table/feed
- stale or collapsed page with no movement/evidence visible

Minimum acceptable frame:
- top controls visible
- at least one vendor feed block or table section visible
- at least one account-level movement or alert-related detail visible

### `evidence-explorer.png`

Route:
- `/evidence`

Preferred state:
- vendor selected with rich witness coverage
- filters visible
- witness results visible
- vault or trace tab available in the same frame if possible

Suggested vendor:
- use a vendor with strong witness density such as `Salesforce`, `HubSpot`, or another vendor with multiple witness rows

What to avoid:
- empty search state
- no filters visible
- only header chrome without evidence rows

Minimum acceptable frame:
- vendor selected
- witness cards or rows visible
- filter controls visible

### `report-detail.png`

Route:
- `/reports`
- then open a strong battle-card or reasoning-heavy report at `/reports/:id`

Preferred state:
- executive summary visible
- citation chips visible
- at least one structured or specialized section below the fold

Suggested report:
- battle card first
- otherwise a reasoning-heavy vendor or comparison report

What to avoid:
- generic lightweight report with no citations
- top-of-page frame that only shows title metadata

Minimum acceptable frame:
- report title
- executive summary
- citations
- one evidence-backed section

## Session Prep

Before capturing:

1. Refresh the app and make sure recent data is loaded
2. Use full-width desktop layout
3. Close unrelated toasts, modals, or debug overlays
4. Prefer dark theme if the surface supports it cleanly
5. Avoid real customer emails or phone numbers in visible rows

## Recommended Capture Order

1. Capture `watchlists.png`
2. Capture `evidence-explorer.png`
3. Capture `report-detail.png`

If time remains after those three:

4. `opportunities.png`
5. `pipeline-review.png`

## Exact Recommended Targets

Use these defaults unless the underlying data is empty on the day you capture.

### `watchlists.png`

Recommended vendor focus:
- `Salesforce`
- fallback: `HubSpot`

Recommended state:
- open `/watchlists`
- select the saved view or competitive set centered on `Salesforce` if available
- make sure the vendor feed is visible
- expand one account-movement drawer row for `Salesforce`

Best details to keep visible:
- saved-view controls
- competitive-set controls
- one vendor feed block
- one expanded account movement with urgency or timing context

### `evidence-explorer.png`

Recommended vendor:
- `Salesforce`
- fallback: `HubSpot`

Recommended state:
- open `/evidence`
- search and select `Salesforce`
- keep filters visible
- stay on `Witnesses` if the result set is rich
- if `Witnesses` is thin, switch to `Vault` or `Trace`

Best filters:
- no filter, if witness volume is already good
- otherwise add one high-signal witness filter such as `displacement` or `pain_point`

Best details to keep visible:
- selected vendor
- witness cards or rows
- filters
- snapshot of the right-side evidence context if available

### `report-detail.png`

Recommended report type:
- `battle_card`
- fallback: a reasoning-heavy vendor comparison or vendor scorecard

Recommended vendor:
- `Salesforce`
- fallback: `HubSpot`

Recommended state:
- open `/reports`
- filter to the chosen vendor if needed
- open the strongest `battle_card` or reasoning-rich report
- keep the page high enough to show:
  - title
  - executive summary
  - citation chips
  - first structured section below

Best details to keep visible:
- report badge/type
- executive summary block
- citations
- one specialized or structured section

### `opportunities.png`

Recommended vendor:
- `Salesforce`
- fallback: `HubSpot`

Recommended state:
- open `/opportunities?vendor=Salesforce`
- keep urgency and buying-stage context visible
- expand one row if the page supports a richer detail view

Best details to keep visible:
- urgency
- stage
- reasoning summary
- one action such as campaign generation or CRM push

### `pipeline-review.png`

Recommended tab/view:
- B2B efficiency or delivery operations
- fallback: watchlist-delivery operations

Recommended state:
- open `/pipeline-review`
- prefer a section with current runs, costs, or delivery statuses
- avoid a frame that only shows an empty queue table

Best details to keep visible:
- one summary card row
- one table with statuses
- operator controls or filters

## Exact Capture Toggles

### For `watchlists.png`
- keep drawers open if one shows account movement clearly
- avoid opening create/edit modals
- keep vendor feed and controls on the same screen if possible

### For `evidence-explorer.png`
- prefer `Witnesses` tab first
- if the witness list looks sparse, use `Vault` or `Trace`
- keep the filter rail visible

### For `report-detail.png`
- do not crop too high
- include citations in-frame
- include one evidence-backed section below the summary

### For `opportunities.png`
- keep the vendor query param active if it improves density
- avoid empty selected states or hidden side panels

### For `pipeline-review.png`
- prefer delivery/cost surfaces over raw queue-only views
- keep summary cards and one detailed table in the same frame
