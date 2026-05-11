# AI Content Ops Deferred Backlog

Date: 2026-05-11

## Purpose

This is the current ordered backlog for AI Content Ops deferrals surfaced by
older PR plans and post-merge audits. It is not a full product roadmap; it is
the short list of follow-up work that still matters after the execution,
reasoning, export, and compact UI parity work already merged.

## Retired Historical Deferrals

The following items appear in older plan docs but are no longer active backlog:

- Execution service wiring for `email_campaign`, `blog_post`, `report`,
  `landing_page`, `sales_brief`, and `signal_extraction`.
- Host LLM and skill adapters for the Content Ops execution bundle.
- File-backed and DB-backed reasoning provider wiring.
- Catalog-level reasoning provider status.
- Consumed reasoning payloads and compact consumed-context summaries for the
  five LLM-backed generated assets.
- Atlas Intel compact rendering for reasoning source and consumed contexts.
- Parse-retry parity and retry-adjusted usage/cost reporting across generated
  assets.
- Generated asset export and review paths for report, blog post, landing page,
  and sales brief drafts.
- `blog_post` reasoning catalog/fixture parity.

## Active Backlog

### 1. Live execute persistence smoke for all generated assets

**Priority:** P0

**Why:** This is the strongest operational proof that AI Content Ops is usable
outside Atlas. Existing tests cover bundles, route boundaries, offline smokes,
and provider contracts, but there is still no single live smoke proving:

1. `POST /content-ops/execute` enters the hosted route.
2. Tenant scope is applied.
3. Host adapters are used.
4. Each LLM-backed output persists a real draft.
5. The response reports generated ids, reasoning usage, and failures correctly.

**Likely slice:** one PR with a focused route-level smoke for the smallest
fixture set, then expand asset-by-asset if the first version gets too large.

### 2. Blog blueprint population path

**Priority:** P1

**Why:** Blog execution is wired, but the sellable blog path needs a reliable
way to populate `blog_blueprints`. Older storage work intentionally deferred
the host autonomous task or ETL that writes blueprints. Without this, blog
generation can be technically wired while still depending on pre-seeded rows.

**Likely slice:** define the host-side blueprint ingestion/population seam and
add one CLI or task adapter that writes `PostBlueprint` rows through the
product-owned repository.

### 3. Intervention or autonomous reasoning provider

**Priority:** P1

**Why:** File and DB providers are enough for handoff, but they do not yet turn
Atlas reasoning/intervention outputs into Content Ops context automatically.
This is the bridge to the higher-value product story: reasoning as a separate
layer that content products can consume.

**Likely slice:** add a provider that reads
`intelligence/autonomous_narrative_architect` or equivalent intervention output
and normalizes it into the existing `CampaignReasoningContextProvider` port.

### 4. DB reasoning provider hardening

**Priority:** P2

**Why:** The DB provider is functional, but older plans deferred operational
polish:

- Per-target-mode read filtering.
- Settings integration for provider selection/config.
- Upsert semantics for saved contexts.
- Stale-context sweeper.
- Admin editing workflow for context rows.

**Likely slice:** start with per-target-mode filtering and settings integration;
defer admin UI until the storage semantics are stable.

### 5. Full reasoning context drawer/detail UX

**Priority:** P2

**Why:** Atlas Intel now renders compact consumed-context summaries. The richer
drawer/detail UI remains deferred. This is useful for trust and debugging, but
it should follow the live smoke/provider hardening work because the runtime
contract is already inspectable at a compact level.

**Likely slice:** add a drawer over the existing `reasoning.consumed_contexts`
payload first. Do not add a new backend shape unless the current bounded
payload is insufficient.

### 6. Operator review UX and richer result previews

**Priority:** P3

**Why:** Older frontend/result-summary plans deferred batch review/status
updates, richer generated-asset previews, and component-level tests. These are
product polish, not core readiness blockers.

**Likely slice:** batch review/status updates before richer preview cards,
because batch review improves operator throughput across all asset types.

## Current Pick Recommendation

Take item 1 next: live execute persistence smoke. It proves the standalone
product works end-to-end through the same route and adapter seams customers
would use. If that smoke exposes setup friction, fix the friction before adding
more UI surface.
