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
- Live execute persistence smoke for all generated assets.
- Blog blueprint population path.
- Intervention or autonomous reasoning provider.
- Full reasoning context drawer/detail UX.

## Active Backlog

### 1. DB reasoning provider hardening

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

### 2. Operator review UX and richer result previews

**Priority:** P3

**Why:** Older frontend/result-summary plans deferred batch review/status
updates, richer generated-asset previews, and component-level tests. These are
product polish, not core readiness blockers.

**Likely slice:** batch review/status updates before richer preview cards,
because batch review improves operator throughput across all asset types.

## Current Pick Recommendation

Take item 1 next: DB reasoning provider hardening. The intervention fallback
and detail UI now exist, so the next highest-leverage work is tightening the
durable provider semantics before adding more operator polish.
