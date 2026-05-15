# AI Content Ops Deferred Backlog

Created: 2026-05-11
Last updated: 2026-05-15

## Purpose

This is the current ordered backlog for AI Content Ops deferrals surfaced by
older PR plans and post-merge audits. It is not a full product roadmap; it is
the short list of follow-up work that still matters after the execution,
reasoning, export, review, preview, detail, and DB reasoning admin seams already
merged.

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
- Generated asset batch review/status updates and one-query batch updates.
- Generated asset preview cards and detail drawer.
- `blog_post` reasoning catalog/fixture parity.
- Live execute persistence smoke for all generated assets.
- Blog blueprint population path.
- Intervention or autonomous reasoning provider.
- Full reasoning context drawer/detail UX.
- DB reasoning provider target-mode filtering and settings-backed provider
  selection.
- DB reasoning context upsert semantics.
- DB reasoning context list/export CLI.
- DB reasoning stale-context cleanup CLI.
- DB reasoning upsert dry-run.
- DB reasoning upsert metadata audit log.
- DB reasoning upsert live-opportunity validation.
- DB reasoning context hosted admin list/upsert API.
- DB reasoning context scoped delete/retire API.
- DB reasoning context admin visibility events.

## Active Backlog

### 1. Reasoning product depth and source breadth

**Priority:** P2

**Why:** AI Content Ops can consume file-backed, DB-backed, single-pass, and
multi-pass reasoning context. The next value step is deciding how much of the
standalone reasoning layer each content type should expose and how hosts should
feed richer customer-specific source bundles into it.

Remaining work:

- More source adapters for customer-specific data bundles.
- Continued `extracted_reasoning_core` work if reasoning is sold as a stronger
  standalone layer.
- Host policy for richer falsification/cache/narrative-planning knobs.
- Per-content-type opt-in rules so simple assets avoid heavy reasoning paths
  while long-form assets can use stateful reasoning.

**Likely slice:** start with a narrow source-bundle adapter or a small
reasoning-provider capability check that directly improves host setup, rather
than a broad architecture refactor.

## Current Pick Recommendation

Take item 1 next. The generated asset operator workflow is now usable; the
remaining leverage is increasing the quality and breadth of reasoning/source
inputs that feed AI Content Ops.
