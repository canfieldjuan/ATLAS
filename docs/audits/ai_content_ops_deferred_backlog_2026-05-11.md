# AI Content Ops Deferred Backlog

Created: 2026-05-11
Last updated: 2026-05-15

## Purpose

This is the current ordered backlog for AI Content Ops deferrals surfaced by
older PR plans and post-merge audits. It is not a full product roadmap; it is
the short list of follow-up work that still matters after the execution,
reasoning, export, compact UI parity, and DB reasoning admin seams already
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
- Generated asset batch review/status updates.

## Active Backlog

### 1. Reasoning context admin workflow

**Priority:** P2

**Why:** DB-backed reasoning context storage is now functional and has CLI
coverage for read, export, upsert, dry-run, stale cleanup, and metadata audit
logging. What is still missing is an operator-facing workflow for managing
those rows without hand-running scripts.

Remaining work:

- Browse and inspect reasoning context rows from the hosted admin surface.
- Create or edit a context row with the same validation rules as the upsert
  CLI.
- Delete or retire a row deliberately, with tenant/account scoping.
- Show recent metadata audit-log entries or equivalent operator-visible
  provenance.

**Likely slice:** start with a small API/admin surface over the existing
repository methods. Keep full audit-table persistence separate unless a host
needs centralized audit storage immediately.

### 2. Operator review UX and richer result previews

**Priority:** P3

**Why:** Batch review/status updates are now done. Older frontend/result-summary
plans still deferred richer generated-asset previews and component-level tests.
These are product polish, not core readiness blockers.

**Likely slice:** improve preview cards for report, blog post, landing page,
and sales brief rows, then add component-level frontend tests for the review
surface.

### 3. Scale hardening for batch review

**Priority:** P4

**Why:** Generated asset batch review currently reuses the existing scoped
single-row status update path. That preserves tenant filtering and keeps the
implementation simple. If hosts start reviewing large batches, repository-level
bulk SQL may be worth adding.

**Likely slice:** defer until actual batch sizes justify it.

### 4. Reasoning product depth and source breadth

**Priority:** P4

**Why:** AI Content Ops can consume file-backed, DB-backed, single-pass, and
multi-pass reasoning context. The remaining strategic questions are broader
than this backlog:

- More source adapters for customer-specific data bundles.
- Continued `extracted_reasoning_core` work if reasoning is sold as a stronger
  standalone layer.
- Host policy for richer falsification/cache/narrative-planning knobs.

**Likely slice:** handle through product roadmap or the reasoning-core backlog,
not as Content Ops cleanup.

## Current Pick Recommendation

Take item 1 next if we want operator-facing progress: reasoning context admin
workflow. Take item 2 next if we want frontend polish: richer generated-asset
previews and component tests.
