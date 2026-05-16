# AI Content Ops Deferred Backlog

Created: 2026-05-11
Last updated: 2026-05-16

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
- Campaign operations status reasoning-provider capability check.
- Source-adapter field alias support for common provider-style exports.
- Source-adapter cumulative audit and decision rules.
- Source-type precedence consolidation.
- Source-row field lookup cache for provider-style aliases.
- Host-facing reasoning policy audit for falsification, narrative planning,
  validation, and per-content-type depth.
- Reasoning preset catalog for host-facing depth choices.
- Operator-facing strict validation telemetry in Content Ops execution results.

## Active Backlog

### 1. Reasoning product depth

**Priority:** P2

**Why:** AI Content Ops can consume file-backed, DB-backed, single-pass, and
multi-pass reasoning context. Source ingestion is now broad enough for current
standalone use. The next value step is deciding how much of the standalone
reasoning layer each content type should expose.

Current policy audit:

- `docs/audits/content_ops_reasoning_policy_audit_2026-05-16.md`

Current policy catalog:

- `extracted_content_pipeline/reasoning_policy.py`

Remaining work:

- Continued `extracted_reasoning_core` work if reasoning is sold as a stronger
  standalone layer.
- Host-owned falsification policy wiring for strict presets.

**Shipped slice:** structured and strict multi-pass reasoning are wired for
`report` and `sales_brief`. Strict mode now fails closed before those assets are
generated when validation blockers are present, and the generated-asset error
reason includes the validation blocker identifiers. Content Ops execution also
mirrors those strict validation failures into per-step reasoning telemetry for
operator inspection.

### 2. Source breadth from real host exports

**Priority:** P3

**Why:** The source adapter now supports the current generic families, explicit
source-type precedence, tolerant field aliases, nested bundles, and cached
field lookup. More breadth should be driven by an actual customer export, not
by plausible platform shapes.

Remaining work:

- Add the minimum aliases/source keys required by a real host export fixture.
- Add end-to-end generated-asset quality tests by source type when that fixture
  exists.

**Likely slice:** wait for a real export. If none exists, skip this item.
If no export appears after the next reasoning-policy pass, decide whether this
remains roadmap work or should be removed from active backlog.

## Current Pick Recommendation

Take item 1 next, specifically structured reasoning for `report` and
`sales_brief` using the preset catalog. Source-adapter consolidation is
complete for now; the remaining leverage is controlled reasoning-policy depth
for long-form and multi-asset outputs. Source breadth should pause until a
real host export or field-loss risk appears.
