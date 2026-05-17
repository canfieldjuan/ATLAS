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
- Strict validation blocked-event logging in Content Ops execution.
- Blog-specific packaged narrative pack for structured reasoning.

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

**Shipped slice:** structured multi-pass reasoning is wired for `blog_post`,
`report`, and `sales_brief`; strict multi-pass reasoning is wired for `report`
and `sales_brief`. Blog posts use a separate `content_ops_blog` narrative pack
so they do not inherit generic report section policy. Strict mode now fails
closed before report/sales generation when validation blockers are present, and
the generated-asset error reason includes the validation blocker identifiers.
Content Ops execution also mirrors those strict validation failures into
per-step reasoning telemetry for operator inspection and logs a structured
warning when strict validation blocks a step. Plan and execute paths share
packaged reasoning runtime constants so unsupported reasoning requests fail
consistently. Hosts can now attach explicit falsification rules to the strict
preset; no falsification LLM calls are made unless the host supplies those
rules.

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
No export appeared before the reasoning-policy parity pass closed. Keep this as
roadmap work, not an active slice, until a real host export fixture exists.

## Current Pick Recommendation

The host-facing AI Content Ops reasoning-policy arc is complete for the current
standalone product surface: preset catalog, packaged structured runtime support
for all reasoning-aware generated outputs (`email_campaign`, `blog_post`,
`report`, `landing_page`, `sales_brief`), strict report/sales validation,
explicit strict falsification rules, and blog-specific narrative packs have
shipped.

PR #566 and PR #567 were audit-recommended parity closures that landed after
the previous closeout. They do not reopen the reasoning-policy arc; they
complete the deferred audit scope that already existed at closeout time.

Do not take another AI Content Ops reasoning-policy slice unless it answers one
of these concrete needs:

- A host asks for a new packaged runtime output or preset.
- A real generated-asset run exposes validation metadata that operators cannot
  act on.
- `extracted_reasoning_core` advances enough that AI Content Ops needs a new
  stable provider port or capability check.

Until then, pause speculative Content Ops reasoning-policy work. The next
highest-leverage code should come from either a real source export fixture
(item 2) or the separate `extracted_reasoning_core` productization track. If a
future slice touches reasoning policy, it should name the concrete trigger from
the list above in its plan doc.
