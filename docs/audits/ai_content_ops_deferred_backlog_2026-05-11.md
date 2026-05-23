# AI Content Ops Deferred Backlog

Created: 2026-05-11
Last updated: 2026-05-23

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
- Atlas Intel ingestion default-fields entry for source-row imports.
- Source-adapter cumulative audit and decision rules.
- Source-type precedence consolidation.
- Source-row field lookup cache for provider-style aliases.
- Review-source readiness summary for scraped review sources.
- Host-facing reasoning policy audit for falsification, narrative planning,
  validation, and per-content-type depth.
- Reasoning preset catalog for host-facing depth choices.
- Operator-facing strict validation telemetry in Content Ops execution results.
- Strict validation blocked-event logging in Content Ops execution.
- Blog-specific packaged narrative pack for structured reasoning.
- Extracted blog SEO field persistence into first-class `blog_posts` columns.
- Blog SEO/AEO and GEO readiness summaries in generated-asset export/review
  output.
- Blog SEO/AEO and GEO save-time quality gates plus GEO repair loop.
- Blog publish-level SEO/GEO/JSON-LD/crawler-visible article verification.
- Atlas Intel blog readiness review and breakdown UI.
- Landing-page SEO/AEO/GEO input contract, prompt consumption, save-time
  readiness, review UI, draft edit/repair, public rendering,
  sitemap/prerender, publish verification, and generation smoke coverage.

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

**Review-source readiness update:** PR #591 shipped a source summary mode for
the review-source exporter. A live Atlas check reported quote-grade rows for
G2 (364), Capterra (154), and TrustRadius (31). Trustpilot reported 0
quote-grade rows, so it should not be used for Content Ops export until those
reviews are re-enriched with v4 phrase metadata. This closes the "which scraped
review source is usable now?" uncertainty without adding a new ingestion slice.

**Review-source Postgres smoke update:** PR #597 added the repeatable
review-source Postgres smoke, and PR #598 added a schema preflight so missing
Content Ops tables fail before import with a migration-runner instruction. A
live local Atlas run against G2/Slack now passes end to end after creating the
product-owned `campaign_opportunities` table: 1 quote-grade G2 review source
row imported, 2 offline deterministic campaign drafts persisted
(`email_cold`, `email_followup`), and the draft export CLI returned both rows
under `account_id=content_ops_smoke`.

**Live-provider smoke update:** PR #628 added optional `--llm pipeline` mode to
the review-source Postgres smoke, and PR #630 added the same mode to the CFPB
support-ticket-like Postgres smoke. Both paths still default to deterministic
offline generation for CI and host readiness, but operators can now run the
same imported source rows through the product `PipelineLLMClient` seam when
database and provider credentials are available. A live operator run remains
manual because this session did not have those credentials loaded.

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

The blog SEO/AEO/GEO arc is also closed for the current product contract.
The original discovery audit now records the merged closeout chain:

- `docs/audits/ai_content_ops_blog_seo_aeo_geo_discovery_2026-05-20.md`

Do not take another Content Ops blog SEO/GEO slice unless it is driven by a
new live-output failure, a UI/operator need, or a publish verifier regression.

The landing-page SEO/AEO/GEO arc is also closed for the current product
contract. The landing-page contract audit now records the implemented chain:

- `docs/audits/ai_content_ops_landing_page_seo_aeo_geo_contract_2026-05-21.md`

Do not take another Content Ops landing-page SEO/GEO slice unless it is driven
by a new live-output failure, a UI/operator need, or a publish verifier
regression. Historical plan-doc deferrals for landing-page input capture,
prompt usage, readiness gating, review UI, edit, repair, public rendering,
prerendering, and publish verification are stale after the merged closeout
chain.

The review-source readiness and Postgres smoke closeouts do not create a new
active Content Ops implementation backlog. G2 can use the existing exporter,
source-row import, DB-backed draft persistence, optional live-provider
generation mode, and draft export path.
Capterra and TrustRadius now also pass the Postgres smoke. The TrustRadius run
surfaced and closed a row-export gap: summary counts found 31 quote-grade rows
while the row exporter returned 0 because quote-grade filtering happened after
the first urgency/date scan window. The row exporter now applies the same
quote-grade predicate in SQL before ordering/paging, and a live TrustRadius
run imported 1 BambooHR review source row and persisted 2 offline deterministic
drafts. Trustpilot is blocked on data quality, not extractor code.

**Support-ticket/source UI update:** PR #620 added common help desk export
aliases such as ticket/case numbers, requester/customer contact fields,
organization names, issue descriptions, latest comments, and ticket/case
titles. PR #621 exposed the existing backend `default_fields` contract in
Atlas Intel so operators can bind fallback account/contact/vendor metadata
without editing each source row. These close the known generic source-row
ingestion friction. PR #630 added optional live-provider mode to the CFPB
support-ticket-like Postgres smoke, matching the review-source path. Further
source breadth should still wait for a real host export fixture.

**FAQ output-contract update:** PR #666 proved the persisted support-ticket FAQ
lifecycle: source rows -> generated Markdown draft -> export -> review/status
update -> reviewed export. PR #667 tightened the packaged support-ticket FAQ
demo so it proves the three output checks operators care about: customer
vocabulary, repeated-intent condensation, and action items. The FAQ Markdown
CLI now has an opt-in `--require-output-checks` guard for host smoke runs. This
does not create a new active implementation backlog. PR #673 upgraded the
deterministic FAQ renderer from a thin evidence summary to article-style
answers with grounded summaries, numbered next steps, support escalation
guidance, and cited ticket quotes. PR #684 added generic complaint-narrative
question extraction and billing intent grouping so real public complaint rows
such as CFPB narratives can still satisfy the customer-vocabulary check without
CFPB-specific renderer logic. PR #687 added a live CFPB-to-FAQ Markdown smoke
that fetches public complaint rows, converts them through the generic
source-row adapter, and fails closed when FAQ output checks do not pass.
The local CFPB export at `/home/juan-canfield/Downloads/archive (1)/rows.csv`
has 1,282,355 rows, 383,564 with usable complaint narratives. A 50-row debt
collection sample exposed generic gaps: provider-style complaint narrative
fields needed direct source-row aliases, and debt/credit-report complaints
needed financial complaint action policy before SaaS-style reporting/account
rules. Those source-level fixes landed in the FAQ complaint/source-policy
chain. Future FAQ/source work should be driven by a real customer help desk
export, hosted UI need, or another real dataset exposing a generic policy gap.

**Real CFPB output-quality update:** A follow-up run against three 150-row
samples from the same local CFPB export (`Debt collection`, `Credit reporting,
credit repair services, or other personal consumer reports`, and `Mortgage`)
showed that the source adapter preserved provider context such as `Product`,
`Issue`, and `Sub-issue`, but the FAQ classifier did not read that context
when choosing intent/action policy. That caused financial complaint rows to
fall into generic SaaS reporting/account workflows.
PR #699 landed the generic fix: include normalized source-context fields in FAQ
intent classification, add mortgage-servicing policy, and keep mortgage
classification anchored to mortgage-specific source language. This is still
not a CFPB-specific branch; CFPB is the real public fixture that exposed the
generic source-context gap.
