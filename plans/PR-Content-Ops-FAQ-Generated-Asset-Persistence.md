Plan: PR-Content-Ops-FAQ-Generated-Asset-Persistence

## Why this slice exists

PR-Content-Ops-FAQ-Generated-Asset-Seam made `faq_markdown` runnable through
AI Content Ops execution, but the result is still ephemeral: `/execute` returns
Markdown and then loses it. That is useful for smoke testing, not for a hosted
product workflow where operators expect generated outputs to be reviewed,
approved, exported, or published later.

This slice adds the persistence seam only. It keeps the deterministic FAQ
builder intact, saves generated FAQ Markdown drafts when a repository is
configured, and reports saved IDs in the execution result. Generated-asset API,
export CLI, review CLI, and frontend switchboard support are intentionally
deferred so this PR stays focused on the runtime persistence contract.

This PR is over the 400 LOC target because the persistence seam needs the full
generated-asset storage contract in one place: port, Postgres adapter,
migration, service save hook, host wiring, and regression tests. Splitting the
adapter from the service hook would leave either untested storage code or a
save path with no concrete host implementation.

## Scope (this PR)

1. Add a `TicketFAQDraft` / `TicketFAQRepository` port for persisted FAQ
   Markdown drafts.
2. Add a Postgres adapter and owned migration for the `ticket_faq_markdown`
   table.
3. Extend `TicketFAQMarkdownService` with an optional repository; if configured
   and the builder generated FAQ items, save one draft and expose `saved_ids`.
4. Wire Atlas's Content Ops service factory to use the Postgres FAQ repository
   when DB-backed services are enabled, while preserving deterministic no-DB
   execution.
5. Update tests/docs/coordination for the new persisted seam.

### Files touched

- `atlas_brain/_content_ops_services.py`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `extracted_content_pipeline/ticket_faq_ports.py`
- `extracted_content_pipeline/ticket_faq_postgres.py`
- `atlas_brain/storage/migrations/325_ticket_faq_markdown.sql`
- `extracted_content_pipeline/storage/migrations/325_ticket_faq_markdown.sql`
- `plans/PR-Content-Ops-FAQ-Generated-Asset-Persistence.md`
- `tests/test_atlas_content_ops_execution_services.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_extracted_ticket_faq_postgres.py`

## Mechanism

`TicketFAQMarkdownService` keeps the same public `generate(...)` signature. The
constructor gains an optional `ticket_faqs` repository. When `generate(...)`
produces at least one FAQ item and `ticket_faqs` is configured, it builds a
single `TicketFAQDraft` containing the Markdown, items, source counts,
normalization warnings, and output checks, then persists it with
`ticket_faqs.save_drafts(...)`.

Without a repository, behavior remains exactly as the previous seam shipped:
the service returns Markdown and metadata with `saved_ids=()`. This lets the
CLI/offline and no-DB host paths remain deterministic.

The Postgres adapter mirrors the existing generated-asset adapters:
tenant-scoped `save_drafts`, `list_drafts`, `update_status`, and
`update_statuses`. The Atlas migration is canonical because Atlas startup runs
`atlas_brain/storage/migrations`; the extracted package receives a synced copy
through `manifest.json`.

## Intentional

- No generated-asset API or CLI switchboard update in this PR. Persisting first
  gives the next slice a concrete repository contract to route to.
- One persisted draft per FAQ generation run, not one row per FAQ item. The
  Markdown document is the asset; individual FAQ entries stay in `items` JSONB.
- No LLM, quality pack, or reasoning pass. The output remains extractive and
  grounded in support-ticket source rows.
- The default no-DB service remains wired so `/execute` can still return FAQ
  Markdown in lightweight host installs.

## Deferred

- Add `faq_markdown` to generated-asset API list/export/review routes.
- Add `faq_markdown` to generated-asset export/review CLIs.
- Add frontend review/export UI support for the persisted FAQ asset.
- Add optional public/help-center FAQ render mode after internal Markdown
  review quality is validated.

## Verification

- Focused pytest sweep over FAQ service, FAQ Postgres adapter, and Atlas host
  service bundle -> 34 passed
- Python compile sweep over edited modules/tests -> passed
- validate_extracted_content_pipeline -> passed
- forbid_atlas_reasoning_imports -> passed
- audit_extracted_standalone --fail-on-debt -> passed
- check_ascii_python -> passed
- git diff --check -> passed
- run_extracted_pipeline_checks -> 1482 passed, 1 existing torch/pynvml warning

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Ports, Postgres adapter, migration | ~295 |
| Service + host wiring | ~95 |
| Tests | ~255 |
| Docs + coordination | ~70 |
| **Total** | **~715** |
