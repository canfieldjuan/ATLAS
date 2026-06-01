# Gap Log: FAQ Generator <-> Landing/Blog Coupling

Date: 2026-05-22
Archived context: this note was preserved from a local worktree cleanup. Treat
it as a historical gap log, not current implementation truth. Current capability
truth lives in the codebase, merged plan docs, and PR history.

Scope: Whether the FAQ generator work (`ticket_faq_*`) is wired to the
landing-page / blog-post generator work, and what FAQ lacks relative to them.
Method: cross-reference greps + read of the content-ops execution framework,
the FAQ generator, the landing/blog generators, and their export/persistence/
public-serving layers at working-tree HEAD.

This log records observed gaps and supporting evidence only. It does not
prescribe a target design or how the pieces should be coupled.

---

## Gap 1 - No code coupling between FAQ and landing/blog (framework siblings only)

- `ticket_faq_markdown.py`, `ticket_faq_export.py`, `ticket_faq_postgres.py`,
  `ticket_faq_input_contract.py` import only FAQ modules + `campaign_ports` /
  `ticket_faq_ports` / `storage._jsonb_helpers`. No `landing_page_*` or
  `blog_*` imports.
- No `landing_page_*` or `blog_*` module references any FAQ symbol.
- The only shared surface is the execution framework, not data:
  - `generated_assets.py:54` `ASSET_CHOICES = ("blog_post", "report",
    "landing_page", "sales_brief", "faq_markdown")` - all five are sibling
    outputs of one `/content-ops/execute` pipeline.
  - `generation_plan.py` has a per-output `_<output>_config_for_request` +
    dispatch branch (`_faq_markdown_config_for_request` at ~210, dispatch at
    ~394; landing at ~166/330; blog at ~193/365).
  - `content_ops_execution.py` holds all outputs in one services bundle.
  - `generated_assets.py` routes export/status per asset (FAQ at lines
    669-718).
- No data flow in either direction: FAQ output does not feed blog/landing,
  and blog/landing output does not feed FAQ.

## Gap 2 - FAQ has no SEO / structured-data / public-web infrastructure that landing and blog have

What landing/blog have and FAQ lacks:

| Capability | Landing / Blog | FAQ (`ticket_faq_*`) |
|---|---|---|
| schema.org JSON-LD | `landing_page_structured_data.py` (`WebPage` + `FAQPage` + `Question`/`Answer`, canonical wiring); blog structured data | none |
| robots policy | `public_landing_page_robots()` (`landing_page_export.py:166`) | none |
| sitemap candidates | `list_public_sitemap_candidates()` (`landing_page_postgres.py`) | none |
| SEO/AEO/GEO readiness | `landing_page_readiness.py`; blog `seo_aeo_readiness`/`geo_readiness` | none |
| slug / canonical URL | landing `meta` + slug | none |
| public web route | `/landing_page/public/sitemap.xml` + `/landing_page/public/{id}` (`generated_assets.py:549/568`); blog via `atlas_brain/api/blog_public.py` | none - FAQ is only returned as markdown text from the execute route and persisted as drafts |

## Gap 3 - FAQ content is deterministic/templated (no LLM)

- The FAQ generator produces answers via a fixed template, not an LLM:
  `ticket_faq_markdown.py:667` ->
  `"answer": f"Customers mention: {snippets} Evidence comes from {len(source_ids)} ticket source(s)."`
- Plan/registry marks the FAQ output `reasoning_requirement="absent"`, whereas
  blog_post / landing_page are `"optional_host_context"`.

## Gap 4 - FAQ signal does not feed the embedded `faq` metadata that blog/landing already carry

- Blog generation already emits an embedded `faq` Q&A list as metadata
  (`blog_generation.py:546` `"faq": _mapping_list(parsed.get("faq"))`, also
  601; persisted in `blog_post_postgres.py` faq column at 100/133/141/174/224).
  This is independent of the ticket-FAQ generator.
- The ticket-FAQ generator's per-item Q&A (`question`, `answer`,
  `question_source`, `source_ids`, frequency - `ticket_faq_markdown.py`
  ~645-667) is not connected to that embedded `faq` field.

---

## Neutral facts relevant to the gaps (not recommendations)

- The schema.org `FAQPage` JSON-LD builder already exists, but inside
  `landing_page_structured_data.py` (line ~87) and is typed to
  `LandingPageDraft`; it builds an embedded FAQ subgraph within a `WebPage`
  node, not a standalone page.
- FAQ output items already carry a `question`/`answer`/`heading`/`source_ids`
  shape (`ticket_faq_markdown.py` ~187, 645-667).
- FAQ has its own persistence/export (`ticket_faq_postgres.py`,
  `ticket_faq_export.py`) but the export columns are minimal (target_id,
  title, markdown, items, status) - no readiness/robots/slug/structured-data
  columns.
