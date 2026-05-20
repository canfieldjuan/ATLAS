# PR-Blog-SEO-Field-Persistence

## Why this slice exists

The AI Content Ops blog generator already asks the LLM for SEO/AEO-facing
fields (`seo_title`, `seo_description`, `target_keyword`,
`secondary_keywords`, and `faq`) and threads them into `BlogPostDraft.metadata`.
The Postgres adapter for the extracted Content Ops blog path then stores that
metadata under `data_context["_metadata"]` instead of writing the first-class
`blog_posts` SEO columns that the public blog API and frontend read.

That creates a product gap: the generated draft can contain SEO-ready data while
the published/public path sees empty SEO columns.

This slice closes only that persistence mismatch. It does not redefine GEO,
add a new quality gate, or change generation prompts.

This slice is slightly over the 400 LOC target because it includes the discovery
note that identified the SEO/AEO/GEO contract gap alongside the focused runtime
fix. The code change remains small, but keeping the evidence note with the
persistence fix gives the reviewer the product context without chasing a
separate branch.

## Scope (this PR)

1. Persist generated blog SEO fields into the existing `blog_posts` SEO columns
   from `PostgresBlogPostRepository.save_drafts(...)`.
2. Add the existing Atlas blog SEO migration to the extracted package's synced
   migration manifest so standalone installs have the columns too.
3. Read those SEO columns back into `BlogPostDraft.metadata` from
   `PostgresBlogPostRepository.list_drafts(...)`.
4. Add focused regression coverage for save and read behavior.
5. Include the discovery note that identified the SEO/AEO/GEO contract gap.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-SEO-Field-Persistence.md` | Plan doc for this slice. |
| `docs/audits/ai_content_ops_blog_seo_aeo_geo_discovery_2026-05-20.md` | Discovery note for current SEO/AEO/GEO state and gaps. |
| `extracted_content_pipeline/blog_post_postgres.py` | Persist and hydrate SEO metadata through first-class columns. |
| `extracted_content_pipeline/manifest.json` | Register the synced blog SEO migration. |
| `extracted_content_pipeline/storage/migrations/120_blog_seo.sql` | Synced copy of the existing Atlas SEO column migration. |
| `tests/test_extracted_blog_post_postgres.py` | Regression coverage for SEO field persistence and hydration. |

## Mechanism

`BlogPostGenerationService._build_draft(...)` already puts the generated SEO
fields in `draft.metadata`. The Postgres adapter will map those metadata values
into the existing `blog_posts` columns during insert/update:

- `seo_title`
- `seo_description`
- `target_keyword`
- `secondary_keywords`
- `faq`

The adapter's list path will select those same columns and merge them back into
`BlogPostDraft.metadata` so generated-asset exports and review helpers see the
same metadata shape regardless of whether the value came from
`data_context["_metadata"]` or a first-class column.

The Atlas migration already exists at `atlas_brain/storage/migrations/120_blog_seo.sql`.
This slice copies that migration into the extracted package and records the
source mapping in `manifest.json`; it does not add a new Atlas migration.

## Intentional

- No new Atlas schema migration is needed. The columns already exist in
  `atlas_brain/storage/migrations/120_blog_seo.sql`; the extracted migration is
  a synced package copy.
- No prompt changes. The generator already requests the SEO fields.
- No GEO validator in this slice. GEO needs a product definition before it can
  become a deterministic pass/fail contract.
- Metadata remains in `data_context["_metadata"]` as well. That preserves the
  existing extracted generated-asset export shape and avoids a breaking change
  for callers already reading metadata there.

## Deferred

- Add a named SEO/AEO/GEO readiness summary to blog review/export output.
- Decide whether GEO is a separate product contract or part of AEO readiness.
- Add quality-gate checks for SEO title length, meta description length, FAQ
  count, answer-first sections, and citable section structure.
- Audit the publish path end to end after this persistence fix lands.

## Verification

- Focused Postgres adapter tests -> 7 passed.
- Python compile check over the edited adapter and test module -> passed.
- Diff whitespace check -> passed.
- Extracted content pipeline validation -> passed.
- Atlas reasoning import guard for extracted_content_pipeline -> passed.
- Standalone extracted audit with fail-on-debt -> passed.
- ASCII Python policy check -> passed.
- Extracted package sync -> refreshed 43 files.
- Full extracted pipeline checks -> 1522 passed, 1 existing torch/pynvml warning.
- Local PR review advisory run before commit -> passed; plan/code consistency skipped because the branch was not committed yet.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan + discovery doc | ~285 |
| Postgres adapter | ~85 |
| Migration + manifest | ~20 |
| Tests | ~135 |
| **Total** | **~525** |

This may land slightly over the 400 LOC target. The overage is justified in
Why this slice exists.
