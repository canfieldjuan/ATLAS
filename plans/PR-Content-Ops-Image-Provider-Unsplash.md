# PR: Content Ops Image Provider Unsplash

## Why this slice exists

Plan PR #1238 identified the next product gap after the landing/blog upload
handoff: generated landing pages and blog posts have public review/publish
paths, but no host-owned way to attach a real visual asset. Landing-page drafts
already expose `hero.image_url` and `meta.og_image_url`, and blog drafts have
flexible metadata, so the smallest shippable slice is a provider port plus one
concrete free-first adapter that enriches drafts before persistence.

This PR implements the Unsplash half of that plan. It keeps paid AI fallback
behind the same port but defers the Flux/OpenRouter adapter until the free path
is proven and reviewed.

## Scope (this PR)

Ownership lane: content-ops/image-provider

Slice phase: Vertical slice

1. Add a package-owned content image provider port and Unsplash adapter that
   searches one landscape image using a typed config object and mocked
   transport in tests.
2. Wire the optional provider into landing-page and blog-post generation so
   successful image lookup enriches drafts before save:
   - landing page: `hero.image_url`, `meta.og_image_url`, `metadata.content_image`
   - blog post: `metadata.cover_image`
3. Keep image lookup best-effort: missing config, no results, malformed
   response, download-tracking failure, or provider exception yields the
   original draft rather than failing content generation.
4. Add typed host config fields under the existing Content Ops/B2B campaign
   settings surface using `ATLAS_CONTENT_OPS_IMAGE_*` aliases; no direct
   environment reads.
5. Add focused tests for adapter request shape, Unsplash attribution/download
   tracking, draft enrichment, and provider failure fallback.

### Files touched

- `extracted_content_pipeline/content_image_provider.py`
- `extracted_content_pipeline/landing_page_generation.py`
- `extracted_content_pipeline/blog_generation.py`
- `extracted_content_pipeline/manifest.json`
- `atlas_brain/config.py`
- `atlas_brain/_content_ops_services.py`
- `tests/test_extracted_content_image_provider.py`
- `tests/test_extracted_landing_page_generation.py`
- `tests/test_extracted_blog_generation.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `plans/PR-Content-Ops-Image-Provider-Unsplash.md`

## Mechanism

`ContentImageProvider` exposes one async method:

```python
async def select_image(request: ContentImageRequest) -> ContentImageAsset | None: ...
```

`UnsplashContentImageProvider` uses the official public-auth shape
(`Authorization: Client-ID <access-key>`) against `/search/photos`, selects the
first result, then calls the returned `links.download_location` before returning
the hotlinked URL and attribution. Tests mock the opener and assert both
requests, so CI never calls Unsplash.

Generation services accept `image_provider: ContentImageProvider | None`.
When absent, behavior is unchanged. When present, the service builds a compact
query from trusted draft/campaign/blueprint fields, calls the provider, and
copies the returned asset into draft metadata. Provider errors are caught
locally because visual enrichment must not fail an otherwise successful content
generation run.

## Intentional

- Unsplash only in this slice. Flux/OpenRouter fallback remains deferred behind
  the same port because OpenRouter image-generation request/response shape and
  cost accounting need their own focused tests.
- No schema migration. Landing pages already have image URL fields in JSONB
  hero/meta, and blog posts already have JSONB metadata.
- No client-side Unsplash key exposure. The provider runs server-side and is
  wired only from typed host config.
- No live API calls in CI. The adapter is transport-mocked.

## Deferred

- Future PR: Flux/OpenRouter fallback adapter with cost metadata and mocked
  provider transport.
- Future PR: UI rendering polish for blog cover images if the current public
  template does not surface `metadata.cover_image`.
- Parked hardening: none. `HARDENING.md` was scanned; the current entries do
  not touch the content-ops/image-provider lane.

## Verification

- `pytest tests/test_extracted_content_image_provider.py tests/test_extracted_landing_page_generation.py tests/test_extracted_blog_generation.py -q`
  - 124 passed.
- `python -m py_compile extracted_content_pipeline/content_image_provider.py extracted_content_pipeline/landing_page_generation.py extracted_content_pipeline/blog_generation.py atlas_brain/_content_ops_services.py atlas_brain/config.py`
  - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main`
  - OK: 142 matching tests are enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh`
  - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - Atlas runtime import findings: 0.
- `bash scripts/check_ascii_python.sh`
  - ASCII check passed for extracted_content_pipeline Python files.
- `bash scripts/run_extracted_pipeline_checks.sh`
  - 2911 passed, 10 skipped, 1 warning; all extracted content pipeline checks completed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-image-provider-unsplash-pr-body.md`
  - local PR review passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~129 |
| Provider port + adapter | ~238 |
| Generator integration | ~129 |
| Host config/wiring | ~49 |
| Tests + CI enrollment | ~323 |
| **Total** | **~878** |

This is above the soft cap because the provider, generation integration, host
typed-config wiring, manifest enrollment, and failure-detection tests need to
ship together; splitting the adapter from the draft enrichment would leave an
unused provider or an untested generation hook. The Flux/OpenRouter fallback is
deferred specifically to keep this already-over-budget vertical slice bounded.
