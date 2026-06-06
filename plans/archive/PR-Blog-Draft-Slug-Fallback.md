# PR-Blog-Draft-Slug-Fallback

Ownership lane: `content-ops/blog-live-smoke`

## Why this slice exists

The live Haiku blog smoke generated one draft and passed quality checks, but
returned no saved draft ids. The generated slug already existed under another
smoke account, and `blog_posts.slug` is globally unique. The extracted blog
repository intentionally avoided cross-tenant overwrites, but it silently
reported no save instead of storing the new tenant's draft under a safe alternate
slug.

## Scope (this PR)

1. Make `PostgresBlogPostRepository.save_drafts` retry blocked slug writes with a
   deterministic tenant-scoped fallback slug.
2. Preserve the existing guard that prevents cross-tenant or published-row
   overwrites.
3. Update focused repository tests to prove the live-smoke failure mode returns a
   saved draft id.
4. Warn if every bounded fallback attempt is still blocked, so a future shortfall
   does not become another silent save mismatch.

### Files touched

- `extracted_content_pipeline/blog_post_postgres.py`
- `tests/test_extracted_blog_post_postgres.py`
- `plans/PR-Blog-Draft-Slug-Fallback.md`

## Mechanism

`save_drafts` still attempts the requested draft slug first. If Postgres returns
no id because the global slug conflict was blocked by the scoped `WHERE`, the
repository retries with `"{base_slug}-{account_slug}"` and then numbered
variants. The same insert/upsert SQL is used for every attempt, so the tenant and
published-row safety guard remains load-bearing. If every bounded attempt is
blocked, the repository leaves the draft unsaved and emits a warning with the
base slug, account id, and attempt count.

## Intentional

- No schema migration in this slice. The legacy table has a global slug
  uniqueness contract, and older blog writers still use `ON CONFLICT (slug)`.
- No cross-tenant overwrite. A blocked slug becomes an alternate slug for the new
  draft, not an update to the existing row.

## Deferred

- Parked hardening: none.
- Replacing the legacy global slug uniqueness model with a composite
  account/slug contract would be a broader migration and caller audit. This PR
  fixes the current production save path without changing public slug semantics.

## Verification

- `pytest tests/test_extracted_blog_post_postgres.py -q` - 9 passed.
- `python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_content_ops_custom_blog_smoke_haiku_diagnostics --user-id codex-smoke --blog-blueprint-json /tmp/atlas-custom-blog-blueprint.json --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file /tmp/atlas-haiku-override.env --output-result /tmp/atlas-custom-blog-smoke-haiku-diagnostics-result-after-fallback.json --json` - passed and returned saved draft id `7ff3c7b9-e6e5-476c-8a73-47d55d3e3adc`.
- Export check for `acct_content_ops_custom_blog_smoke_haiku_diagnostics` - confirmed draft slug `how-support-tickets-become-faq-answers-customers-can-actually-find-acct-content-ops-custom-blog-smoke-haiku`.
- Extracted content pipeline validation wrapper - passed.
- Extracted reasoning-import guard - passed.
- Extracted standalone audit with debt failure enabled - passed.
- Extracted Python ASCII check - passed.
- Extracted content pipeline sync wrapper - passed.

## Estimated diff size

| Area | Estimate |
|---|---:|
| Blog Postgres repository slug fallback + warning | ~130 LOC |
| Repository tests | ~50 LOC |
| Plan doc | ~80 LOC |
| **Total** | **~325 LOC** |
