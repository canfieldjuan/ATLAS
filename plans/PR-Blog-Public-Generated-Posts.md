# PR: Blog Public Generated Posts

## Why this slice exists

PR #1223 finished the landing-page generation-to-review handoff. The matching
blog path is still not productized: Content Ops can generate and approve
`blog_post` drafts, and the backend already exposes public blog endpoints, but
those endpoints only read `status = 'published'` while generated-asset review
sets drafts to `approved`. The React `/blog` and `/blog/:slug` pages also still
render only static `POSTS`, so approved generated blog posts do not show up in
the public product surface.

This slice makes approved generated blog posts visible on the public blog
pages while keeping the existing static posts as a safe fallback.

## Scope (this PR)

Ownership lane: content-ops/blog-public-productization

Slice phase: Vertical slice

1. Update the public blog API to treat `approved` generated posts as public
   alongside legacy `published` posts.
2. Add a small frontend public-blog API adapter for `/api/v1/blog/published`
   and `/api/v1/blog/published/{slug}`.
3. Make `/blog` load generated public posts at runtime and merge them ahead of
   static posts by slug.
4. Make `/blog/:slug` fetch generated posts when the slug is not static, while
   preserving the static rendering path for existing prerendered content.
5. Add focused backend and frontend tests for approved visibility and runtime
   generated-post fetching.
6. Enroll both focused suites in CI.

### Files touched

- `atlas_brain/api/blog_public.py`
- `tests/test_blog_public.py`
- `atlas-intel-ui/src/api/blog.ts`
- `atlas-intel-ui/src/pages/Blog.tsx`
- `atlas-intel-ui/src/pages/BlogPost.tsx`
- `atlas-intel-ui/scripts/blog-public-generated-posts.test.mjs`
- `atlas-intel-ui/package.json`
- `.github/workflows/atlas_intel_ui_checks.yml`
- `.github/workflows/atlas_blog_public_checks.yml`
- `plans/PR-Blog-Public-Generated-Posts.md`

## Mechanism

The backend public query keeps the existing `/blog/published` route names for
compatibility, but its public status predicate becomes:

```sql
status IN ('published', 'approved')
```

The frontend adapter converts the backend wire shape into the existing
`BlogPost` view model. The list page starts with static `POSTS`, fetches public
generated posts, and merges by slug with generated posts taking precedence. The
detail page uses the static post immediately when present; otherwise it fetches
the slug from the public API and renders the same `BlogPost` template.

## Intentional

- No generation, review, or blog-admin rewrite. This is the public read path
  for already generated and approved posts.
- Static posts remain in place for existing prerendered SEO pages and as a
  fallback if the public API is unavailable.
- The backend route names remain `/published` to avoid breaking existing
  callers; only the public eligibility predicate expands to include approved
  generated-asset drafts.

## Deferred

- Full static prerender/sitemap ingestion for generated blog posts remains a
  later SEO/GEO hardening slice. This vertical slice makes the public pages
  usable at runtime.
- A consolidated frontend API fetch helper remains out of scope; existing
  modules already duplicate small fetch wrappers.
- Parked hardening: none. `HARDENING.md` and `ATLAS-HARDENING.md` were scanned;
  current blog entries focus on generator content quality, not this public
  read path.

## Verification

- `pytest tests/test_blog_public.py -q` - 4 passed, 1 existing torch CUDA
  warning from the local environment.
- `cd atlas-intel-ui && npm ci` - passed; npm reports the existing 6 audit
  findings already parked in `HARDENING.md`.
- `cd atlas-intel-ui && npm run test:blog-public-generated-posts` - 3
  passed.
- `cd atlas-intel-ui && npm run test:content-ops-landing-page-e2e-ui` - 4
  passed.
- `cd atlas-intel-ui && npm run test:landing-page-prerender` - 4 passed.
- `cd atlas-intel-ui && npm run build` - passed; TypeScript and Vite build
  completed.
- `cd atlas-intel-ui && npm run verify:blog-geo` - verified 14 blog pages.
- `bash scripts/local_pr_review.sh --current-pr-body-file
  /home/juan-canfield/Desktop/blog-public-generated-posts-pr-body.md` -
  passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~95 |
| Backend API + tests | ~45 |
| Frontend API adapter | ~85 |
| Blog pages | ~120 |
| Frontend tests + CI enrollment | ~75 |
| Backend CI enrollment | ~35 |
| **Total** | **~455** |

The slice may slightly exceed the 400 LOC target because the backend status
predicate and both frontend public routes must land together to produce a usable
generated-blog public path, and the host API test must be enrolled in a
dedicated Atlas workflow.
