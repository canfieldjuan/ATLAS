# PR: Landing Page Prerender Env Docs

## Why this slice exists

Generated landing-page public prerendering now depends on build-time
configuration. The code supports `VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL` and
`VITE_API_BASE`, but the Atlas Intel UI has no local README explaining that
deployment contract, and the Vite env type declaration only lists
`VITE_API_BASE`.

This slice documents the deployment knobs that make generated landing-page
prerendering active.

Ownership lane: content-ops/landing-page-prerender-env-docs

## Scope (this PR)

1. Add a focused Atlas Intel UI README section for public landing-page
   prerendering.
2. Document when to set `VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL`.
3. Document how `VITE_API_BASE` is used for public page payload fetches.
4. Add the missing Vite env type for `VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL`.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Prerender-Env-Docs.md` | Plan doc for this env-doc slice. |
| `atlas-intel-ui/README.md` | Document Atlas Intel UI build and public landing-page prerender env vars. |
| `atlas-intel-ui/src/vite-env.d.ts` | Add the public landing-page sitemap env type. |

## Mechanism

The README records the production setup:

- `VITE_API_BASE` points browser and build-time public payload fetches at the
  Atlas backend origin.
- `VITE_PUBLIC_LANDING_PAGE_SITEMAP_URL` points the Vite build at the backend
  public generated landing-page sitemap feed.
- If the sitemap URL is absent, generated landing-page prerendering is skipped
  and the verifier no-ops.

## Intentional

- No runtime code changes.
- No build plugin changes.
- No CI behavior changes.

## Deferred

- `HARDENING.md` still tracks landing-page repair legacy-lock rollout cleanup
  and repair lock connection hold time. Both are parked under
  `Owner/session: landing-page repair session` and are not required for env
  docs.

## Parked hardening

- None added.

## Verification

- TypeScript build -> passed.
- Local PR review -> passed.

## Estimated diff size

| File | Estimated LOC |
| --- | ---: |
| `atlas-intel-ui/README.md` | 55 |
| `atlas-intel-ui/src/vite-env.d.ts` | 1 |
| `plans/PR-Landing-Page-Prerender-Env-Docs.md` | 65 |
| **Total** | **121** |
