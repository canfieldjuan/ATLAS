# PR: Landing Page Public Prerender Verify

## Why this slice exists

PR #804 made approved generated landing pages prerender into static `/lp/...`
HTML when the public sitemap feed is configured. That closes the crawler-visible
rendering gap, but the build currently has no verifier that fails when sitemap
entries exist without matching static HTML, metadata, JSON-LD, body copy, or CTA
content.

This slice adds a publish-surface guard for generated landing-page prerendering.

Ownership lane: content-ops/landing-page-public-prerender-verify

## Scope (this PR)

1. Add a landing-page prerender verification script for built `dist` output.
2. Check every `/lp/...` URL in `dist/sitemap.xml` has a matching
   `dist/lp/.../index.html`.
3. Verify each generated landing-page HTML file includes title/meta,
   canonical, robots `index,follow`, JSON-LD, prerendered body marker, H1, and
   CTA markup.
4. Add an npm script and wire it into the Atlas Intel UI CI workflow after the
   production build.
5. Keep the verifier a no-op when the build has no generated `/lp/...` sitemap
   entries.
6. Add fixture tests for the verifier's pass, missing-file failure, and no-op
   paths.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Public-Prerender-Verify.md` | Plan doc for this verifier slice. |
| `atlas-intel-ui/scripts/verify-landing-page-geo-prerender.mjs` | Verify built public generated landing-page HTML. |
| `atlas-intel-ui/scripts/verify-landing-page-geo-prerender.test.mjs` | Cover verifier behavior with temporary built-output fixtures. |
| `atlas-intel-ui/package.json` | Add the landing-page GEO verification script. |
| `.github/workflows/atlas_intel_ui_checks.yml` | Run the landing-page verifier in CI after build. |

## Mechanism

The verifier reads `dist/sitemap.xml`, extracts URLs whose path starts with
`/lp/`, maps each URL path to the corresponding static file under `dist`, and
checks the HTML for the crawler-visible artifacts the public landing-page
contract needs.

No generated landing-page feed is required for local or CI builds. If the
sitemap contains no `/lp/...` entries, the verifier reports that and exits
successfully.

## Intentional

- No runtime route changes.
- No build plugin changes.
- No live network calls.
- No public landing-page content fixtures.

## Deferred

- `HARDENING.md` still tracks landing-page repair legacy-lock rollout cleanup
  and repair lock connection hold time. Both are parked under
  `Owner/session: landing-page repair session` and are not required for public
  prerender verification.

## Parked hardening

- None added.

## Verification

- Atlas Intel UI lint -> passed.
- Landing-page prerender verifier test -> 3 passed.
- Atlas Intel UI production build -> passed.
- Blog GEO prerender verification -> verified 14 blog pages.
- Landing-page GEO prerender verification -> passed; no generated landing-page
  sitemap entries found in local build.
- Local PR review -> passed.

## Estimated diff size

| File | Estimated LOC |
| --- | ---: |
| `atlas-intel-ui/scripts/verify-landing-page-geo-prerender.mjs` | 165 |
| `atlas-intel-ui/scripts/verify-landing-page-geo-prerender.test.mjs` | 105 |
| `atlas-intel-ui/package.json` | 3 |
| `.github/workflows/atlas_intel_ui_checks.yml` | 6 |
| `plans/PR-Landing-Page-Public-Prerender-Verify.md` | 75 |
| **Total** | **354** |
