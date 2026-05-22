# PR: Landing Page Prerender Fallback Title Guard

## Why this slice exists

The generated landing-page prerender verifier rejects the default SPA title so a
broken prerender cannot ship crawler-visible pages with generic Atlas metadata.
The guard currently checks the ASCII-hyphen spelling, while `index.html` uses an
em dash. That leaves the real fallback title undetected.

This slice fixes the verifier so it catches the actual fallback title.

Ownership lane: content-ops/landing-page-public-prerender-verify

## Scope (this PR)

1. Centralize fallback-title detection in the landing-page prerender verifier.
2. Reject both the legacy ASCII-hyphen fallback and the current em-dash
   fallback without putting non-ASCII source text in the script.
3. Add a fixture test proving the current fallback title fails verification.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Prerender-Fallback-Title-Guard.md` | Plan doc for this verifier guard slice. |
| `atlas-intel-ui/scripts/verify-landing-page-geo-prerender.mjs` | Detect current and legacy fallback title variants. |
| `atlas-intel-ui/scripts/verify-landing-page-geo-prerender.test.mjs` | Cover current fallback-title failure. |

## Mechanism

The verifier now uses an `isFallbackTitle(...)` helper that accepts either a
hyphen or `\u2014` between the Atlas brand and fallback title text. The test
uses a temporary built-output fixture with the current fallback title and
asserts that verification fails.

## Intentional

- No build plugin changes.
- No runtime route changes.
- No changes to the public landing-page rendering contract.

## Deferred

- `HARDENING.md` still tracks landing-page repair legacy-lock rollout cleanup
  and repair lock connection hold time. Both are parked under
  `Owner/session: landing-page repair session` and are not required for this
  verifier guard.

## Parked hardening

- None added.

## Verification

- Landing-page prerender verifier test -> 4 passed.
- Atlas Intel UI production build -> passed.
- Landing-page GEO prerender verification -> passed; no generated landing-page
  sitemap entries found in local build.
- Local PR review -> passed.

## Estimated diff size

| File | Estimated LOC |
| --- | ---: |
| `atlas-intel-ui/scripts/verify-landing-page-geo-prerender.mjs` | 12 |
| `atlas-intel-ui/scripts/verify-landing-page-geo-prerender.test.mjs` | 25 |
| `plans/PR-Landing-Page-Prerender-Fallback-Title-Guard.md` | 65 |
| **Total** | **102** |
