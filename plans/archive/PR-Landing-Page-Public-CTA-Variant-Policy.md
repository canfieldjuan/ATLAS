# PR: Landing Page Public CTA Variant Policy

## Why this slice exists

PR-Landing-Page-Public-CTA-Index-Policy made the public robots policy reject
the known local placeholder CTA /demo. The review on PR #810 found a narrow
gap: /demo/, /demo?utm=..., and /demo#anchor are still treated as
indexable because the policy compares only the exact string before allowing
local paths.

This slice closes that variant gap at the public index-policy source so a
readiness-passing page cannot be indexed with the same placeholder CTA expressed
with a trailing slash, query string, or fragment.

Ownership lane: content-ops/landing-page-public-index-policy

## Scope (this PR)

1. Normalize same-origin local CTA paths before the public CTA indexability
   decision.
2. Keep /demo and simple variants such as /demo/, /demo?utm=..., and
   /demo#anchor as noindex,follow.
3. Keep real local routes, absolute HTTP(S) URLs, mailto:, and tel: CTAs
   indexable when the draft is otherwise approved and ready.
4. Add helper-level and public API regression coverage for the reviewed gap.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Public-CTA-Variant-Policy.md` | Plan doc for this follow-up CTA variant slice. |
| `extracted_content_pipeline/landing_page_export.py` | Normalize local CTA paths before rejecting the /demo placeholder. |
| `tests/test_extracted_landing_page_export.py` | Cover /demo variants in the robots helper. |
| `tests/test_extracted_content_asset_api.py` | Cover /demo variants through the public generated-asset API. |

## Mechanism

`_public_cta_url_indexable(...)` keeps the existing allowlist shape, but parses
local relative URLs before the final local-path allow. If the URL has no scheme
or netloc, its parsed path is lowercased and stripped of a trailing slash for
the placeholder comparison. That catches /demo/, /demo?utm=..., and
/demo#anchor without turning every absolute URL ending in /demo into a
blocked destination.

## Intentional

- No generator prompt changes.
- No draft quality-gate changes.
- No public route registry or frontend route assumptions beyond the known local
  /demo placeholder.
- Absolute HTTP(S) URLs remain allowed, including external demo-booking URLs,
  because the public policy cannot know whether a third-party /demo path is a
  real destination.

## Deferred

- `HARDENING.md` still tracks landing-page repair legacy-lock rollout cleanup
  and repair lock connection hold time. Both are parked under
  `Owner/session: landing-page repair session` and are not required for this
  public CTA variant-policy slice.
- Anchor-fragment placeholder variants (`/#<frag>`, e.g. `/#section`) remain
  indexable. Only `/demo` gets variant normalization in this slice. Unlike a
  `/demo` variant -- which is unambiguously the demo placeholder -- a
  `/#pricing`-style CTA can be a legitimate same-page anchor on a finished
  page, so blanket-rejecting `/#<frag>` would de-index real pages. The bare
  `#` and `/#` placeholders stay rejected via the exact-match set.

## Parked hardening

- None added.

## Verification

- pytest tests/test_extracted_landing_page_export.py tests/test_extracted_content_asset_api.py -q -> 75 passed.
- bash scripts/local_pr_review.sh -> passed.

## Estimated diff size

| File | Estimated LOC |
| --- | ---: |
| `extracted_content_pipeline/landing_page_export.py` | 20 |
| `tests/test_extracted_landing_page_export.py` | 15 |
| `tests/test_extracted_content_asset_api.py` | 15 |
| `plans/PR-Landing-Page-Public-CTA-Variant-Policy.md` | 70 |
| **Total** | **120** |
