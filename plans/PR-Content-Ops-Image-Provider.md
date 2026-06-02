# PR: Content Ops Image Provider (Unsplash-first, Flux fallback)

> Scoping plan for a future vertical slice. This PR lands the plan only.
> The pattern is ported from a standalone website-redesign skill
> (its `images.py` module) that already does Unsplash-first, paid-AI-fallback hero
> images; Atlas has no image capability today, and its content outputs
> (landing pages, blog posts, future socials) need one. The skill's pattern
> reuses OpenRouter, which Atlas already wires.

## Why this slice exists

Atlas content-ops generates landing pages and blog posts (and, on the roadmap,
social copy), but has **no image capability** -- a trace of `origin/main` found
no Unsplash, no Flux, no text-to-image anywhere in `extracted_content_pipeline`
or `atlas_brain`. Every image-bearing output currently ships text-only.

A standalone website-redesign skill solved this cleanly with a two-path image
provider: free Unsplash photography first, paid AI generation only as fallback.
That pattern ports well because Atlas already has the expensive half:
`extracted_llm_infrastructure/services/llm/openrouter.py` (`OpenRouterLLM`,
base URL `https://openrouter.ai/api/v1`) provides the OpenRouter key + provider
plumbing and the package has LLM cost tracking. So the slice is a port + adapt,
not a new integration.

Validation precondition: only build this once an image-bearing content output is
actually in front of a buyer. The image is an enhancement; do not port it
speculatively.

## Why Unsplash-first matters

Free, real photography covers most B2B hero/cover needs and costs nothing; paid
AI generation is the fallback for when no stock photo fits. Always-AI is more
expensive and often reads as AI slop for B2B. Preserving the free-first ordering
is the cost discipline this slice exists to keep.

## Scope (this PR)

Ownership lane: content-ops/image-provider
Slice phase: Workflow/process

This PR lands the scoping plan only. The implementing slice (separate PR) will:

1. Add a content-ops image provider with an Unsplash-first search/download path
   and a Flux-via-OpenRouter fallback path, adapted from the website-redesign
   skill's image module.
2. Reuse Atlas's existing OpenRouter key/provider config
   (`extracted_llm_infrastructure/services/llm/openrouter.py`) for the Flux call
   and route the paid generation through the package's existing LLM cost
   tracking -- do not hand-roll a second OpenRouter client.
3. Add typed `ATLAS_*` config (Unsplash access key, image model, an enable flag)
   via `atlas_brain/config.py`; image generation is opt-in.
4. Attach the resulting image to the generated output's image slot
   (landing hero / blog cover), adding the field where the output schema lacks
   one.
5. Tests that mock both transports; no live Unsplash/OpenRouter calls in CI;
   same-PR workflow enrollment (the AGENTS section 3e rule).

### Explicitly NOT in scope

- The website-redesign skill's full pipeline, theme system, Vercel deploy, and
  pitch-email layers (a separate adjacent product).
- Templated data-driven cards (stat/quote/comparison) -- a different image
  output, its own slice.
- Per-platform social image sizing and logo/brand overlays.

### Files touched

- `plans/PR-Content-Ops-Image-Provider.md`

## Mechanism

The provider exposes one entry point that, given a search query / generation
prompt and an output location, returns an image reference or `None`:

1. **Unsplash first** -- if a key is configured, search Unsplash, download the
   chosen photo, store it, and return the reference plus attribution
   (`credit_url`).
2. **Flux fallback** -- if Unsplash returns nothing, call the configured image
   model through the existing OpenRouter plumbing, decode the base64 result to
   storage, and return the reference.
3. **Best-effort** -- if both are unavailable or fail, return `None`; the
   content generation proceeds without an image and never fails on the image
   step.

## Intentional

- **Best-effort, never fail the content run.** The image is an enhancement;
  Unsplash/Flux errors degrade to "no image," they do not break generation
  (the build-resilience lesson from #1227's sitemap bridge).
- **Free-first ordering preserved** to minimize paid Flux calls; paid calls go
  through existing cost tracking.
- **Reuse, do not re-fork** the OpenRouter client config -- a new sibling client
  has drifted weaker before (#1224/#1227).
- **Unsplash attribution preserved** (`credit_url`) for licensing.

## Deferred

- Templated data-driven graphics (stat/quote/comparison cards) -- the
  socials-specific image play, a separate slice.
- Per-platform sizing, brand/logo overlays, and the skill's theme/layout system
  for visually upgrading landing pages.

## Parked hardening

None.

## Verification

- Unsplash path: with a key + a search result, the provider stores a photo and
  returns its reference plus attribution (mocked transport).
- Flux fallback: when Unsplash returns nothing, the provider calls the image
  model through the OpenRouter plumbing and stores the decoded result (mocked
  transport).
- Best-effort: with neither available, the provider returns `None` and the
  content generation still completes -- demonstrated, not assumed.
- Paid generation is recorded by the existing cost tracking.
- No live Unsplash/OpenRouter calls in CI; the test is enrolled in its workflow
  in the same PR.

## Estimated diff size

| Area | LOC |
|---|---:|
| This PR (plan doc) | ~115 |
| **Total** | ~115 |

This PR is plan-only (~115 LOC). The implementing slice is estimated at
~200-350 LOC (provider + config + output image slot + tests), porting and
adapting the website-redesign skill's image logic onto Atlas's OpenRouter
plumbing; split the provider and the output wiring into two PRs if over budget.
