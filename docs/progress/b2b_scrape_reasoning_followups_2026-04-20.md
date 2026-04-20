# B2B Scrape + Reasoning Follow-Ups

Date: 2026-04-20

## Closed Decisions

- Keep `anthropic/claude-sonnet-4-5` as the default model for vendor reasoning synthesis.
- Keep `anthropic/claude-sonnet-4-5` as the default model for pairwise cross-vendor reasoning.
- Keep `anthropic/claude-opus-4.6` as a manual premium override only, not the default path.
- Keep `vLLM` as first-tier enrichment on `localhost:8082`.

## Current State

- Scraping is materially more robust on the structured review sources.
- Vendor refresh works end-to-end from shell and app paths.
- Manual/scoped reasoning synthesis works synchronously.
- Vendor confidence is now real numeric confidence, not hard-mapped `medium -> 0.55`.
- Product-facing validation vendors currently include:
  - `Zoho Desk`
  - `Zendesk`
  - `Pipedrive`
  - `Monday.com`
  - `Klaviyo`
  - `RingCentral`
  - `DigitalOcean`
  - `HubSpot`

## Next Work

1. Pre-fetch scrape dedupe / high-water stopping
- Add source-specific stop logic before deeper pagination so we stop paying to re-fetch known review pages.
- Priority sources:
  - `capterra`
  - `gartner`
  - `trustradius`
  - `software_advice`
- Goal:
  - stop after known-page / old-page detection earlier than current fetch-then-stop behavior.

2. Cross-vendor reasoning rollout policy
- Cross-vendor reasoning is still effectively manual/scoped.
- Decide the gating for broader re-enable:
  - minimum vendor evidence quality
  - pairwise confidence floor
  - budget/cost guardrails
- Do not re-enable globally until those thresholds are explicit.

3. Continue fresh vendor refresh from new enriched wave
- Use vendor-scoped refresh on newly enriched vendors with healthy source mix.
- Validate:
  - battle-card usefulness
  - reasoning confidence spread
  - whether accounts-in-motion remains empty or starts to become useful

4. Accounts in motion validation
- Current persisted rows exist but are mostly empty.
- Investigate whether this is:
  - expected weak account signal
  - thresholding too strict
  - downstream assembly issue

5. Optional UI/API follow-up
- If premium manual reasoning runs remain useful, expose model override cleanly in UI/API.
- Keep default behavior pinned to `Sonnet 4.5`.

## Things Not To Revisit Without New Evidence

- Do not switch default vendor reasoning to `Opus 4.6`.
- Do not switch default pairwise cross-vendor reasoning to `Opus 4.6`.
- Do not raise scheduled maintenance depth globally beyond the current selective policy.
- Do not spend more repair cycles on degraded `g2` maintenance until provider quality improves.
