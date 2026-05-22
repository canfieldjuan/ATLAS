# PR-Landing-Page-SEO-GEO-AEO-Input-Contract

## Why this slice exists

The current landing-page pipeline can generate, quality-gate, repair, review,
publish, index, and include generated pages in the frontend sitemap. The code
audit after PR #764 showed the remaining source-level gap is input capture:
operators still do not have a clean first-class way to provide SEO/GEO/AEO
landing-page inputs, and only a narrow allowlist survives into
`MarketingCampaign.context`.

This slice adds the input contract before changing prompts, validators,
editing, or public prerendering.

## Current State From Code

- Generation already has save-time quality gates and a repair loop through
  `LandingPageGenerationService`.
- Export/public routes already compute `seo_aeo_readiness` and
  `geo_readiness`.
- The review UI already renders landing-page readiness panels, repair history,
  structured data, sections, references, public URL state, and raw row.
- Static sitemap bridging is merged in PR #764.
- `content_ops_execution._marketing_campaign_from_inputs` still only allows
  `industry`, `pain_points`, `differentiators`, `customer_segments`,
  `key_metrics`, `proof_points`, and `competitive_alternatives` into
  `MarketingCampaign.context`.
- The new-run UI still primarily relies on free-form JSON inputs for
  landing-page-specific fields.

## Scope (this PR)

Ownership lane: content-ops/landing-page-input-contract

This planning PR locks the next implementation slice after PR #764. It does
not implement the input contract yet.

1. Document the verified current state after the landing-page public rendering,
   robots, sitemap feed, and frontend sitemap bridge work.
2. Define the next implementation scope around first-class SEO/GEO/AEO inputs.
3. Record the intended input keys, implementation files, verification, and
   deferred follow-up slices.

### Files touched

- `plans/PR-Landing-Page-SEO-GEO-AEO-Input-Contract.md`

## Implementation Scope (Next PR)

1. Define first-class landing-page SEO/GEO/AEO input keys.
2. Add those keys to the backend landing-page context allowlist.
3. Expose the input contract through the control-surfaces API/catalog.
4. Add structured `atlas-intel-ui` controls for the landing-page input fields.
5. Prove those fields survive from UI/request inputs into
   `MarketingCampaign.context`.

### Candidate Input Keys

- `target_keyword`
- `secondary_keywords`
- `search_intent`
- `primary_entity`
- `audience_entity`
- `competitors`
- `objections`
- `faq_questions`
- `source_period`
- `internal_links`
- `cta_label`
- `cta_url`

Already-supported context keys should remain supported:

- `industry`
- `pain_points`
- `differentiators`
- `customer_segments`
- `key_metrics`
- `proof_points`
- `competitive_alternatives`

### Files Expected In Implementation

- `plans/PR-Landing-Page-SEO-GEO-AEO-Input-Contract.md`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/control_surfaces.py`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/domain/contentOps/*`
- focused backend/frontend tests

## Mechanism

Backend control surfaces should publish a landing-page input contract so the UI
does not hard-code undocumented fields. The execution layer should treat those
fields as explicit landing-page context and pass them into
`MarketingCampaign.context`, without widening back into the previous
negative-list leak pattern.

UI controls should write normal request `inputs` values. They should coexist
with the raw JSON editor and keep JSON as the transport, but operators should
not need to hand-author these fields for normal landing-page generation.

## Intentional

- No prompt behavior changes in this slice beyond making fields available to
  the existing `{campaign_json}` payload.
- No readiness validator changes.
- No edit/PATCH route.
- No review-drawer repair button.
- No public prerendering changes.

## Deferred

- `PR-Landing-Page-Prompt-Consumes-SEO-GEO-AEO-Inputs` should explicitly teach
  the generation prompt how to use these new fields.
- `PR-Landing-Page-Readiness-Validator-Unification` should centralize the
  export/public readiness checks and save-time quality gate.
- `PR-Landing-Page-Editable-Drafts` should add PATCH + structured edit UI.
- `PR-Landing-Page-Review-Repair-Action` should add operator-triggered repair
  for already-saved drafts.
- `PR-Landing-Page-Public-Prerender` remains later, after input quality and
  validator consistency are tightened.

## Verification

For this planning PR:

- Git whitespace check.
- Local PR review wrapper.

For the implementation PR:

- Backend tests for context allowlist behavior.
- Control-surface tests proving the input contract is exposed.
- UI/domain tests proving typed controls round-trip into request inputs.
- Existing content-ops execution tests.
- Existing `atlas-intel-ui` lint/build checks.
- Local PR review wrapper.

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Plan | ~145 |
| **Total** | **~145** |
