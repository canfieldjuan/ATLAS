# PR-Landing-Page-Prompt-Consumes-SEO-GEO-AEO-Inputs

## Why this slice exists

PR #768 adds first-class landing-page SEO/GEO/AEO inputs to the control
surface and threads them into `MarketingCampaign.context`. That closes the
input-capture gap, but the bundled landing-page prompt still only gives broad
readiness guidance. The generator can receive fields such as
`target_keyword`, `search_intent`, `faq_questions`, `objections`, `cta_label`,
and `cta_url`, but the prompt does not explicitly tell the model how to use
them.

This slice makes the existing prompt consume those optional context fields so
generated landing pages can use operator-supplied SEO, GEO, and AEO inputs
without changing the output schema or readiness validators.

## Scope (this PR)

Ownership lane: content-ops/landing-page-prompt-contract

1. Update the packaged landing-page generation prompt with explicit
   instructions for optional `campaign.context` SEO/GEO/AEO fields.
2. Keep the existing JSON output contract unchanged.
3. Preserve the source-row evidence and fake-proof protections already in the
   prompt.
4. Add prompt-registry tests that pin the new SEO/GEO/AEO instructions.
5. Add a generator test proving campaign context fields still reach the system
   prompt payload the model receives.

### Files touched

- `plans/PR-Landing-Page-Prompt-Consumes-SEO-GEO-AEO-Inputs.md`
- `extracted_content_pipeline/skills/digest/landing_page_generation.md`
- `tests/test_extracted_campaign_skill_registry.py`
- `tests/test_extracted_landing_page_generation.py`

## Mechanism

The landing-page generator already serializes the full campaign payload into
the system prompt through `{campaign_json}`. This PR does not add a new runtime
mapping layer. Instead, it makes the packaged prompt treat
`campaign.context` as the location for optional operator-supplied landing-page
SEO/GEO/AEO inputs.

The prompt will instruct the model to:

- use `target_keyword` in visible copy and `meta.title_tag` when it is present;
- use `secondary_keywords` naturally, without keyword stuffing;
- align hero/subheadline and the first answer-shaped sections with
  `search_intent`;
- use `primary_entity` and `audience_entity` to clarify the offer and audience;
- cover supplied `objections` and `faq_questions` in objection/FAQ sections;
- respect supplied `cta_label` and `cta_url`;
- use `source_period` as freshness context when relevant;
- use `internal_links` only as real page links when they are supplied;
- avoid inventing competitors, proof, or sources when those fields are absent.

## Intentional

- No schema changes. The existing output shape is still the contract.
- No quality-gate changes. Validator unification is a separate slice.
- The prompt now depends on PR #768's shared contract keys in tests so prompt
  coverage fails if the input contract drifts.
- No generated-page edit path. Editable drafts remain a later slice.

## Deferred

- `PR-Landing-Page-Readiness-Validator-Unification` should centralize
  export/public readiness checks and save-time quality gates.
- `PR-Landing-Page-Editable-Drafts` should add PATCH + structured edit UI.
- `PR-Landing-Page-Review-Repair-Action` should add operator-triggered repair
  for saved drafts.
- `PR-Landing-Page-Public-Prerender` remains later, after input quality and
  validator consistency are tightened.

## Verification

- Passed after review NIT fix: `pytest tests/test_extracted_campaign_skill_registry.py tests/test_extracted_landing_page_generation.py -q` - 43 passed.
- Passed after review NIT fix: `git diff --check`.
- Passed after review NIT fix: local PR review wrapper.

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Plan | ~80 |
| Prompt | ~35 |
| Tests | ~35 |
| **Total** | **~150** |
