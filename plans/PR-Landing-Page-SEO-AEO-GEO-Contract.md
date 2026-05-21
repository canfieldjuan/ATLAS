# PR-Landing-Page-SEO-AEO-GEO-Contract

## Why this slice exists

AI Content Ops blog generation now has a defined SEO/AEO/GEO contract and
runtime readiness work. Landing-page generation does not yet have the same
source-of-truth definition, so adding checks directly to code would risk mixing
blog rules into a different asset type.

Landing pages need a separate contract because the buyer job is different.
Blog GEO is about evidence-rich article content that can be cited or
summarized. Landing-page GEO is about making the offer, audience, problem,
trust cues, objections, and conversion path clear enough for people, search
engines, and answer engines to understand.

## Scope (this PR)

1. Define the landing-page SEO/AEO/GEO readiness contract.
2. Separate draft-level readiness from publish-level verification.
3. Document the implementation sequence for future code slices.
4. Add customer-facing language guardrails so Atlas does not overclaim what GEO
   can guarantee.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-SEO-AEO-GEO-Contract.md` | Plan doc for this docs-only slice. |
| `docs/audits/ai_content_ops_landing_page_seo_aeo_geo_contract_2026-05-21.md` | New landing-page SEO/AEO/GEO contract and roadmap. |

## Mechanism

This change does not alter runtime code. It establishes the contract future
implementation slices should follow.

The audit defines:

- SEO checks for metadata, slug quality, and metadata consistency.
- AEO checks for answer-first hero copy, problem/solution clarity, audience
  specificity, and objection coverage.
- GEO checks for offer clarity, audience clarity, extractable answers,
  section semantics, trust-signal visibility, conversion-path clarity, and
  claim safety.
- Draft-ready JSON shapes for `seo_aeo_readiness` and `geo_readiness`.
- Publish-ready checks that belong at the public rendered page boundary, not
  inside the generator alone.

## Intentional

- No generator changes.
- No quality-gate changes.
- No generated-asset API or export changes.
- No frontend review UI changes.
- No claim that GEO guarantees placement in AI answer engines.

## Deferred

- Add landing-page readiness helpers to generated-asset rows and CSV export.
- Extend the landing-page quality gate with blocking checks where appropriate.
- Update the landing-page generation prompt so the LLM produces the structures
  the validators expect.
- Surface landing-page readiness in the review UI.
- Add publish-level crawler-visible HTML, metadata, structured-data, and CTA
  verification once generated landing pages have a public renderer.

## Verification

- `git diff --check` -> passed with 0 whitespace errors.
- `bash scripts/local_pr_review.sh origin/main` -> passed 3/3 top-level
  checks: pre-push audit wrapper, plan/code consistency, and `git diff
  --check`.
- The pre-push audit wrapper inside local review reported all 8 internal checks
  passed: MCP tool counts, MCP port assignments, MCP tool-name inventories,
  extracted manifest sync, plan shape, plan files touched, plan diff size, and
  ASCII Python policy.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Landing-page contract audit | ~310 |
| Total | ~395 |
