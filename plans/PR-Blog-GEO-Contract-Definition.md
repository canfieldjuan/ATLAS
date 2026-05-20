# PR-Blog-GEO-Contract-Definition

## Why this slice exists

The blog SEO/AEO work now preserves SEO fields, surfaces readiness output, and
blocks incomplete SEO/AEO drafts before save. GEO is still undefined, which
means we cannot safely claim "GEO-ready" or build a validator without making up
the rules inside code.

This slice defines GEO as a product and engineering contract first.

## Scope (this PR)

1. Add a first-class GEO contract doc for AI Content Ops blog posts.
2. Separate draft readiness from publish readiness.
3. Define the minimum checks required before Atlas can claim GEO readiness.
4. Cross-link the contract from the discovery audit that identified the gap.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-GEO-Contract-Definition.md` | Plan doc for this slice. |
| `docs/audits/ai_content_ops_blog_geo_contract_2026-05-20.md` | First-class GEO definition. |
| `docs/audits/ai_content_ops_blog_seo_aeo_geo_discovery_2026-05-20.md` | Link the new contract and update the status of Gap 1. |

## Mechanism

The contract doc defines:

- What GEO means for Atlas.
- What GEO does not promise.
- The difference between GEO draft readiness and GEO publish readiness.
- The exact checklist a validator should implement.
- The customer-facing language that is safe before and after implementation.

## Intentional

- No code changes in this slice. A validator needs a stable written contract
  before it becomes a gate.
- No claim that GEO guarantees ChatGPT, Perplexity, or AI Overview placement.
- No replacement of SEO or AEO. GEO builds on them, but it adds evidence,
  citation, entity, and publish-surface requirements.

## Deferred

- Implement `geo_readiness` output on generated blog asset rows.
- Add a save-time GEO gate after deciding which checks belong in generation
  versus public rendering.
- Add publish-surface verification for canonical, JSON-LD, FAQ schema, and
  crawler-visible article HTML.

## Verification

- Diff whitespace check -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| GEO contract doc | ~195 |
| Discovery audit update | ~30 |
| **Total** | **~290** |
