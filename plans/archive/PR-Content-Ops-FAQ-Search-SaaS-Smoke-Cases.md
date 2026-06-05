# PR-Content-Ops-FAQ-Search-SaaS-Smoke-Cases

## Why this slice exists

The hosted FAQ search smoke currently seeds a generic hit query (`password
reset`) and a consumer-finance miss query (`escrow shortage`). That is useful
for route mechanics, but it is a poor default for the support-ticket FAQ product
lane: a future operator or demo runner can see non-SaaS search language in the
seeded cases and mistake it for product-facing demo data.

This slice keeps the smoke deterministic while making the seeded hit and miss
queries B2B SaaS-shaped. It does not build the public on-domain demo corpus;
that larger sample-source decision remains parked until we are ready.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Functional validation

1. Replace the seeded FAQ search hit query with a B2B SaaS reporting/export
   support-ticket query.
2. Replace the seeded miss query with an unrelated but still SaaS-shaped admin
   query.
3. Update seeded FAQ document topic, question, answer summary, and search text
   to match the reporting/export hit.
4. Update focused smoke tests to assert the new SaaS-shaped cases.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-SaaS-Smoke-Cases.md` | Plan contract for this validation slice. |
| `scripts/smoke_content_ops_faq_search_concurrency.py` | Seed SaaS-shaped hit/miss search cases and matching FAQ document text. |
| `tests/test_smoke_content_ops_faq_search_concurrency.py` | Update direct seeded-search smoke assertions. |
| `tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py` | Update seeded route/detail case assertions. |

## Mechanism

The seed smoke still writes one approved FAQ draft per corpus and projects
`TicketFAQSearchDocument` rows through the existing repository. Only the fixed
seed text changes:

- Hit query: `export attribution report`
- Miss query: `saml domain verification`

The miss remains a SaaS admin query but is absent from the projected
`search_text`, so the existing require-results and allow-empty route cases still
exercise both branches.

## Intentional

- No search scoring, API, repository, schema, or route behavior changes.
- No public demo corpus is added in this slice.
- CFPB scale and durability validations remain documented separately; this
  slice only changes the small hosted search smoke seed vocabulary.

## Deferred

- A future demo-corpus slice should create a larger labeled synthetic B2B SaaS
  support-ticket corpus and guard it against CFPB/consumer-finance leakage.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_seeded_route_e2e.py -q - 63 passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Search-SaaS-Smoke-Cases.md - passed.
- git diff --check - passed.
- Local PR review bundle - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 77 |
| Smoke seed text | 14 |
| Tests | 59 |
| **Total** | **~150** |
