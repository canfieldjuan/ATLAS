# PR-Content-Ops-FAQ-SaaS-Demo-Corpus

## Why this slice exists

The FAQ search smoke now uses SaaS-shaped query terms, but the repo still only
has a tiny checked-in support-ticket sample. For an on-domain FAQ demo, we need a
defensible source artifact that is clearly synthetic, B2B SaaS-shaped, and
guarded against CFPB or consumer-finance leakage.

This slice creates that starter corpus without claiming it is real customer data
or an anonymized design-partner run.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Functional validation

1. Add a labeled synthetic B2B SaaS support-ticket CSV under examples.
2. Cover common SaaS support intents: reporting exports, dashboard freshness,
   SSO, permissions/seats, API/webhooks, integrations, imports, automation, and
   billing.
3. Add tests that reject CFPB/consumer-finance terms in the demo corpus.
4. Add a generator-level test proving the corpus normalizes without warnings and
   produces a FAQ artifact with all output checks passing.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Corpus.md` | Plan contract for this validation slice. |
| `extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv` | Synthetic labeled B2B SaaS support-ticket corpus. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Domain-clean and generator-output tests for the corpus. |
| `scripts/run_extracted_pipeline_checks.sh` | Enroll the new Content Ops test in extracted CI. |

## Mechanism

The corpus uses provider-style support-ticket columns already accepted by the
source adapter (`Ticket ID`, `Account Name`, `Vendor Name`, `Subject`,
`Description`, `Pain Category`, and `Source Type`). Every row carries
`Dataset Label=synthetic_b2b_saas_demo` so downstream demo code can present the
provenance honestly.

The test reads the CSV, asserts the corpus is large enough for search/demo
coverage, rejects consumer-finance leakage terms, normalizes rows through
`source_rows_to_campaign_opportunities`, then runs `build_ticket_faq_markdown`
with fail-closed output-check assertions.

## Intentional

- This is synthetic sample data, not an anonymized design-partner run.
- No public landing-page or frontend demo code changes land here.
- No generator, search, API, repository, or schema behavior changes.

## Deferred

- A future demo-seeding slice can load this corpus into the hosted FAQ search
  route for an on-domain searchable demo.
- A future design-partner slice can replace or supplement this corpus with an
  anonymized real SaaS export when one exists.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q - 2 passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . - passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Corpus.md - passed.
- git diff --check - passed.
- Local PR review bundle - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 78 |
| CSV corpus | 37 |
| Tests | 103 |
| CI enrollment | 1 |
| **Total** | **~219** |
