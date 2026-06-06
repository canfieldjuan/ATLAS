# Content Ops FAQ Financial Action Language

## Why this slice exists

The live CFPB source-row smoke proved the real-public data path works, but it
also exposed a generic FAQ quality issue: financial complaints containing words
like "account" can be routed to login/profile action steps. The fix belongs in
the FAQ action classifier, not in the CFPB exporter, so all ticket-like sources
benefit.

## Scope (this PR)

1. Add a financial/billing action branch before the generic account/login
   branch.
2. Add escalation guidance for fees, payments, statements, loans, disputes, and
   account charges.
3. Lock the behavior with CFPB-shaped source-row tests while keeping the
   renderer source-agnostic.
4. Remove the merged CFPB live-fetch coordination row and claim this slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Financial-Action-Language.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Replace stale CFPB row with this FAQ-quality slice claim. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Add financial action and escalation rules. |
| `tests/test_extracted_ticket_faq_markdown.py` | Regression coverage for CFPB-shaped financial rows. |

## Mechanism

The action classifier remains deterministic and source-agnostic. It checks the
topic plus evidence text against ordered keyword groups. This PR adds a
financial group before the login/account group so terms like fees, billing,
payments, statements, loans, disputes, and charges produce financial next steps
instead of profile-setting steps.

The test fixture uses CFPB-shaped source rows because that is what exposed the
bug, but the assertions only depend on generic source-row fields:
`source_type`, `source_title`, `pain_points`, and evidence text.

## Intentional

- No CFPB-specific branch in the renderer.
- No LLM generation. FAQ output stays deterministic and easy to audit.
- No platform-specific legal or financial advice; the output tells users to
  gather records, compare charges, contact support, and keep evidence.

## Deferred

- More specialized action templates for real customer product docs remain
  deferred until a host supplies those docs or a real ticket export.

## Verification

- Focused pytest for FAQ Markdown: 44 passed.
- Python compile for FAQ renderer and tests: passed.
- Packaged support-ticket FAQ CLI with required output checks: passed.
- Local PR review: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Financial-Action-Language.md` | 65 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 35 |
| `tests/test_extracted_ticket_faq_markdown.py` | 55 |
| **Total** | **159** |
