# PR-FAQ-Report-Contract-Example

## Why this slice exists

The FAQ generator and search/detail route now produce a stable full FAQ report
shape, and a parallel landing-page session needs that shape for truthful demo
wiring. Right now the contract is discoverable only by reading Python
dataclasses, builder output, and tests. This slice adds a checked-in handoff doc
and current JSON example generated from the canonical builder.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Product polish.

1. Document the generated FAQ report result shape used by execute, persistence,
   detail hydration, and landing-page rendering.
2. Add a compact generated JSON example from the current deterministic FAQ
   builder.
3. Add a focused fixture test that keeps the checked-in example parseable and
   aligned to the documented core keys.
4. Keep the example scoped to frontend/demo consumption; no runtime behavior or
   API schema changes.

### Files touched

- `plans/PR-FAQ-Report-Contract-Example.md`
- `docs/frontend/content_ops_faq_report_contract.md`
- `docs/frontend/content_ops_faq_report_example.json`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

The doc names the canonical producer (`TicketFAQMarkdownResult.as_dict()`), the
persisted detail wrapper (`TicketFAQDraft.as_dict()`), and the compact search
projection envelope. The JSON example is generated from representative
support-ticket and search-log rows using `build_ticket_faq_markdown(...)`, then
trimmed to the fields a frontend/demo needs to render proof cards and the full
report preview. The focused test parses the JSON and asserts the documented core
item fields, evidence status values, output checks, and doc-to-example link.

## Intentional

- No generated FAQ code changes; this is a handoff artifact for downstream UI
  work.
- The search route remains compact. Full report rendering should use the detail
  route or execute result, not the search result row alone.
- The example uses synthetic ticket/search IDs and an example support URL; it is
  not customer data.

## Deferred

- Parked hardening: none.
- A formal OpenAPI/schema export remains deferred until the hosted API schema is
  generated from the FastAPI app.

## Verification

- JSON validation for `docs/frontend/content_ops_faq_report_example.json` -
  passed.
- Focused pytest for the contract doc fixture and CI enrollment audit - 11
  passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` - passed with
  119 matching tests enrolled.
- Plan/code consistency audit for this plan - passed.
- `git diff --check` - passed.
- Local PR review bundle - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 79 |
| Contract doc | 144 |
| JSON example | 122 |
| CI runner enrollment | 1 |
| Fixture test | 52 |
| **Total** | **398** |
