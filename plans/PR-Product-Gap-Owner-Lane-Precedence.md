# PR-Product-Gap-Owner-Lane-Precedence

## Why this slice exists

Issue #1925 found that owner-lane inference already uses customer/topic text
before routing metadata, but that precedence was not pinned as an explicit
contract. Product-gap trust depends on knowing whether routing metadata can
override misleading text, especially for conflicts such as an invoice question
with auth/product routing labels.

This slice documents and tests the current text-first contract so future
hardening can change it deliberately rather than accidentally.

## Scope (this PR)

Ownership lane: deflection/product-gap-hardening
Slice phase: Production hardening

1. Add an explicit owner-lane precedence note at the inference point.
2. Add regression coverage for conflicting customer text and routing metadata:
   customer/topic wording wins, while routing labels remain visible on the
   action item.
3. Add fallback coverage for neutral customer text: routing labels still infer
   an owner lane when text has no deterministic lane.

### Review Contract

Acceptance criteria:
- Invoice/billing customer text with auth routing labels resolves to `Billing`.
- Auth routing labels remain present in `routing_signals`; they support the
  handoff but do not override customer/topic text.
- Neutral customer text with auth routing labels still resolves to
  `Auth / Product UX`, proving routing metadata remains a fallback.

Affected surfaces:
- Product-gap report owner-lane inference.
- Focused deflection report regression tests.

Risk areas:
- Accidentally making routing metadata override customer text.
- Accidentally disabling routing metadata as a useful fallback.

Reviewer rules triggered: R1, R10.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Product-Gap-Owner-Lane-Precedence.md`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

The owner-lane function already searches deterministic tokens in this order:
topic, question, customer wording, then routing-signal text, then topic-title
fallback. This slice leaves that behavior intact and adds a short contract note
beside the function so the precedence is not hidden in implementation order.

The regression tests build report models with repeated product-gap items and
explicit `routing_signals`. One test creates the #1925 conflict: invoice text
with auth routing labels. The second creates neutral text with auth routing
labels to prove routing metadata is still used when the text layer has no
deterministic match.

## Intentional

- This slice does not change owner-lane behavior; it locks the current
  text-first contract.
- This does not add a conflict-confidence field or review label. That would be
  a product-shape change and should be a separate slice if desired.
- Routing metadata remains visible in paid action rows even when it does not
  control `owner_lane`.

## Deferred

- Optional #1925 follow-up: add an explicit conflict/review signal if we want
  the report to call out text-vs-routing disagreements instead of only exposing
  both fields.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_deflection_report.py::test_owner_lane_precedence_customer_text_beats_conflicting_routing_signals tests/test_content_ops_deflection_report.py::test_owner_lane_uses_routing_signals_when_customer_text_is_neutral -q` -
  passed, 2 tests.
- `python -m pytest tests/test_content_ops_deflection_report.py::test_csv_product_gap_owner_lane_vertical_routes_login_gap tests/test_content_ops_deflection_report.py::test_support_platform_provenance_does_not_route_owner_lane tests/test_content_ops_deflection_report.py::test_owner_lane_precedence_customer_text_beats_conflicting_routing_signals tests/test_content_ops_deflection_report.py::test_owner_lane_uses_routing_signals_when_customer_text_is_neutral tests/test_content_ops_deflection_report.py::test_owner_lane_keyword_matching_uses_tokens_not_substrings -q` -
  passed, 5 tests.
- Local PR review gate with the current PR body file wired in - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 2 |
| `plans/PR-Product-Gap-Owner-Lane-Precedence.md` | 96 |
| `tests/test_content_ops_deflection_report.py` | 87 |
| **Total** | **185** |
