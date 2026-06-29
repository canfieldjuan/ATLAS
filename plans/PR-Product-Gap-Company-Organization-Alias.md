# PR-Product-Gap-Company-Organization-Alias

## Why this slice exists

Issue #1925 found one remaining product-gap routing edge after PR #1926:
plain CSV `Company` currently normalizes into both `company_name` and routing
`organization`. That can make customer/account identity look like product-gap
routing evidence, which weakens the trust story for owner-lane handoff data.

This slice keeps company/account data as customer identity while requiring
explicit organization-style headers for routing `organization`.

## Scope (this PR)

Ownership lane: deflection/product-gap-hardening
Slice phase: Production hardening

1. Remove plain `company` from routing `organization` aliases in the support
   ticket normalizer.
2. Keep `company`, `account`, `account_name`, and `company_name` as
   `company_name` aliases.
3. Add focused coverage for a plain `Company` row and an explicit
   `Organization` row so the boundary is not inferred from prose.

### Review Contract

Acceptance criteria:
- A row with only `Company` preserves `company_name` and does not emit routing
  `organization`.
- Explicit organization aliases still emit routing `organization`.
- Existing product-gap report output that uses explicit `Organization` keeps
  its routing signal.

Affected surfaces:
- Support-ticket input package normalization.
- Focused support-ticket/report regression tests.

Risk areas:
- Losing legitimate organization routing metadata.
- Continuing to leak account identity into product-gap routing evidence.

Reviewer rules triggered: R1, R10.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Product-Gap-Company-Organization-Alias.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

The support-ticket normalizer already has separate alias lists for routing
context and company identity. This slice narrows only
`_ROUTING_CONTEXT_KEYS["organization"]` by dropping `company` from that alias
tuple and keeping explicit organization aliases (`organization`,
`organisation`, `org`, and `requester_organization`).

The existing `_COMPANY_KEYS` mapping remains unchanged, so plain company data
continues to normalize into `company_name` for customer/account identity.

## Intentional

- This does not remove `organization` as a routing signal; explicit
  organization headers still flow through.
- This does not change owner-lane precedence. Text-vs-routing conflict behavior
  remains the next #1925 slice.
- This does not add provider-specific organization aliases beyond
  `requester_organization`; broader alias expansion should be evidence-driven.

## Deferred

- #1925: document and test owner-lane text-vs-routing precedence.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_support_ticket_input_package.py::test_support_ticket_bundle_inherits_parent_fields_and_comment_text tests/test_extracted_support_ticket_input_package.py::test_support_ticket_input_package_keeps_company_out_of_routing_organization tests/test_extracted_support_ticket_input_package.py::test_support_ticket_input_package_preserves_explicit_routing_organization -q` -
  passed, 6 tests.
- `python -m pytest tests/test_content_ops_deflection_report.py::test_csv_product_gap_owner_lane_vertical_routes_login_gap -q` -
  passed, 1 test.
- Grep for organization/company alias tables in
  `extracted_content_pipeline/support_ticket_input_package.py` -
  confirmed `company` remains only in `_COMPANY_KEYS`, while routing
  `organization` uses explicit organization aliases.
- Local PR review gate with the current PR body file wired in - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 2 |
| `plans/PR-Product-Gap-Company-Organization-Alias.md` | 95 |
| `tests/test_extracted_support_ticket_input_package.py` | 43 |
| **Total** | **140** |
