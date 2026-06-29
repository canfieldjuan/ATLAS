# PR-Product-Gap-Platform-Provenance

## Why this slice exists

#1925 identified a trust-hardening gap after the CSV-first product-gap work
landed: `support_platform` is kept in hosted submit/report metadata, but it is
not preserved on normalized support-ticket source rows. The root cause is that
support-ticket normalization has routing aliases and company/contact aliases,
but no provenance alias for platform. This fixes the root for platform
provenance without changing owner-lane routing semantics.

## Scope (this PR)

Ownership lane: deflection/product-gap-hardening
Slice phase: Production hardening
Max files: 6

1. Preserve safe support-platform aliases on normalized support-ticket rows as
   provenance metadata.
2. Preserve hosted-submit platform defaults when sparse CSV platform columns
   contain blank cells, while still letting non-empty row platform aliases win.
3. Keep platform metadata out of routing signals and owner-lane inference.
4. Add focused submit/default, normalization, and report-model regression
   coverage.

### Review Contract

Acceptance criteria:
- Support-ticket rows preserve `support_platform`, `Support Platform`, and
  `platform` aliases as `support_platform`.
- Existing hosted submit defaults that add `support_platform` continue to flow
  into normalized source material, including when a sparse CSV
  `support_platform` column contains blank cells.
- Non-empty row aliases such as `Support Platform` and `platform` are not
  shadowed by the hosted-submit default.
- `support_platform` does not appear inside product-gap `routing_signals`.
- Adding `support_platform` does not change owner-lane routing.

Affected surfaces:
- Support-ticket input package normalization.
- Hosted deflection submit row defaults.
- Focused support-ticket/report regression tests.

Risk areas:
- Accidentally treating provider provenance as a routing/owner signal.
- Disturbing existing submit metadata or private-text exclusions.

Reviewer rules triggered: R1, R2, R4, R8, R10, R14.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Product-Gap-Platform-Provenance.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

The normalizer gets a dedicated support-platform alias tuple and copies the
first non-empty platform value into `support_platform` during the same
source-row shaping pass that already handles company/vendor/contact metadata.
Hosted submit row defaults preserve the first non-empty row platform alias when
present, and otherwise restore the selected form platform if a sparse CSV
`support_platform` cell is blank. Platform is deliberately not added to
`_ROUTING_CONTEXT_KEYS`, so grouped FAQ items and product-gap action rows do
not treat the provider as a routing signal.

## Intentional

- No routing-signal projection for `support_platform`. Platform is provenance,
  not a product owner lane.
- No owner-lane precedence change. The existing text-first routing contract is
  handled by a later hardening slice.
- No `Company`/`organization` alias cleanup in this PR; that is the next
  separate hardening slice.

## Deferred

- #1925 follow-up: stop plain `Company` from aliasing into routing
  `organization`.
- #1925 follow-up: document and test owner-lane text-vs-routing precedence.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_content_deflection_submit.py::test_deflection_submit_defaults_fill_blank_support_platform_cells tests/test_extracted_support_ticket_input_package.py::test_support_ticket_input_package_preserves_support_platform_provenance tests/test_content_ops_deflection_report.py::test_support_platform_provenance_does_not_route_owner_lane -q` - passed, 3 tests.
- Local PR review gate with the current PR body file wired in - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 43 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 8 |
| `plans/PR-Product-Gap-Platform-Provenance.md` | 102 |
| `tests/test_content_ops_deflection_report.py` | 30 |
| `tests/test_extracted_content_deflection_submit.py` | 37 |
| `tests/test_extracted_support_ticket_input_package.py` | 29 |
| **Total** | **249** |
