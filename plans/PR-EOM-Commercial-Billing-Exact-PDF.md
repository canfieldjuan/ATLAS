# PR-EOM-Commercial-Billing-Exact-PDF

## Why this slice exists

Coordinating issue #2362 and deferred item #2363 H-01 require financial
precision before an approved commercial-billing writer can produce PDF
attachments. The existing invoice renderer formats and calculates money via
`float`, so the future exact-cent writer cannot safely reuse it.

### Problem-derived contract

- Root cause: `invoice_pdf.py` converts invoice money and line totals through
  binary float before displaying a customer-facing amount.
- Correct fix must touch/change: replace renderer-only money normalization and
  arithmetic with finite, cent-quantized `Decimal` values and test exact-cent
  inputs plus the existing PDF contract.
- Must not change: invoice persistence, existing invoice schema/API/MCP flows,
  PDF layout/copy, customer records, payment flows, Gmail behavior, and invoice
  status. No invoice or email is created by this PR.

## Scope (this PR)

Ownership lane: eom/billing-approved-gmail-drafts
Slice phase: Production hardening

1. Format invoice totals, line items, discounts, and tax using Decimal-safe
   monetary conversion instead of float conversion.
2. Add focused regression tests for fractional-cent rounding, cents/Decimal/
   string inputs, malformed values, and valid branded PDF bytes.

### Review Contract

- Acceptance criteria: `tests/test_invoice_pdf.py` proves all rendered monetary
  text is cent-quantized with Decimal, string, integer-cent, malformed, and
  negative/discount inputs; it proves `render_invoice_pdf` still emits a valid
  PDF. A source-level assertion proves no `float(` remains in the renderer.
- Reachability proof: existing invoice PDF callers (`export_invoice_pdf`,
  monthly invoice generation, and the future approval writer) call
  `render_invoice_pdf`; a fixture invocation produces a `%PDF-` artifact with
  expected displayed cents.
- Affected surfaces: `atlas_brain/services/invoice_pdf.py`, focused PDF tests,
  and invoicing workflow test enrollment.
- Risk areas: rounding mode, malformed legacy data, discount/tax branches,
  existing layout, and unintended sender/financial side effects.
- Reviewer rules triggered: R3, R8, R10, R13.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: PDF-only money normalizer and line-total calculation.
- Replaced-path behaviors: finite Decimal/string/integer inputs format exactly;
  invalid/non-finite values retain the existing safe `$0.00` fallback.
- Guard-relevant fields: monetary fields, quantity, flat fee, discount, and
  tax amount.
- Caller x input shape: legacy JSON values may be numbers or strings; new
  exact-cent documents will provide Decimal-compatible values. Tests admit both
  and reject `NaN`/infinity/non-numeric values without emitting unsafe amounts.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no config, credential, or route change.
- Explicit value probe: N/A - no config boundary change.
- Absent value probe: N/A - no config boundary change.
- Default-session/default-context probe: N/A - no config boundary change.
- Side-effect ordering: pure in-memory rendering only; no database, Gmail, or
  email side effect is reachable from this change.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/services/invoice_pdf.py`
- `plans/PR-EOM-Commercial-Billing-Exact-PDF.md`
- `tests/test_invoice_pdf.py`

## Mechanism

One private helper parses a finite `Decimal(str(value))`, quantizes it to cents
with `ROUND_HALF_UP`, and returns a stable display value. Existing table code
uses it for money fields and calculates totals with Decimal multiplication/
subtraction. The renderer’s page geometry and business copy are unchanged.

## Intentional

- This intentionally contains H-01 at the PDF boundary instead of rewriting
  legacy invoice persistence; the new approval writer will use its own
  exact-cent database adapter in the next slice.
- Invalid legacy values retain `$0.00` display fallback rather than causing a
  customer-facing PDF failure.

## Deferred

- Exact-cent invoice persistence plus durable selected-candidate approval:
  #2362 next provider slice.
- No-send Gmail draft creation/recovery and verified sent-mail reconciliation:
  #2362 subsequent slices.
- Legacy repository float removal outside this boundary remains #2363 H-01.

Parking predicate: product-shape changes and non-PDF legacy refactors are
parked. Parked hardening: none.

## Verification

- Pending before push: focused pytest, invoicing regression, ruff, compileall,
  plan sync/check, local diff budget, financial source scan, and local PR
  review through `scripts/push_pr.sh` exactly once.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 5 |
| `atlas_brain/services/invoice_pdf.py` | 60 |
| `plans/PR-EOM-Commercial-Billing-Exact-PDF.md` | 121 |
| `tests/test_invoice_pdf.py` | 92 |
| **Total** | **278** |
