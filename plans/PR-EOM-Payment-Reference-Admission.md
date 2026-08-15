# PR-EOM-Payment-Reference-Admission

## Why this slice exists

The Billing & Payments functional contract requires a check number for checks
and a confirmation or transaction ID for ACH and Square. H-07 in the
[Billing & Payments hardening ledger #2363](https://github.com/canfieldjuan/ATLAS/issues/2363)
confirmed that both active EOM payment HTTP models still admit a missing
`reference`, even though the deployed tracker and Website already reject it.
This closes that provider-side admission gap before any financial service,
receipt outbox, or allocation writer runs.

**BREAKING (EOM private API only):** a direct EOM receivables caller that omits
or sends a blank `reference` will now receive HTTP 422. The current tracker and
Website already send a trimmed nonblank value for every admitted method. The
legacy invoicing MCP payment contract remains unchanged.

### Problem-derived contract

- Root cause: the EOM-specific HTTP request boundary accepts an optional
  receipt identifier, so a direct authenticated request can persist a check,
  ACH, or Square payment with no required reference despite the operator
  contract and current consumer validation. The general receivables service is
  not the root cause: it also supports the separately preserved legacy MCP
  payment surface.
- Correct fix must touch/change: both parallel EOM request models must require
  a string reference and normalize/reject blank input before their route
  functions can call `ReceivablesService`; route-level ASGI tests must prove
  the full and slim entrypoints return 422 without invoking their service;
  model tests must cover all three admitted payment methods and the MCP
  compatibility test must prove its optional reference remains optional.
- Must not change: the `ReceivablesService` method signature and financial
  transaction/idempotency behavior; customer, allocation, receipt-outbox,
  deposit, clearing, Gmail, PDF, and Square lifecycle writers; legacy MCP
  request/response behavior; tracker and Website contracts; migrations and
  customer-facing copy.

## Scope (this PR)

Ownership lane: eom/billing-payments
Slice phase: Functional validation

Max files: 4

1. Require and trim one nonblank payment reference at the active full and slim
   EOM receivables HTTP request models for each existing `check`, `ach`, and
   `square` payment method.
2. Reject absent, null, whitespace-only, non-string, and over-length references
   with FastAPI/Pydantic validation before the route calls a financial service.
3. Preserve valid current EOM requests and the separate legacy MCP optional
   reference contract; add focused regression/reachability evidence.

### Review Contract

- Acceptance criteria:
  - [ ] The active full and slim `POST /api/v1/receivables/payments` entrypoints
    return 422 for every admitted method when `reference` is absent, null, or
    blank, and their injected `create_payment` service records zero calls;
    settled by the new ASGI reachability test in `tests/test_receivables.py`.
  - [ ] Both `CreatePaymentRequest` models accept a nonblank check number, ACH
    confirmation, or Square transaction ID, trim surrounding whitespace, and
    reject non-string or 257-character values; settled by the parametrized
    model-contract test in `tests/test_receivables.py`.
  - [ ] Existing valid EOM route forwarding and unchanged-key payment retry
    behavior retain a nonblank reference; settled by the focused receivables
    route/retry regression tests.
  - [ ] The legacy `record_customer_payment` MCP tool still admits an omitted
    reference and forwards `None` to its service, settled by a focused MCP
    compatibility test in `tests/test_receivables.py`.
- Reachability proof: real full `main.app` and slim `main_eom.app` ASGI
  requests exercise the protected payment route with valid auth and assert the
  422 response plus zero service calls; no financial database or external
  delivery operation occurs.
- Affected surfaces: full and slim EOM FastAPI request models, their existing
  protected payment routes, EOM tracker/Website callers, and the legacy MCP
  compatibility seam.
- Risk areas: money-record admission, direct mixed-version caller behavior,
  validation ordering before mutation, duplicate full/slim model drift,
  idempotency preservation, and legacy MCP compatibility.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

This is a closed request-admission boundary. The relevant payment-method set is
**CLOSED**: `check`, `ach`, and `square` are the current `Literal` values in
each active model. Its membership is **ENUMERATED** by the existing full and
slim API models while H-06 tracks their later shared-schema extraction. An
out-of-set method receives Pydantic validation rejection; for every in-set
method, a non-string, absent, null, blank, or over-length reference is rejected
before the endpoint calls a service. The safe financial default is rejection:
the cost of an incomplete receipt identifier is a persistently ambiguous
financial record, while a caller can correct and resubmit before any write.

- Boundary path/seam: full
  `atlas_brain/api/invoicing/receivables.py::CreatePaymentRequest` and slim
  `atlas_brain/eom_api/receivables.py::CreatePaymentRequest`, both consumed by
  their existing `POST /api/v1/receivables/payments` routes.
- Replaced-path behaviors: optional/missing/blank `reference` previously bound
  to `None` or an empty string and could enter `ReceivablesService`; this PR
  replaces only that admission with a 422. Valid nonblank references continue
  to route unchanged except for whitespace trimming.
- Guard-relevant fields: `payment_method` selects the closed allowed method
  set; `reference` supplies the required receipt identifier. Customer,
  allocation, amount, dates, actor, and idempotency key retain their existing
  independent validation and are not part of this decision.
- Caller x input shape: deployed tracker requests a validated trimmed string
  for each method; the Website validates a nonblank form value before proxying;
  direct EOM callers with missing/null/blank/non-string/over-length references
  receive 422 and make zero writes; legacy invoicing MCP bypasses these EOM
  models and keeps its optional reference contract.

### Deployed-config probing

- Deployed/default config values: the active full ATLAS service currently has
  the receivables API enabled (read-only `/api/v1/receivables/ready` proof at
  the deployed `fde6e16f` revision). There is no reference-related environment
  default or config fallback, and this PR changes none.
- Explicit value probe: a valid trimmed reference reaches the existing model
  and route-forwarding behavior for each closed method, demonstrated by the
  focused model and route tests.
- Absent value probe: absent, null, empty, and whitespace-only references are
  ASGI-posted to both protected routes and return 422.
- Default-session/default-context probe: there is no reference default and no
  customer/actor fallback involved in this decision; the ASGI test uses a valid
  service token and actor only to prove rejection happens at body admission.
- Side-effect ordering: Pydantic body validation rejects the request before
  either route function can resolve canonical customer data or call
  `create_payment`; the injected service's zero call count is the observable
  proof.

### Files touched

- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/eom_api/receivables.py`
- `plans/PR-EOM-Payment-Reference-Admission.md`
- `tests/test_receivables.py`

## Mechanism

Each parallel EOM `CreatePaymentRequest` changes `reference` from optional to
required and uses the same small field validator: accept only a string, strip
it, then reject an empty normalized value. The existing closed payment-method
literal remains unchanged. Because FastAPI constructs the request model before
entering the endpoint function, an invalid body returns 422 before any canonical
customer lookup, service call, receipt-outbox attempt, payment write,
allocation, or idempotency lookup.

The general `ReceivablesService` stays permissive because legacy invoicing MCP
tools deliberately call it directly with an optional reference. Tests use the
real ASGI router with only auth/service dependencies faked, then separately
assert the existing MCP helper continues to forward `None`.

## Intentional

- This is a provider-only EOM API admission change. Current tracker and Website
  clients are already compliant, so they need no companion deployment; deploy
  ATLAS first and retain their existing behavior.
- The validation is deliberately not method-format-specific: ATLAS requires an
  identifying value but does not invent a regex for bank or Square identifiers
  it does not own.
- The two validators remain small mirrored code. Extracting a shared schema is
  existing H-06 work and would widen this financial admission slice.
- The legacy MCP `record_customer_payment` reference remains optional. Tightening
  the shared service would break that established surface and is not required to
  close the EOM route gap.
- No migration, financial correction, test payment, Gmail call, receipt email,
  or production-record mutation is used for verification.

## Deferred

- H-06 / #2363: consolidate the duplicate full and slim request schemas behind
  one shared contract after the financial admission behavior is stable.
- H-15 / #2363: explicit Gmail missing-draft replacement and scheduled
  observation remain separate delivery-state work.
- Stronger bank- or processor-specific reference formatting and duplicate-ID
  policy are not inferred here; they require an operator-owned business rule
  and are not necessary to require a receipt identifier.

Parking predicate: broader service/MCP policy changes, reference semantic
formatting, migrations, delivery behavior, and refactors are parked unless the
new EOM admission boundary cannot safely reject before financial mutation.

Parked hardening: none.

## Verification

- `git diff --check` — passed.
- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest -q tests/test_receivables.py -k 'payment_http_models or payment_routes_forward_omitted_allocations or full_and_slim_payment_entrypoints_reject_invalid_reference_before_service or legacy_mcp_customer_payment_keeps_reference_optional or multi_invoice_mcp_is_registered_with_bounded_strict_schema'` — 23 passed.
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=<local test database> /home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest -q tests/test_receivables.py` — 118 passed.
- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest -q tests/test_eom_payment_receipts.py` — 11 passed.
- Scoped Ruff lint for the two route modules and receivables test file — passed.
- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -m py_compile atlas_brain/api/invoicing/receivables.py atlas_brain/eom_api/receivables.py tests/test_receivables.py` — passed.
- Pending before push: repository `scripts/local_pr_review.sh`, the guarded full unit baseline, maturity matrix, and current-head AI reconciliation. Hosted Actions are diagnostic only per the operator's local-check direction; their known pre-step allocation failure remains H-11, not acceptance evidence.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/invoicing/receivables.py` | 18 |
| `atlas_brain/eom_api/receivables.py` | 18 |
| `plans/PR-EOM-Payment-Reference-Admission.md` | 203 |
| `tests/test_receivables.py` | 156 |
| **Total** | **395** |
