# PR-EOM-MCP-Payment-Projection

## Why this slice exists

The first attempt to add optional EOM check metadata (closed PR #2368) exposed
a deployment-order defect: the payment record is selected with `cp.*`, while
the long-lived invoicing MCP serializes its service result as its established
response contract. Read-only runtime inspection confirms that
`atlas-invoicing-mcp.service` is a separate long-lived process from the full
receivables provider (`atlas-api.service`). Applying the new database columns
before every MCP process runs a projection would therefore change MCP responses
during a mixed-version rollout.

This is the required first stage recorded in Billing & Payments coordination
#2362. It deliberately ships no schema, route, or payment-model change. Once
deployed and the MCP service has been restarted and verified, the next provider
slice can safely add the durable check metadata with readiness and real-entrypoint
coverage.

The local Unit Gate prerequisite #2369 merged as `12808da63` before this branch
was rebased. Its actual selector now escalates this staging diff to the full
suite because four reached OAuth/MCP tests are recorded as bare file-level
collection failures; the entries remain untouched because only full-suite
collection can prove them resolved.

### Problem-derived contract

- Root cause: `_record_customer_payment_with_service` serializes the entire
  canonical payment view. Because that view uses `cp.*`, a future additive
  `customer_payments` column reaches legacy MCP clients unless the MCP boundary
  explicitly owns its response projection before the schema changes.
- Correct fix must touch/change: create the projection at the MCP response
  boundary and add a direct helper regression that supplies the future EOM-only
  fields from a fake canonical service result, then proves the published JSON
  excludes exactly those fields while preserving all other payment data.
- Must not change: the database schema, migrations, payment validation,
  idempotency, allocation, deposit/clearing/return/void lifecycle, MCP input
  schema, persisted financial rows, Gmail, full/slim EOM routes, tracker, and
  Website behavior, or the local unit-gate baseline.

## Scope (this PR)

Ownership lane: eom/billing-payments
Slice phase: Vertical slice
Max files: 3

1. Project the future EOM-only `check_date` and `received_through` keys from
   the successful `record_customer_payment` MCP response without changing the
   canonical payment view or its stored financial record.
2. Prove the projection against the real service-helper serialization seam with
   a fake service result containing both fields and ordinary payment/allocations
   data.
3. Publish the narrow prerequisite for deployment before the separate metadata
   migration slice; the PR body and #2362 state the mandatory restart/verify
   handoff.

### Review Contract

- Acceptance criteria:
  - [ ] A successful `record_customer_payment` helper response excludes
    `check_date` and `received_through` when a canonical service result includes
    them; settled by
    `tests/test_receivables.py::test_mcp_payment_response_projects_future_eom_check_metadata`.
  - [ ] The same response retains ordinary canonical payment fields and nested
    allocations unchanged; settled by the same regression's exact JSON
    assertions.
  - [ ] The projection occurs after the canonical service completes, so it
    cannot change creation, idempotency, or allocation inputs; settled by the
    call-argument assertions in the same regression and
    `atlas_brain/mcp/invoicing_server.py::_record_customer_payment_with_service`.
  - [ ] No migration, EOM route, payment model, or receivables service code is
    changed in this prerequisite; settled by the three-file diff enumeration.
- Reachability proof: the test invokes the same private service helper reached
  by the registered `record_customer_payment` MCP tool, verifies its JSON
  output, and uses a fake service only to supply a future row shape that cannot
  exist before the later migration.
- Affected surfaces: the stable invoicing MCP payment response boundary and
  its regression coverage.
- Risk areas: backward-compatible MCP output, mixed-version provider/MCP
  deployment, and accidental change to a financial write path.
- Reviewer rules triggered: R1, R2, R5, R8, R14.

### Boundary-change enumeration

N/A - no admission, identity, routing, or validation boundary changes. The
existing MCP output is projected after a successful canonical service result.

### Deployed-config probing

N/A - no environment/config fallback changes. The deployment handoff is an
operational prerequisite, not a code-level configuration decision.

### Files touched

- `atlas_brain/mcp/invoicing_server.py`
- `plans/PR-EOM-MCP-Payment-Projection.md`
- `tests/test_receivables.py`

## Mechanism

After `ReceivablesService.create_payment` returns, the MCP helper builds a
shallow response dictionary that omits only `check_date` and
`received_through`; it then serializes the projected dictionary with the
existing JSON encoder. The canonical service still receives exactly the same
arguments and persists exactly the same financial record. The direct regression
uses valid helper inputs and a service fake that returns a representative
payment/allocations view plus both future fields, proving this older MCP binary
will remain response-compatible after a later schema expansion.

## Intentional

- The projection lists only the two approved future EOM check-metadata fields;
  it is not a broad or speculative filtering layer that could hide legitimate
  existing MCP payment fields.
- This slice contains no migration. It must be deployed and the standalone
  invoicing MCP service restarted/verified before the metadata schema PR is
  allowed to deploy.
- The fake service is intentional: the prerequisite must prove handling of a
  row shape that does not yet exist in production, without creating a test or
  production financial record.

## Deferred

- After verified MCP-projection deployment, add migration 368, optional EOM
  request fields, legacy fingerprint compatibility, route readiness gating, and
  authenticated full/slim ASGI proof in the next provider slice (#2362).
- Receipt delivery, customer ledger/history, billing-run previews, Gmail-draft
  recovery, sent-mail reconciliation, and Square queue remain planned Billing &
  Payments slices (#2362); nonessential hardening remains #2363.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_receivables.py -q -k "mcp_payment_response_projects_future_eom_check_metadata"` — 1 passed, 73 deselected.
- `python -m pytest tests/test_receivables.py -q` — 66 passed, 8 skipped;
  skipped isolated-PostgreSQL cases require an explicit local test URL and are
  unrelated to this non-schema projection.
- `python -m py_compile atlas_brain/mcp/invoicing_server.py tests/test_receivables.py` and targeted `ruff check ... --ignore F841` — passed; the MCP module's ignored F841 bindings pre-date this diff.
- `python scripts/select_impacted_tests.py --base origin/main` — `FULL`, naming
  exactly the four pre-existing OAuth/MCP file-level collection baselines after
  merged #2369. The managed full local gate is the final acceptance evidence.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/mcp/invoicing_server.py` | 9 |
| `plans/PR-EOM-MCP-Payment-Projection.md` | 151 |
| `tests/test_receivables.py` | 66 |
| **Total** | **226** |
