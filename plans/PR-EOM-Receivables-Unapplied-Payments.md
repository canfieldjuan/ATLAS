# PR-EOM-Receivables-Unapplied-Payments

Issue: #2362

## Why this slice exists

ATLAS must record an EOM canonical-customer payment without an invoice; provider models/service otherwise require an allocation.

### Problem-derived contract

- Root cause: initial creation and the shared normalizer required at least one allocation.
- Correct fix: keep the default guard while allowing only contextual EOM creation.
- Must not change: adjustments, MCP, invoice lifecycle, migrations, email, or UI.

## Scope (this PR)

Ownership lane: eom/receivables
Slice phase: Vertical slice
Max files: 5

Admit missing/empty allocations only through two EOM routes, create an active canonical customer's payment/event without invoice work, and preserve legacy contracts.

### Review Contract

- Acceptance: EOM route accepts omitted/[]; active EOM customer gets one payment/event, zero allocated cents, full unapplied cents, no invoice writes; unknown/foreign/lead/inactive contacts fail before insert.
- Default service callers, adjustments, and legacy MCP/invoice callers remain strict;
  only both EOM routes explicitly opt into `allow_unapplied`.
- Reachability: tracker targets Funnel full `main:app`; full/slim routes have parity.
  The no-invoice query uses the slim profile's contact fields only: EOM context,
  customer type, and active status. No flag, migration, credential, UI, or email changes.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R8, R11, R12, R14; R2 tests both cardinalities and R3 scopes zero-allocation lookup to active `effingham_maids` customers.
- R8: same-key transactions take global-event then source/key advisory locks; winner
  inserts one parent/event, waiter rechecks and returns matching original (or conflicts).
  Real-Postgres gather samples this; locks plus parent idempotency index carry invariant.

### Boundary-change enumeration

- Only POST create widens `allocations` to omitted/[] or existing one-to-100 rows;
  adjustments/non-routed callers remain one-to-100. No-invoice creation key-share locks
  an active EOM customer by context/type/status, then skips invoice locks/recalculation.

### Deployed-config probing

- Read-only evidence: full `main:app` is live, tracker targets Funnel `/api/v1`, and slim Render remains an undeployed disabled candidate.

### Files touched

- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/eom_api/receivables.py`
- `atlas_brain/services/receivables.py`
- `plans/PR-EOM-Receivables-Unapplied-Payments.md`
- `tests/test_receivables.py`

## Mechanism

Routes pass the EOM context into a default-strict service; only an active canonical customer can take the zero-allocation branch.

## Intentional

- Provider-only, additive behavior; no schema, UI, receipt, Gmail, or live financial-data operation.
- The slim profile does not own lead-pipeline migrations, so this payment boundary
  deliberately does not query lead-pipeline columns; `contact_type='customer'`
  is the customer-versus-lead eligibility boundary.

## Deferred

- #2362 later slices; #2363 H-01/H-06 remain separate. No schema or UI expansion here.

## Verification

- Focused/optional Postgres concurrency, EOM profile, ruff, diff/plan and full gates; no live financial write.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/invoicing/receivables.py` | 5 |
| `atlas_brain/eom_api/receivables.py` | 5 |
| `atlas_brain/services/receivables.py` | 41 |
| `plans/PR-EOM-Receivables-Unapplied-Payments.md` | 82 |
| `tests/test_receivables.py` | 291 |
| **Total** | **424** |
