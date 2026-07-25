# PR-EOM-Atomic-Portal-Reconciliation

## Why this slice exists

ATLAS #2162 was originally described as non-blocking hardening because the
final portal-ID stamp rejects a conflicting link. A cold source audit found
that the create-race path can call the general CRM provider, receive an
existing match, merge portal fields into that contact, and only then discover
the portal-link conflict. The final error therefore does not imply zero writes.
This blocks the first production portal reconciliation and onboarding Slice B.

### Problem-derived contract

- Root cause: portal reconciliation is split across provider dedup/merge,
  matched-field update, legacy claim, and a later portal-ID stamp. The rejection
  predicate does not guard the first mutation, and the run has no roster-wide
  collision proof before apply begins.
- Correct fix must touch/change: add an opt-in non-merging provider resolution
  mode; make the portal sync use it; replace claim/update/stamp sequencing with
  one portal-specific conditional mutation that guards tenant/archive, current
  identity, and portal link before changing fields; insert new rows with all
  portal-owned fields and metadata in the initial insert; and preflight the
  complete roster for duplicate normalized identities and duplicate CRM
  resolutions before any apply write.
- Must not change: the provider's default merge behavior; public CRM/API
  contracts; the portal backend; schema; demotion provenance; dry-run default;
  Calendar-veto semantics; leads, money paths, or unrelated callers. This slice
  does not add the email veto or execution receipts tracked in #2191/#2190.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: production hardening

The phase is appropriate out of sequence because this closes a customer-data
corruption risk discovered in the already-built reconciliation path and blocks
its first production apply.

1. Enforce zero-write rejection and atomic portal reconciliation in the
   existing operator entrypoint.
2. Add behavioral mutation-boundary and provider-compatibility regressions,
   plus the escaped-defect gate in the reviewer flywheel.

### Review Contract

- Acceptance criteria:
  - [x] A create-race match with a conflicting portal link returns an error and
        performs zero updates.
  - [x] Same-tenant unlinked and same-linked contacts reconcile through one
        conditional statement.
  - [x] A NULL-tenant legacy row is claimed and reconciled in that same
        statement; a lost claim or identity drift writes nothing.
  - [x] A new contact is inserted already carrying tenant, provenance, tags,
        phone, and portal metadata.
  - [x] Provider `create_contact` keeps merge behavior by default, while
        `merge_existing=False` returns only same-tenant matches without claim or
        update and does not consult NULL-tenant fallback rows.
  - [x] Apply preflight rejects duplicate normalized portal identities or two
        portal customers resolving to one CRM contact before every write.
  - [x] Dry-run and demotion contracts remain backward compatible.
- Reachability proof: tests invoke `sync_one` and `run` through the real
  operator module with DB/provider edge doubles and assert the executed
  mutation set and returned outcome; provider tests invoke the production
  class.
- Affected surfaces: `DatabaseCRMProvider.create_contact`, EOM portal sync
  resolution/reconciliation/preflight, focused tests, reviewer process ledger.
- Risk areas: customer data corruption, tenant claim race, identity TOCTOU,
  create race, duplicate roster identity, backward compatibility, idempotency.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R8, R10, R13, R14.

### Files touched

- `REVIEW_MISSES.md`
- `atlas_brain/services/crm_provider.py`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/PR-EOM-Atomic-Portal-Reconciliation.md`
- `scripts/sync_eom_portal_customers.py`
- `tests/test_crm_read_scoping.py`
- `tests/test_eom_live_calendar_import.py`
- `tests/test_sync_eom_portal_customers.py`

## Mechanism

`create_contact(..., merge_existing=False)` limits dedup resolution to
same-tenant email matches, returns them without claim/merge, and otherwise
inserts the complete supplied payload. The portal sync passes that mode after
its stronger phone/email/address resolver. Existing or race-resolved contacts
flow to one dynamic SQL update whose `WHERE` clause proves the contact is
unarchived, belongs to EOM or is claimable, still matches the resolution
identity, and is unlinked or linked to this portal customer. The same statement
claims tenant, replaces portal-managed fields/tags, and merges the portal ID.

Before apply, a read-only preflight resolves every roster row, rejects duplicate
normalized portal channels and duplicate contact IDs, and caches the results
consumed by the write loop so no earlier customer is changed before a later
collision is discovered.

## Intentional

- No broad unique index is added to `contacts`; identity channels are not
  globally unique and the needed invariant is specific to one portal roster.
- Non-merging provider mode intentionally skips phone dedup because the
  provider's substring behavior is weaker than the portal resolver. Phone still
  lands on a genuinely new row in its initial insert.
- The atomic statement is portal-specific instead of broadening the generic
  `_guarded_update`, whose callers do not share this portal-link invariant.
- This exceeds the 400-LOC soft cap because the existing portal-sync test
  harness encoded the removed three-write sequence and had to be converted to
  observe one conditional mutation. Splitting the provider seam from its only
  consumer would create an unexercised intermediate API and weaken review of
  the zero-write boundary.

## Deferred

- Email parity in the Calendar demotion veto: #2191.
- Durable SHA-bound execution receipts: #2190.
- Production apply and onboarding Slice B remain blocked until all three
  blockers merge and pass the operator gate.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_sync_eom_portal_customers.py
  tests/test_crm_read_scoping.py tests/test_eom_live_calendar_import.py -q`
  — 173 passed; one third-party `pynvml` deprecation warning.
- Ruff passed for `atlas_brain/services/crm_provider.py` and
  `scripts/sync_eom_portal_customers.py`.
- Python byte-compilation passed for `scripts/sync_eom_portal_customers.py`
  and `atlas_brain/services/crm_provider.py`.
- Local PostgreSQL 16 transaction probe — the production tag expression
  removed `past_customer`, added `commercial`/`portal`, preserved concurrent
  foreign tag `vip` and source `calendar_import`, then rolled back.
- Local PostgreSQL 16 conflict probe — a row linked to portal ID 9 rejected an
  attempted portal-ID 7 reconciliation with `rows_mutated = 0`; name, metadata,
  tags, and source remained unchanged; transaction rolled back.
- `git diff --check` — passed.
- Pending at push: managed `scripts/push_pr.sh` local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `REVIEW_MISSES.md` | 1 |
| `atlas_brain/services/crm_provider.py` | 29 |
| `docs/SESSION_BOOTSTRAP.md` | 1 |
| `plans/PR-EOM-Atomic-Portal-Reconciliation.md` | 152 |
| `scripts/sync_eom_portal_customers.py` | 291 |
| `tests/test_crm_read_scoping.py` | 124 |
| `tests/test_eom_live_calendar_import.py` | 2 |
| `tests/test_sync_eom_portal_customers.py` | 272 |
| **Total** | **872** |
