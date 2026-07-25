# PR-EOM-Calendar-Email-Veto

## Why this slice exists

Issue #2191 and the 2026-07-24 EOM handoff audit found a safety gap in the
portal-sync demotion path: the canonical live Calendar importer extracts and
deduplicates email, but the demotion veto neither emits email guard keys nor
loads and compares candidate emails. An active customer whose only shared
identity is email can therefore be demoted. This is a production data-safety
fix and blocks the EOM reconciliation apply.

### Problem-derived contract

- Root cause: the Calendar guard consumer implements only a subset of the
  canonical producer's customer identity channels. Email survives
  `parse_events` and cross-calendar `dedup_records`, but
  `fetch_calendar_guard_keys`, `on_calendar`, and the demotion candidate query
  omit it.
- Correct fix must touch/change: the real Calendar guard producer must emit a
  normalized email key only for the merged current non-cancelled record; the
  demotion candidate query must load email; the matcher must normalize and
  compare it; behavioral tests must exercise the real producer with a fake
  external provider and prove active, normalized, cancellation, and unrelated
  cases.
- Must not change: portal matching, demotable provenance, phone/address/name
  semantics, cross-calendar cancellation/recency rules, dry-run/apply defaults,
  CRM writes, Calendar writes, or customer-visible product shape.

## Scope (this PR)

Ownership lane: eom-crm/calendar-veto
Slice phase: production hardening

1. Carry normalized email identity through the existing Calendar demotion veto.
2. Prove the behavior through the real guard producer and the demotion entrypoint.

### Review Contract

- Acceptance criteria: an email-only active booking vetoes demotion;
  case/whitespace variants match; a latest cancellation cannot veto through an
  older email; unrelated email does not veto; existing identity channels and
  demotion scope remain unchanged.
- Reachability proof: call `fetch_calendar_guard_keys` with the canonical
  importer and a fake `GoogleCalendarProvider`, then pass its output to
  `demote_unmatched` and observe that the active email-only candidate is kept
  while the stale-cancelled and unrelated candidates receive guarded inactive
  updates.
- Affected surfaces: `scripts/sync_eom_portal_customers.py`, its focused test
  suite, and the durable review checklist.
- Risk areas: stale email surviving cancellation, inconsistent normalization,
  hand-built fixtures diverging from provider output, and accidental widening
  of demotion eligibility.
- Reviewer rules triggered: R1, R2, R4, R5, R8, R10, R12, R13, R14.

### Files touched

- `REVIEW_MISSES.md`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/PR-EOM-Calendar-Email-Veto.md`
- `scripts/sync_eom_portal_customers.py`
- `tests/test_sync_eom_portal_customers.py`

## Mechanism

Extend the guard key set with `emails`. After the canonical importer parses and
cross-calendar merges events, emit the stripped lowercase email only for each
non-cancelled merged record. Load candidate email in the existing
provenance-scoped SELECT and check the same normalized form in `on_calendar`.
Producer-driven tests replace duplicated source-shape assertions for the email
contract.

## Intentional

- Email follows the merged record's existing cancellation state; the fix does
  not invent a second email-specific cancellation algorithm.
- The fake substitutes only the external Google provider. Parsing,
  cross-calendar deduplication, guard emission, matching, and demotion remain
  real.

## Deferred

- Durable reconciliation receipts remain a separate #2190 slice.
- Atomic portal reconciliation remains separate PR #2193.

Parked hardening: none.

## Verification

- Focused pytest over the portal-sync and live-calendar import suites -- 99 passed; one third-party `pynvml` deprecation warning.
- Ruff over the changed runtime file -- passed.
- Ruff over the focused test with F401 and F841 ignored -- passed; those two findings pre-exist this diff in unrelated tests.
- Python compilation of both changed Python files -- passed.
- Git whitespace validation -- passed.
- PR plan synchronization -- passed.
- Pending before push: the managed pre-push review.

## Estimated diff size

| File | LOC |
|---|---:|
| `REVIEW_MISSES.md` | 1 |
| `docs/SESSION_BOOTSTRAP.md` | 1 |
| `plans/PR-EOM-Calendar-Email-Veto.md` | 106 |
| `scripts/sync_eom_portal_customers.py` | 19 |
| `tests/test_sync_eom_portal_customers.py` | 106 |
| **Total** | **233** |
