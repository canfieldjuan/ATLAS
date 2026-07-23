# PR-EOM-Calendar-Demotion-Guard

## Why this slice exists

Owner directive (2026-07-23, during the #2161 dry-run review): demotion
candidates must be checked against the LIVE booking calendars before anyone
is retired. The portal defines `active`, but portal lag or a resolution miss
could otherwise demote a customer who is currently on the schedule. Verified
surface: the #2158 calendar machinery (provider, parse, identity keys) is
already live-proven and reused wholesale.

### Problem-derived contract

- Root cause: #2161's demotion verifies portal membership only; a current
  customer absent from the portal (lag or match miss) would be demoted.
- Correct fix must touch/change: a calendar veto in the demotion pass --
  fetch the three booking calendars (recent month through the next four),
  derive phone/address/name keys via the slice-A parser, KEEP any candidate
  matching a key (reported as "add them to the portal"), and REFUSE to
  demote at all when the calendars cannot be fetched (fail closed, counted
  as a run error).
- Must NOT change: sync/resolution/stamping behavior; slice-A machinery
  (imported, unmodified); demotion SQL predicates; provider code; schema.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: vertical slice

1. `scripts/sync_eom_portal_customers.py`: `fetch_calendar_guard_keys`,
   `on_calendar`, guard wiring in `run()`/`demote_unmatched`.
2. `tests/test_sync_eom_portal_customers.py`: veto coverage.
3. This plan doc.

### Review Contract

- Acceptance criteria:
  1. A demotion candidate matching a live booking-calendar event by phone
     (extension-stripped last-10), address (case-insensitive), or
     normalized name is KEPT and reported for portal reconciliation
     (asserted).
  2. Non-matching candidates still demote through the unchanged guarded
     SQL (asserted).
  3. A calendar fetch failure SKIPS demotion entirely and fails the run
     (source-asserted) -- the veto is never silently absent.
  4. Guard keys derive from the same slice-A parser/identity semantics as
     the import itself (mechanism-asserted by reuse).
  6. The fail-closed guard also catches SystemExit (missing calendar
     config surfaces as DEMOTION SKIPPED, never a mid-apply crash), and
     name keys apply latest-record-wins recency independently since the
     merger unions by channel only (both asserted; Codex round 3).
  5. CANCELLED-latest calendar records never veto, decided on the
     CROSS-CALENDAR merged view (`dedup_records` recency runs before key
     emission, so a newer cancellation on another calendar supersedes an
     older active event) -- source-asserted with ordering (Codex rounds
     1-2, BLOCKER).
- Reachability proof: operator script; the guard runs on every demotion
  pass; `on_calendar` and the kept/demoted split are driven by tests.
- Reviewer rules triggered: R1, R2, R4, R6, R8, R14.
  - R1: implements the owner veto rule verbatim.
  - R2: every criterion asserted; machinery live-proven by #2158/#2161.
  - R4: data safety -- fail-closed guard, unchanged demotion SQL.
  - R6: calendar failure surfaces as a run error, never a guess.
  - R8: idempotent (read-only guard; keys recomputed per run).
  - R14: this contract is the reviewer checklist.

### Files touched

- `plans/PR-EOM-Calendar-Demotion-Guard.md`
- `tests/maturity_sweep/baseline_scripts.json`
- `scripts/sync_eom_portal_customers.py`
- `tests/test_sync_eom_portal_customers.py`

## Mechanism

`run()` builds `guard_keys` via `fetch_calendar_guard_keys` (slice-A
`resolve_calendar_ids` + `GoogleCalendarProvider.list_events` +
`parse_events`; window -30d..+120d) before `demote_unmatched`; each
candidate row (SELECT now includes phone/address) passes `on_calendar`
(phone last-10 / address / name key membership) -- matches are kept and
reported, the rest demote unchanged.

## Intentional

- Names participate in the veto: calendars are name-keyed by the owner, so
  a name-only match is a legitimate current-customer signal for KEEPING
  someone (the veto errs toward keeping; demotion still requires no match
  on ANY key).
- The guard window (-30d..+120d) covers "currently on my schedule"
  including monthly/quarterly cadences.

## Deferred

- Nothing new; #2162 residuals unchanged.

## Verification

- `tests/test_sync_eom_portal_customers.py` — 44 passed.
- `tests/test_eom_live_calendar_import.py` — 55 passed (adjacent).
- `python -m py_compile` — clean.
- NOT run: the live sync (operator-run; the owner re-runs dry-run after
  merge and reviews the KEPT/DEMOTE split).

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-EOM-Calendar-Demotion-Guard.md` | 105 |
| `scripts/sync_eom_portal_customers.py` | 90 |
| `tests/test_sync_eom_portal_customers.py` | 75 |
| `tests/maturity_sweep/baseline_scripts.json` | 3 |
| **Total** | **273** |
