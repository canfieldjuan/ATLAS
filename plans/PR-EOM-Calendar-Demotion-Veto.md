# PR-EOM-Calendar-Demotion-Veto

## Why this slice exists

Website issue canfieldjuan/Effingham_Office_Maids_Website#138 records a
cross-repo defect in the Atlas EOM calendar/portal customer sync: calendar
import can admit bookings up to twelve months ahead, but the portal-sync
demotion veto only checks four months ahead.

### Problem-derived contract

- Root cause: the same future-booking horizon is encoded as separate literals in
  two scripts. `scripts/import_eom_customers_live.py` imports future booking
  customers through a twelve-month default window, while
  `scripts/sync_eom_portal_customers.py` builds the calendar demotion veto
  through a four-month default window. A customer imported from a booking five to
  twelve months out can therefore be absent from the portal roster and also
  absent from the demotion guard, so the "calendar vetoes demotion" owner rule is
  shorter than the importer's own admitted future window.
- Correct fix must touch/change: the importer must expose a single named default
  for the forward booking-import horizon; the portal-sync demotion guard must
  default to that importer-owned horizon instead of a shorter literal; regression
  coverage must prove a booking beyond four months but within the importer
  default produces guard keys and vetoes demotion.
- Must not change: do not narrow the import horizon; do not widen the demotable
  source set; do not change calendar parsing, cancellation precedence, portal
  authentication, CRM contact matching, receipt behavior, or any EOM
  acknowledgement/onboarding lanes.

## Scope (this PR)

Ownership lane: eom-calendar-demotion-veto
Slice phase: production hardening

1. Make the demotion-veto future window follow the importer default horizon.
2. Add regression coverage for a far-future active calendar event that is beyond
   the old four-month guard but inside the import horizon.

### Review Contract

- Acceptance criteria:
  - The importer parser's `--months-forward` default is a named constant in
    `scripts/import_eom_customers_live.py`, preserving the existing twelve-month
    default.
  - `fetch_calendar_guard_keys()` in `scripts/sync_eom_portal_customers.py`
    defaults `months_forward` from that importer-owned constant, so the demotion
    veto covers the same future window that default imports admit.
  - `tests/test_sync_eom_portal_customers.py` proves an active booking roughly
    ten months ahead emits a guard email and keeps the unmatched CRM contact
    active; this is the old failure window because it is beyond four months and
    within twelve months.
- Reachability proof: `run()` calls `fetch_calendar_guard_keys()` before
  `demote_unmatched()` in `scripts/sync_eom_portal_customers.py`; the regression
  test exercises the real guard producer and `demote_unmatched()` path.
- Affected surfaces: EOM live calendar import default constant; EOM portal-sync
  demotion veto; sync regression tests.
- Risk areas: demotion safety window, cancellation precedence, calendar-provider
  query range, and avoiding accidental changes to import output or CRM write
  eligibility.
- Reviewer rules triggered: R2 test evidence, R14 codebase verification and
  boundary-probe for a guard-shaped change.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: calendar-active customer demotion veto in
  `fetch_calendar_guard_keys()` -> `demote_unmatched()`.
- Replaced-path behaviors: four-month default veto horizon is replaced by the
  importer-owned default forward horizon.
- Guard-relevant fields: calendar event start/end range, parsed email/phone/name
  and address guard keys, CRM contact email/phone/name/address.
- Caller x input shape: `run()` still calls `fetch_calendar_guard_keys()` with
  no explicit override; tests may still pass explicit `months_forward` values.

### Closure declaration

- Boundary inventory membership: CLOSED for this slice. The changed decision is
  the future time window that feeds the existing demotion-veto guard; the
  identity-key inventory remains the code-owned set already read by
  `fetch_calendar_guard_keys()` and `on_calendar()`:
  phone/email/address/name keys plus calendar event start/end/cancellation
  state.
- Membership source: DERIVED from the existing implementation at use time.
  Calendar guard keys are emitted by `fetch_calendar_guard_keys()` from
  `import_eom_customers_live` parsed records, and CRM row membership is consumed
  by `on_calendar()`. This PR does not introduce an authored duplicate list of
  customer identity channels.
- Outside-set behavior: event dates outside the computed
  `months_forward * 30` window produce no guard key and therefore do not veto
  demotion. Calendar/CRM fields outside the existing identity-key set remain
  non-participating, so they cannot silently widen the veto or demotable source
  boundary.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: no deployed environment value is changed; the
  default script argument remains twelve months forward.
- Explicit value probe: existing explicit `months_forward` callers continue to
  override the default.
- Absent value probe: the no-argument guard path now resolves to the importer
  default.
- Default-session/default-context probe: the regression test exercises the
  no-argument guard call.
- Side-effect ordering: unchanged; `run()` still obtains guard keys before any
  demotion attempt and refuses demotion when guard construction fails.

### Files touched

- `plans/PR-EOM-Calendar-Demotion-Veto.md`
- `scripts/import_eom_customers_live.py`
- `scripts/sync_eom_portal_customers.py`
- `tests/test_sync_eom_portal_customers.py`

## Mechanism

`import_eom_customers_live.py` names the default forward import horizon.
`sync_eom_portal_customers.py` uses that value as the default forward guard
horizon. The test's fake calendar provider filters returned events by the
requested time window, then proves a ten-month-ahead active booking appears in
guard keys and vetoes demotion.

## Intentional

- The import horizon stays twelve months; this slice does not narrow what the
  calendar importer sees.
- The demotable source set stays exactly as-is; this slice only changes how far
  into booking calendars the existing veto looks.
- No live calendar or CRM write is performed by verification.

## Deferred

None.

Parking predicate: this slice parks only hardening or polish that does not
change whether default-import-horizon calendar bookings can veto portal-sync
demotion and does not affect PR-shape, reconciliation, or CI gates.

Parked hardening: none.

## Verification

- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-eom-calendar-demotion-veto.local.md pytest -q tests/test_sync_eom_portal_customers.py` -- 54 passed.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-eom-calendar-demotion-veto.local.md pytest -q tests/test_eom_live_calendar_import.py` -- 55 passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-EOM-Calendar-Demotion-Veto.md` | 159 |
| `scripts/import_eom_customers_live.py` | 3 |
| `scripts/sync_eom_portal_customers.py` | 5 |
| `tests/test_sync_eom_portal_customers.py` | 32 |
| **Total** | **199** |
