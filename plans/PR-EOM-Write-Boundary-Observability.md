# PR-EOM-Write-Boundary-Observability

## Why this slice exists

Website #113, the last child of #107 (Slice 0), under umbrella #105.

#107 promises that incomplete cross-system customer operations stay "durable,
observable, idempotently retryable, and reconcilable." Slice 0C
(eom-timetracker#149, website#155, ATLAS#2334) shipped the durable and retryable
halves. Nothing watches the boundary, so a bypass — a writer reaching the
database without the domain tier, or a tracker customer minted with no Atlas
contact — is silent until somebody runs an audit by hand. That is precisely how
the original defect survived long enough to need a backfill on 2026-08-09.

The baseline is clean right now and was measured, not assumed, immediately
before this was written: Atlas unknown-source 0 of 711 EOM contacts, null-tenant
0, operator-provenance-without-event 0; tracker unlinked-customers 0,
stale-pending-reservations 0. #113 requires that clean start — "a monitor
calibrated against the bug it is meant to catch" is the failure it guards
against — and the window closes the first time a legacy writer fires.

### Problem-derived contract

- Root cause: the canonical write boundary has no runtime observer. Every
  invariant 0A–0C established is enforced only at the code paths that agreed to
  honor it; a write that arrives another way leaves no signal anyone sees.
- Correct fix must touch/change: add a scheduled check over the two datastores
  the invariants live in (Atlas `contacts` / `eom_lead_lifecycle_events`, tracker
  `customers` / `eom_customer_atlas_reservations`), with alerting on breach and
  a declared clean baseline at enablement.
- Must not change: any write path. This slice observes only. No schema change,
  no migration, no change to the operator mutation boundary, the tracker saga, or
  the website portal.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice
Max files: 5

1. Add `scripts/eom_write_boundary_audit.py`: five signals, transition-based
   alerting to the existing ntfy topic, non-zero exit on breach.
2. Add its systemd unit and hourly timer, plus tests proving each signal fires
   on a violation and stays silent on a clean reading.

### Review Contract

- Acceptance criteria:
  1. Each of the five signals breaches on its own violation and on nothing else
     — settled by
     `tests/test_eom_write_boundary_audit.py::test_each_signal_breaches_on_its_own_violation`
     (parametrized over all five) and `::test_a_clean_reading_reports_ok`.
  2. The Atlas SQL actually detects each violation against the real migrated
     schema, and a well-formed contact trips nothing — settled by
     `::test_the_atlas_query_detects_each_violation_and_ignores_clean_rows`,
     which seeds a clean contact plus one violation of each kind and asserts
     `[0,0,0]` then `[1,1,1]`.
  3. An unreadable datastore alerts rather than reporting clean — settled by
     `::test_an_unreadable_source_breaches_rather_than_reporting_clean`.
  4. Alerting fires on the transition into breach, stays quiet until the
     re-alert interval, and sends exactly one recovery notice — settled by
     `::test_first_breach_alerts_then_stays_quiet_until_the_reminder` and
     `::test_recovery_notifies_exactly_once`.
  5. Partial or unparseable query output is refused rather than read as a low
     count — settled by
     `::test_partial_output_is_refused_rather_than_read_as_a_low_count`.
- Reachability proof: systemd user timer → `eom-write-boundary-audit.service` →
  `eom_write_boundary_audit.py` → `psql` against Atlas and `render psql` against
  the tracker → ntfy push. Observable effect: a push notification and a non-zero
  unit result. Verified by running the script against live production data
  (read-only): all five signals reported 0, exit 0.
- Affected surfaces: new script, new unit files, new test. No application module
  is imported or modified; nothing in `atlas_brain/` changes.
- Risk areas: alert fatigue if a signal is noisy; the Render CLI losing auth and
  blinding the tracker half; a partially parsed count reading as healthy.
- Reviewer rules triggered: R1 (problem-derived contract), R2 (reachability),
  R5 (test evidence).

### Boundary-change enumeration

N/A - no boundary change. This slice adds an observer. It changes no guard,
validator, normalizer, resolver, router, or admission path, and writes to no
production table.

### Deployed-config probing

- Deployed/default config values: Atlas DSN `postgresql://atlas:atlas@localhost:5433/atlas`;
  tracker Render DB id `dpg-d723r3buibrs739nnpg0-a`; ntfy topic
  `eom-atlas-api-health-64ac52777bf9` (the existing healthcheck topic, reused so
  no new phone subscription is needed); re-alert every 24 runs at an hourly
  cadence.
- Explicit value probe: every setting is overridable by flag and by
  `EOM_AUDIT_*` env var; the tests drive `--state-dir` explicitly.
- Absent value probe: with no state file, `decide_alert` treats the run as the
  first — covered by `::test_a_clean_run_from_cold_state_says_nothing` and the
  first-breach case.
- Default-session/default-context probe: unreadable-source handling is the
  default-context probe; it breaches rather than defaulting to clean.
- Side-effect ordering: state is written before the notification is sent, so a
  failed push cannot cause an alert storm on the next run.

### Files touched

- `config/eom-write-boundary-audit.service`
- `config/eom-write-boundary-audit.timer`
- `plans/PR-EOM-Write-Boundary-Observability.md`
- `scripts/eom_write_boundary_audit.py`
- `tests/test_eom_write_boundary_audit.py`

## Mechanism

The script shells out to `psql` and the Render CLI rather than importing the
application, so it keeps working when the application does not. It is
stdlib-only and runs under bare `/usr/bin/python3`, and the unit runs it from an
installed copy under `~/.local/bin` rather than from the runtime worktree —
that worktree was deleted once (2026-07-31) and took the API down with it. A
monitor living inside the thing it watches goes silent exactly when it is needed.

Signals, all threshold 0:

| Signal | Catches |
|---|---|
| `atlas_unknown_source` | a writer emitting a `source` no known EOM writer produces |
| `atlas_null_tenant` | an untenanted contact write |
| `atlas_operator_provenance_without_event` | a row carrying operator provenance with no lifecycle event — i.e. written around the domain tier |
| `tracker_unlinked_customers` | the primary defect recurring: a customer Atlas does not know |
| `tracker_stale_pending_reservations` | a saga nobody came back for |

The known-source allowlist is derived from the code sweep of every create path,
not from the values present in the table: an allowlist built from observed data
would bless a bypass that had already run.

An unmeasured signal counts as breached. A monitor that reports clean while
seeing nothing converts an outage into false assurance, which is the same
silent-failure class this slice exists to close.

Alerting copies the installed atlas-api healthcheck (under `~/.local/bin`, not
versioned in this repo): fire on the transition into breach,
re-alert every N consecutive runs, one recovery notice.

## Intentional

- Reuses the existing healthcheck ntfy topic rather than opening a new one, so
  no new phone subscription is needed; Title and Tags keep the two
  distinguishable. Easy to split later if drift alerts prove noisy against
  outage alerts.
- Hourly, not minutely. This is drift detection; `atlas-api-healthcheck.timer`
  already covers liveness at five minutes.
- Reads the tracker through the Render CLI rather than adding an Atlas→tracker
  API credential. The credential would be a new authenticated surface on the
  tracker for a read a local operator tool can already do.
- Shell-outs over a database driver: fewer dependencies for a process whose job
  is to survive the failure of everything around it.

## Deferred

- Website #156: the legacy tracker Site routes can still mint an unlinked
  Customer. Declared as the known exception #113 permits. This slice measures
  whether those paths are ever exercised, which is the evidence #156 should be
  built with.
- Website #158 (reconcile identity race) and #163 (shared request-body artifact).
- Migrating the existing atlas-api healthcheck script into the repo; it has the
  same unversioned-ops-script weakness.

Parked hardening: none.

## Verification

- Pending before push:
  - `pytest tests/test_eom_write_boundary_audit.py` against a throwaway
    `postgres:16` (never the live `atlas` database) — Result: pass, 17 passed.
  - Negative control on the SQL: allowlist the seeded rogue writer and confirm
    the schema-backed test fails — Result: pass, failed with
    `assert [0, 1, 1] == [1, 1, 1]` as intended.
  - Live read-only dry run against production data under bare `/usr/bin/python3`
    — Result: pass, all five signals 0, exit 0.
- After merge, before enabling the timer: re-measure the baseline, install the
  script, send one real alert and confirm it arrives on Juan's phone. #113
  requires the channel be verified to deliver rather than merely emit; that is a
  manual confirmation and cannot be self-certified.

## Estimated diff size

| File | LOC |
|---|---:|
| `config/eom-write-boundary-audit.service` | 26 |
| `config/eom-write-boundary-audit.timer` | 13 |
| `plans/PR-EOM-Write-Boundary-Observability.md` | 190 |
| `scripts/eom_write_boundary_audit.py` | 351 |
| `tests/test_eom_write_boundary_audit.py` | 240 |
| **Total** | **820** |
