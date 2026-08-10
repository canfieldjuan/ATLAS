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
0, operator-provenance-without-event 0. (The tracker-side baseline was also
zero; those signals are deferred, see the split note below.) #113 requires that clean start — "a monitor
calibrated against the bug it is meant to catch" is the failure it guards
against — and the window closes the first time a legacy writer fires.

### Why this is one slice, over the 400-line target

The diff is a monitor and the proof that it detects anything. Splitting those
ships a detector whose detection is unverified, which is the failure the ratchet
and review discipline exist to prevent -- and the script and its tests each
exceed the target on their own, before the plan doc the PR gate itself mandates.

The one genuine split has already been taken: the tracker-side signals came out
entirely (Effingham_Office_Maids_Website#167) after three rounds showed their
blockers were not closeable in this repository. What remains is the smallest
thing that both alerts and can be shown to alert.

### Problem-derived contract

- Root cause: the canonical write boundary has no runtime observer. Every
  invariant 0A–0C established is enforced only at the code paths that agreed to
  honor it; a write that arrives another way leaves no signal anyone sees.
- Correct fix must touch/change: add a scheduled check over the two datastores
  the invariants live in (Atlas `contacts` / `eom_lead_lifecycle_events`), with
  alerting on breach and
  a declared clean baseline at enablement.
- Must not change: any write path. This slice observes only. No schema change,
  no migration, no change to the operator mutation boundary, the tracker saga, or
  the website portal.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice
Max files: 6

**Split.** This PR carries the ATLAS half of #113 only. The two tracker-side
signals were removed after three review rounds established that their blockers
cannot be closed in this repository: the tracker's canonical schema lives in
eom-timetracker, no ATLAS CI job can reach the tracker database, and the linkage
predicate needs a cross-datastore reconciliation rather than a NULL check. They
are carried by Effingham_Office_Maids_Website#167, and #113 stays open until it
lands.

1. Add `scripts/eom_write_boundary_audit.py`: the three Atlas-side signals,
   per-signal transition alerting, non-zero exit on breach.
2. Add its systemd unit and hourly timer, plus tests proving each signal fires
   on a violation and stays silent on a clean reading.

### Review Contract

- Acceptance criteria:
  1. Each of the three Atlas signals breaches on its own violation and on nothing else
     — settled by
     `tests/test_eom_write_boundary_audit.py::test_each_signal_breaches_on_its_own_violation`
     (parametrized over all three) and `::test_a_clean_reading_reports_ok`.
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
  `eom_write_boundary_audit.py` → `psql` against Atlas → ntfy push. Observable
  effect: a push notification and a non-zero unit result. Verified by running the
  script against live production data (read-only): all three signals reported 0,
  exit 0.
- Affected surfaces: new script, new unit files, new test. No application module
  is imported or modified; nothing in `atlas_brain/` changes.
- Risk areas: alert fatigue if a signal is noisy; losing the Atlas datastore and
  reporting clean; a partially parsed count reading as healthy.
- Reviewer rules triggered: R1 (problem-derived contract), R2 (reachability),
  R5 (test evidence).

### Boundary-change enumeration

N/A - no boundary change. This slice adds an observer. It changes no guard,
validator, normalizer, resolver, router, or admission path, and writes to no
production table.

### Deployed-config probing

- Deployed/default config values: settings carry the repo's `ATLAS_` prefix
  (`ATLAS_EOM_AUDIT_*`), matching how `scripts/` reads configuration; the strict
  typed-settings rule in CLAUDE.md governs `atlas_brain/`, and binding this
  monitor to the application's config module would give it the failure mode it
  exists to avoid. Atlas DSN defaults to `postgresql://atlas:atlas@localhost:5433/atlas`
  and is passed to `psql` by environment, never argv, because
  `/proc/<pid>/cmdline` is world-readable; ntfy topic supplied at deploy
  time via `EOM_AUDIT_NTFY_TOPIC` with NO default in the repo, because on ntfy.sh
  the topic name is the channel credential and this repository is public;
  re-alert every 24 runs at an hourly cadence.
- Explicit value probe: every setting is overridable by flag and by
  `EOM_AUDIT_*` env var; the tests drive `--state-dir` explicitly.
- Absent value probe: with no state file, `decide_alert` treats the run as the
  first — covered by `::test_a_clean_run_from_cold_state_says_nothing` and the
  first-breach case.
- Default-session/default-context probe: unreadable-source handling is the
  default-context probe; it breaches rather than defaulting to clean.
- Side-effect ordering: the notification is sent FIRST and state advances only
  once delivery is confirmed, so a failed push is retried on the next run
  instead of being recorded as sent. Not a storm: while delivery is broken
  nothing reaches anyone.

### Files touched

- `config/eom-write-boundary-audit.service`
- `config/eom-write-boundary-audit.timer`
- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `plans/PR-EOM-Write-Boundary-Observability.md`
- `scripts/eom_write_boundary_audit.py`
- `tests/test_eom_write_boundary_audit.py`

### Set-valued dependency declaration

`KNOWN_EOM_SOURCES` is a **CLOSED** set: the `source` values EOM contact writers
emit. Sourced from the 2026-08-05 code sweep of every create path that reaches
`DatabaseCRMProvider`, not from the values observed in the table -- an allowlist
built from observed data would bless a bypass that had already run. Out-of-set
behaviour is to ALERT, which is the asymmetric-safe direction: a new legitimate
writer produces one noisy alert and a one-line addition here, while an unknown
writer is exactly what this signal exists to surface.

## Mechanism

The script shells out to `psql` and the Render CLI rather than importing the
application, so it keeps working when the application does not. It is
stdlib-only and runs under bare `/usr/bin/python3`, and the unit runs it from an
installed copy under `~/.local/bin` rather than from the runtime worktree —
that worktree was deleted once (2026-07-31) and took the API down with it. A
monitor living inside the thing it watches goes silent exactly when it is needed.

Signals (Atlas only in this PR), all threshold 0:

| Signal | Catches |
|---|---|
| `atlas_unknown_source` | a writer emitting a `source` no known EOM writer produces |
| `atlas_null_tenant` | an untenanted contact write |
| `atlas_operator_provenance_without_event` | a row carrying operator provenance with no lifecycle event — i.e. written around the domain tier |

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

- The ntfy topic is never committed. It is the channel credential, so it comes
  from `EOM_AUDIT_NTFY_TOPIC` at deploy time and a blank value is refused rather
  than publishing nowhere.
- Hourly, not minutely. This is drift detection; `atlas-api-healthcheck.timer`
  already covers liveness at five minutes.
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
    `postgres:16` (never the live `atlas` database) — Result: pass, 20 passed.
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
| `tests/test_eom_write_boundary_audit.py` | 553 |
| `scripts/eom_write_boundary_audit.py` | 489 |
| `plans/PR-EOM-Write-Boundary-Observability.md` | 224 |
| `config/eom-write-boundary-audit.service` | 37 |
| `config/eom-write-boundary-audit.timer` | 13 |
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 5 |
| **Total** | **1321** |
