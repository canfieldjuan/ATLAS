# PR-EOM-API-Liveness-Contract

## Why this slice exists

The operator reported that the EOM funnel became unavailable after working
earlier the same day. Live investigation found `atlas-api.service` cleanly
stopped at 12:15 CDT; its `Restart=on-failure` policy did not restart a clean
SIGTERM, and the Tailnet `/api` proxy consequently had no listener on port
8012. The existing five-minute `atlas-api-healthcheck` timer detected the
inactive unit for 84 consecutive runs, but its script only notified; it never
restored the service. This is a production availability defect, so this is a
small Production hardening slice justified by a real failed EOM provider path.
The monitor program, its installed user-systemd units, the installer that
activates those units, and the failure-path proof are indivisible: a program
without deployment wiring cannot recover production, while wiring without the
program and its regression coverage is unsafe. That is the reason this slice
exceeds the normal diff target.

### Problem-derived contract

- Root cause: The source of truth for the existing standalone liveness monitor
  is an untracked local shell script. It observes an inactive `atlas-api`
  unit but has no recovery transition. A clean stop therefore persists until a
  person manually starts the provider, even though the monitor has enough
  evidence to distinguish an expected maintenance stop from an unexpected one.
- Correct fix must touch/change: Add a source-managed, stdlib-only health-check
  implementation plus its user-systemd service/timer templates. When the
  provider is inactive and no explicit maintenance lock exists, the monitor
  must start the existing unit, re-probe the existing lead-intake CORS route,
  record/notify the outcome through the existing ntfy and desktop channels, and
  remain installed outside the runtime worktree.
  Add focused tests for the active, inactive/recovered, inactive/failed, and
  maintenance-lock branches.
- Must not change: Do not change `atlas_brain` funnel routes, authentication,
  CRM lifecycle/storage, migrations, invoice behavior, EOM tracker APIs,
  Render environment variables, Tailnet routing, browser UI, or the running
  Atlas worktree. Do not automatically restart an *active but unhealthy*
  process in this slice; preserve the existing alert-only behavior for that
  distinct failure class.

## Scope (this PR)

Ownership lane: eom-provider-liveness
Slice phase: Production hardening
Max files: 6

1. Replace the untracked alert-only health-check logic with a canonical
   installed-outside-the-runtime monitor that auto-recovers only an unexpected
   inactive `atlas-api.service`.
2. Add a maintenance-lock contract so intentional maintenance is never
   automatically undone.
3. Add executable regression coverage and source-managed unit templates that
   make the installation/configuration contract reviewable.
4. Add an explicit post-merge installer/checker that activates the installed
   user-systemd path and invokes it once without touching the live deployment
   during PR review.

### Review Contract

- Acceptance criteria:
  1. `run_healthcheck()` starts `atlas-api.service` exactly once when the
     service is inactive, the maintenance lock is absent, and a subsequent
     probe succeeds; settled by
     `tests/test_atlas_api_healthcheck.py::test_inactive_service_is_started_and_reprobed`.
  2. An existing maintenance lock prevents a start attempt and leaves an
     inactive service explicitly recorded as maintenance; settled by
     `tests/test_atlas_api_healthcheck.py::test_maintenance_lock_never_starts_service`.
  3. A start that fails, or a start followed by a failed probe, remains a
     visible down result rather than a false recovery; settled by the two
     failed-recovery tests in `tests/test_atlas_api_healthcheck.py`.
  4. An active but failed lead-intake probe does not restart the process;
     settled by
     `tests/test_atlas_api_healthcheck.py::test_active_unhealthy_service_alerts_without_restart`.
  5. The systemd template runs the installed copy outside the runtime worktree
   and reads notification settings from a user-local environment file;
   settled by `tests/test_atlas_api_healthcheck.py::test_service_template_uses_installed_script_and_private_environment`.
  6. The installer copies the reviewed monitor and both unit templates, enables
   the timer, invokes the installed health service once, and can later verify
   those deployed artifacts without writing; settled by
   `tests/test_atlas_api_healthcheck.py::test_installer_deploys_source_and_invokes_enabled_timer_path`.
- Reachability proof: On deployment, the timer invokes the installed monitor;
  `python scripts/install_atlas_api_healthcheck.py --install` copies the source,
  reloads user systemd, enables the timer, and invokes the installed health
  service once. That service makes the real OPTIONS request to
  `/api/v1/leads/intake`; `--check` subsequently verifies the installed copies,
  private topic file, and enabled/active timer without writing. Local tests
  exercise the same install command order and probe seam without stopping the
  live provider.
- Affected surfaces: source-managed standalone monitor, its explicit installer,
  user-systemd templates, notification/maintenance configuration, and focused
  tests.
- Risk areas: unexpected clean stops, deliberate maintenance, failed start,
  failed re-probe, notification configuration absence, and keeping the monitor
  independent of the application worktree.
- Reviewer rules triggered: R1, R2, R6, R11, R12, R14.

### Contract revision: current-head review findings

- New evidence: the enabled user timer still executes the legacy
  atlas-api-healthcheck.sh; neither the new Python monitor nor
  its notification environment file is installed. The current template only
  names an `install` command, so the source change has no deployment path.
  Separately, the monitor persists `next_state` after an undelivered transition
  alert, which suppresses its retry, and it drops the details returned by a
  failed `systemctl --user start` command.
- Revised root cause: source-managed monitor files alone cannot change a
  user-systemd deployment, and the alert state machine acknowledges a
  notification before it knows that delivery succeeded.
- Revised required change surface: add a narrowly scoped
  `scripts/install_atlas_api_healthcheck.py` installer/checker that copies the
  monitor and templates outside the runtime worktree, preserves or safely
  migrates the private local notification topic without logging it, reloads and
  enables the user timer, then invokes the installed health service once as the
  deployment proof. Update the template invocation, make transition state
  persist only when no alert is needed or delivery succeeds, carry bounded
  sanitized start-failure detail into the observation, and add focused tests for
  installation, timer/service activation, retry, and diagnostics.
- Revised explicit non-scope: do not run the installer against the live user
  systemd configuration in this PR; deployment remains a post-merge host
  action. Do not change `atlas_brain`, funnel routes, CRM, migrations, tracker
  APIs, Render/Tailnet topology, browser UI, active-but-unhealthy recovery, or
  any notification destination.
- Revised verification plan: unit-test source-to-installed-copy and expected
  systemd command order in temporary paths, test the retry and failed-start
  branches directly, run the focused monitor/installer tests and
  `systemd-analyze verify`, and use the installer `--check` plus its initial
  systemd service invocation as the post-merge deployment proof.

### Boundary-change enumeration

The recovery decision is a system state boundary, not an open-input guard.

- Boundary path/seam: `run_healthcheck()` deciding whether to issue
  `systemctl --user start`.
- Replaced-path behaviors: inactive service previously notified only; it now
  recovers only if maintenance is absent. Active/healthy remains no-op;
  active/unhealthy remains alert-only.
- Guard-relevant fields: unit active state, maintenance-lock path, start result,
  and post-start CORS probe result.
- Caller x input shape: timer invocation x active/healthy; timer x inactive/no
  lock; timer x inactive/lock; timer x inactive/start failure; timer x
  active/failed probe.

### Deployed-config probing

- Deployed/default config values: default service is `atlas-api.service`; default
  probe is local `OPTIONS /api/v1/leads/intake`; maintenance lock defaults under
  the user Atlas config directory; the installer requires or safely migrates
  the notification topic into the user-local environment file before enabling
  the timer.
- Explicit value probe: focused tests inject service, lock, and probe values.
- Absent value probe: absent lock permits recovery; absent notification topic
  must not be committed or silently replaced with a public/default topic.
- Default-session/default-context probe: the systemd template loads a missing
  environment file optionally for manual recovery, while the installer refuses
  to enable an alerting timer until it can preserve a private topic.
- Side-effect ordering: decide maintenance before any start; issue a start only
  after inactive evidence; record success only after the post-start probe.

### Guard-closure declaration

- `TOPIC_RE` is a CLOSED, DERIVED topic grammar: membership is evaluated from
  the bounded ASCII grammar at `scripts/install_atlas_api_healthcheck.py:27,83-87`,
  rather than copied from an operator-maintained list. The installer accepts only
  a normalized string in that grammar; a missing or nonmatching value fails
  before it writes copies or enables systemd, which is safer than directing an
  alert to an unknown destination. The focused grammar-product test exercises
  leading-token, body-family, length, and whitespace-wrapper boundaries against
  an independent oracle.
- `PENDING_ALERTS` is CLOSED and ENUMERATED by the three alert events that this
  monitor can emit (`down`, `recovered`, and `auto-recovered`). An unrecognized
  persisted notification record is not replayed; `pending_notification()`
  discards it and the monitor recomputes the next state from the current
  observation, so arbitrary stored values cannot choose a notification path.

### Files touched

- `config/atlas-api-healthcheck.service`
- `config/atlas-api-healthcheck.timer`
- `plans/PR-EOM-API-Liveness-Contract.md`
- `scripts/atlas_api_healthcheck.py`
- `scripts/install_atlas_api_healthcheck.py`
- `tests/test_atlas_api_healthcheck.py`

## Mechanism

The monitor is a standalone Python program installed under `~/.local/bin`,
not loaded from the Atlas runtime worktree. It checks the maintenance lock
before observing or starting the unit. If the service is inactive without that
lock, it calls the existing user-systemd unit once and then runs the same
lead-intake OPTIONS probe that the current monitor uses. It records and notifies
only after the final outcome is known. The templates keep the timer independent
of `atlas-api.service` and load the private notification topic from a local
environment file rather than version control. The companion installer copies
the reviewed source and templates, migrates an existing private topic without
printing it when necessary, reloads/enables user systemd, and invokes the
installed service once; later `--check` is read-only.

## Intentional

- Do not rely on `Restart=always` as the root fix: an explicit systemd stop can
  still suppress restart behavior, while the independent timer can recover the
  observed inactive state.
- Do not auto-restart an active-but-unhealthy provider. That needs a distinct
  diagnosis and would widen this slice from the observed inactive-unit failure.
- Do not commit the ntfy topic or read it from application configuration; it is
  a private notification credential and remains in a local environment file.
- Do not consider an alert transition acknowledged until its ntfy push succeeds;
  failed delivery records a pending transition so the next invocation retries
  the same event.

## Deferred

- Render-to-Atlas reachability remains a separate deployment-topology slice.
  The current `ATLAS_FUNNEL_BASE_URL` is not proven reachable from Render by
  this local liveness monitor, and this PR must not change that network boundary.
- A deployment-time authenticated, Render-origin funnel smoke is deferred until
  a supported provider hosting/network path is selected and can be tested.

Parking predicate: network-topology changes and active-process recovery are
parked unless they block this inactive-unit recovery path.

Parked hardening: none.

## Verification

- `pytest tests/test_atlas_api_healthcheck.py -q`
- `systemd-analyze verify config/atlas-api-healthcheck.service config/atlas-api-healthcheck.timer`
- `python scripts/sync_pr_plan.py plans/PR-EOM-API-Liveness-Contract.md --check`
- `bash scripts/push_pr.sh <body-file> -u origin HEAD` (runs the mechanical
  local review bundle exactly once before push).

## Estimated diff size

| File | LOC |
|---|---:|
| `config/atlas-api-healthcheck.service` | 18 |
| `config/atlas-api-healthcheck.timer` | 10 |
| `plans/PR-EOM-API-Liveness-Contract.md` | 243 |
| `scripts/atlas_api_healthcheck.py` | 415 |
| `scripts/install_atlas_api_healthcheck.py` | 397 |
| `tests/test_atlas_api_healthcheck.py` | 739 |
| **Total** | **1822** |
