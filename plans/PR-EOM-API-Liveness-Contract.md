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

1. Replace the untracked alert-only health-check logic with a canonical
   installed-outside-the-runtime monitor that auto-recovers only an unexpected
   inactive `atlas-api.service`.
2. Add a maintenance-lock contract so intentional maintenance is never
   automatically undone.
3. Add executable regression coverage and source-managed unit templates that
   make the installation/configuration contract reviewable.

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
- Reachability proof: On deployment, the timer invokes the installed monitor;
  an unexpected inactive unit transitions to `systemctl --user start`, then an
  OPTIONS request to the real `/api/v1/leads/intake` route returns 200/204.
  Local tests exercise the same command/probe seam without stopping the live
  provider.
- Affected surfaces: source-managed standalone monitor, user-systemd
  templates, notification/maintenance configuration, and their focused tests.
- Risk areas: unexpected clean stops, deliberate maintenance, failed start,
  failed re-probe, notification configuration absence, and keeping the monitor
  independent of the application worktree.
- Reviewer rules triggered: R1, R2, R6, R11, R12, R14.

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
  the user Atlas config directory; notification topic comes only from the
  user-local environment file.
- Explicit value probe: focused tests inject service, lock, and probe values.
- Absent value probe: absent lock permits recovery; absent notification topic
  must not be committed or silently replaced with a public/default topic.
- Default-session/default-context probe: the systemd template loads a missing
  environment file optionally and the monitor uses its safe defaults.
- Side-effect ordering: decide maintenance before any start; issue a start only
  after inactive evidence; record success only after the post-start probe.

### Files touched

- `config/atlas-api-healthcheck.service`
- `config/atlas-api-healthcheck.timer`
- `plans/PR-EOM-API-Liveness-Contract.md`
- `scripts/atlas_api_healthcheck.py`
- `tests/test_atlas_api_healthcheck.py`

## Mechanism

The monitor is a standalone Python program installed under `~/.local/bin`,
not loaded from the Atlas runtime worktree. It checks the maintenance lock
before observing or starting the unit. If the service is inactive without that
lock, it calls the existing user-systemd unit once and then runs the same
lead-intake OPTIONS probe that the current monitor uses. It records and notifies
only after the final outcome is known. The templates keep the timer independent
of `atlas-api.service` and load the private notification topic from a local
environment file rather than version control.

## Intentional

- Do not rely on `Restart=always` as the root fix: an explicit systemd stop can
  still suppress restart behavior, while the independent timer can recover the
  observed inactive state.
- Do not auto-restart an active-but-unhealthy provider. That needs a distinct
  diagnosis and would widen this slice from the observed inactive-unit failure.
- Do not commit the ntfy topic or read it from application configuration; it is
  a private notification credential and remains in a local environment file.

## Deferred

- Render-to-Atlas reachability remains a separate deployment-topology slice.
  The current `ATLAS_FUNNEL_BASE_URL` is not proven reachable from Render by
  this local liveness monitor, and this PR must not change that network boundary.
- A deployment-time authenticated, Render-origin funnel smoke is deferred until
  a supported provider hosting/network path is selected and can be tested.

Parking predicate: network-topology changes, active-process recovery, and
notification-delivery hardening are parked unless they block this inactive-unit
recovery path.

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
| `config/atlas-api-healthcheck.service` | 17 |
| `config/atlas-api-healthcheck.timer` | 10 |
| `plans/PR-EOM-API-Liveness-Contract.md` | 169 |
| `scripts/atlas_api_healthcheck.py` | 273 |
| `tests/test_atlas_api_healthcheck.py` | 231 |
| **Total** | **700** |
