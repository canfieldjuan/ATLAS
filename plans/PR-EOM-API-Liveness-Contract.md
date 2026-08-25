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
  2. An existing maintenance lock prevents a start attempt, and the supported
     maintenance-entry command serializes marker creation plus service stop
     against recovery; settled by the two maintenance tests in
     `tests/test_atlas_api_healthcheck.py`.
  3. A start that fails, or a start followed by a failed probe, remains a
     visible down result rather than a false recovery; settled by the two
     failed-recovery tests in `tests/test_atlas_api_healthcheck.py`.
  4. An active but failed lead-intake probe does not restart the process;
     settled by
     `tests/test_atlas_api_healthcheck.py::test_active_unhealthy_service_alerts_without_restart`.
  5. The systemd template runs the installed copy outside the runtime worktree
   and reads notification settings from a user-local environment file;
   settled by `tests/test_atlas_api_healthcheck.py::test_service_template_uses_installed_script_and_private_environment`.
  6. The installer copies the reviewed monitor and both unit templates, proves
     the installed health service before enabling the timer, restores prior
     files/timer state on failure, and can later verify the deployment without
     writing; settled by
   `tests/test_atlas_api_healthcheck.py::test_installer_deploys_source_and_invokes_enabled_timer_path`.
- Reachability proof: On deployment, the timer invokes the installed monitor;
  `python scripts/install_atlas_api_healthcheck.py --install` copies the source,
  reloads user systemd, invokes the installed health service once, and enables
  the timer only after that proof succeeds. That service makes the real OPTIONS request to
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
  Separately, notification transitions need a crash-safe outbox rather than an
  in-memory decision, and failed `systemctl --user start` details must remain
  observable.
- Revised root cause: source-managed monitor files alone cannot change a
  user-systemd deployment, and the alert state machine acknowledges a
  notification before it knows that delivery succeeded.
- Revised required change surface: add a narrowly scoped
  `scripts/install_atlas_api_healthcheck.py` installer/checker that copies the
  monitor and templates outside the runtime worktree, preserves or safely
  migrates the private local notification topic without logging it, reloads and
  proves the installed health service, then enables the user timer. Persist each
  transition in an atomic, fsynced outbox before delivery and acknowledge only
  after success; carry bounded sanitized start-failure detail into the
  observation; and test installation, rollback, crash-point retry, and diagnostics.
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

### Contract revision: closed systemd state and open-execution model

- New evidence: `systemctl is-active` collapses every non-active result into one
  boolean even though recovery is safe only for a loaded unit whose
  `ActiveState` is exactly `inactive`. The installer likewise treats a missing
  old healthcheck unit as a stop failure, while `--check` compares disk bytes
  without proving that systemd loaded them.
- Revised root cause: monitor, installer, and verifier do not share one explicit
  systemd state model, and the contract describes sampled schedules rather than
  the property that must survive every admitted interleaving.
- Revised required change surface: query and parse `LoadState` plus
  `ActiveState` once through the monitor-owned state model; recover only
  `loaded/inactive`; surface absent, transitional, failed, incomplete, and query
  error states without a start. Reuse that model to admit a clean install with
  no old unit and to quiesce a loaded old unit. Require `NeedDaemonReload=no`
  for both installed units before `--check` succeeds.
- Revised explicit non-scope: do not add a supervisor, database, dependency,
  schema, deployment write, active-but-unhealthy restart, or broader systemd
  policy. The private topic, timer cadence, probe, and public CLI stay unchanged.
  Out-of-band edits to the monitor state file and underlying filesystem or
  storage corruption are also excluded; admitted state transitions are the
  atomic, schema-validated writes made by this monitor.
- Revised verification plan: exercise loaded/active, loaded/inactive,
  not-found, transitional, query-failure, incomplete-response, clean-install,
  unsupported-load-state, and daemon-reload-needed boundaries in the focused
  test file. GitHub runs the full unit gate.

#### Admitted actors and serialization

1. The user-systemd timer or a manual healthcheck invocation runs one monitor
   cycle. The oneshot service and `state.lock` serialize monitor cycles.
2. The supported maintenance command holds the same lock while publishing the
   marker and stopping `atlas-api.service`, so maintenance and recovery have a
   total order.
3. The installer first stops the timer, queries the old healthcheck unit, and
   waits for a loaded old oneshot to stop before replacing files. A missing old
   unit is the admitted clean-host state. Direct manual monitor launches during
   installation are excluded; installation is a single-operator maintenance
   action.
4. The systemd user manager owns unit load/active state. The monitor and
   installer act only on successfully parsed `LoadState`/`ActiveState` values;
   query failures and non-admitted states fail closed.
5. The remote ntfy service is an external at-least-once delivery consumer. It
   cannot participate in the local atomic state write.

#### Crash and cancellation boundaries

- Before recovery-intent persistence: no start is issued.
- After intent persistence but before/during start or re-probe: the next cycle
  retains the intent and reconciles the explicit systemd state.
- After outbox persistence but before notification: the next cycle retries.
- After successful notification but before acknowledgement persistence: a
  duplicate notification is admitted; silent loss is not.
- After the installer stops the prior timer or replaces any installation file,
  an operator `KeyboardInterrupt` runs the same rollback as an installation
  failure and is then re-raised. `SIGKILL` remains outside the rollback boundary;
  atomic file replacement still selects a complete prior or next file.
- If state cannot be read, decoded, locked, or atomically written, the cycle
  fails before issuing a new recovery side effect. A SIGKILL cannot interrupt an
  atomic file replacement, although it can select either the prior or next
  complete state.

#### Property invariant

For every admitted interleaving: the monitor never starts an unloaded,
transitional, failed, query-unknown, or maintenance-protected provider; every
issued recovery start is preceded by durable intent; every derived alert is
either durably queued or acknowledged after delivery; and installation is
reported verified only when the provider proof is non-down and source bytes,
private configuration, timer state, and systemd's loaded definitions agree.
Notification delivery is at-least-once, not exactly-once.

#### Closed-surface alternatives

- `Restart=always` alone is rejected because an operator stop is intentionally
  not restarted and it cannot encode the maintenance-marker boundary.
- SQLite is rejected for this single-host, serialized record because it adds a
  schema/migration surface while still being unable to atomically couple a
  remote ntfy response with local acknowledgement. The mode-0600 atomic state
  file plus `flock` is the smaller closed local surface; its admitted duplicate
  boundary is explicit above.

### Contract revision: deployed boundary outcomes

- New evidence: the systemd service accepts recovery exit `2` but rejects exit
  `4`, even though exit `4` means the same recovery/observation is durably
  recorded with only remote notification delivery pending. Separately,
  `http.client.HTTPException` is outside both HTTP catch sets, and numeric
  healthcheck environment defaults are converted before maintenance action
  selection.
- Revised root cause: the monitor classifies internal recovery and persistence
  states, but its systemd, HTTP, and CLI boundaries do not preserve those
  classifications. A queued alert is mistaken for failed recovery, malformed
  HTTP escapes instead of becoming an existing failure outcome, and unrelated
  healthcheck configuration can disable the supported maintenance boundary.
- Revised required change surface: admit exit `4` as a successful systemd
  oneshot completion while retaining its journal/outbox evidence; convert
  `HTTPException` at probe and notification boundaries into the existing down
  and undelivered outcomes; select maintenance mode before converting
  healthcheck-only numeric settings; cover all three boundaries in the focused
  suite.
- Revised explicit non-scope: do not change exit-code meanings, outbox or retry
  semantics, timer cadence, notification destination, installer transaction
  behavior, funnel/CRM APIs, schemas, dependencies, or live deployment.
- Revised verification plan: focused tests for queued-delivery service status,
  malformed HTTP at both boundaries, invalid numeric environment on both
  maintenance actions and the healthcheck path; syntax, systemd verification,
  focused maturity, plan sync, and diff audit. GitHub runs the full unit gate.

### Contract revision: outcome precedence and installer cancellation

- New evidence: notification delivery failure currently returns exit `4` for
  both a healthy/recovered provider and a provider that remains down. Because
  the systemd unit accepts exit `4`, the installer's service proof can therefore
  succeed while the provider is unavailable. Separately, the installer mutates
  timer and file state inside a transaction that catches `OSError` and
  `RuntimeError`, but an operator `KeyboardInterrupt` bypasses rollback.
- Revised root cause: the monitor collapses the provider observation and remote
  alert-delivery result into one scalar without failure precedence, while the
  installer transaction omits an admitted operator-cancellation exception from
  its rollback boundary.
- Revised required change surface: retain exit `3` whenever the current provider
  outcome is down even if its alert remains queued; use exit `4` only when the
  provider outcome is non-down and alert delivery remains pending; run the
  existing installer rollback for `KeyboardInterrupt` and then re-raise the
  cancellation; cover both outcome axes and cancellation after mutation in the
  focused test file.
- Revised explicit non-scope: do not change the systemd success-status set,
  alert queue or retry semantics, timer cadence, notification destinations,
  public CLI, dependencies, schemas, funnel/CRM behavior, or live deployment.
  Non-finite recovery-interval validation and nonzero desktop `notify-send`
  diagnostics are valid hardening but remain parked because neither blocks the
  observed provider recovery proof. Do not add or change a workflow: the
  existing every-PR Unit Gate escalates this config diff to the full unit suite.
- Revised verification plan: focused regressions for down-plus-undelivered,
  recovered-plus-undelivered, and installer `KeyboardInterrupt` rollback;
  focused pytest and syntax checks; systemd unit verification; selector proof
  that the PR runs the full Unit Gate; plan sync and diff audit. GitHub remains
  the source of truth for the full Unit Gate.

### Boundary-change enumeration

The recovery decision is a system state boundary, not an open-input guard.

- Boundary path/seam: `run_healthcheck()` deciding whether to issue
  `systemctl --user start`.
- Replaced-path behaviors: inactive service previously notified only; it now
  recovers only if maintenance is absent. Active/healthy remains no-op;
  active/unhealthy remains alert-only.
- Guard-relevant fields: unit load state, unit active state, maintenance-lock
  path, state-query result, start result, and post-start CORS probe result.
- Caller x input shape: timer invocation x loaded-active/healthy; timer x
  loaded-inactive/no lock; timer x loaded-inactive/lock; timer x not-found;
  timer x transitional; timer x query failure; timer x loaded-inactive/start
  failure; timer x loaded-active/failed probe.

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
- Side-effect ordering: supported maintenance entry and recovery share one
  process lock; issue a start only after inactive evidence; prove installed
  service behavior before timer enrollment.

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
  persisted notification/state record resets safely; recognized pending events
  remain queued while each run still derives state from the current observation,
  so arbitrary stored values cannot choose or suppress a notification path.

### Files touched

- `config/atlas-api-healthcheck.service`
- `config/atlas-api-healthcheck.timer`
- `plans/PR-EOM-API-Liveness-Contract.md`
- `scripts/atlas_api_healthcheck.py`
- `scripts/install_atlas_api_healthcheck.py`
- `tests/test_atlas_api_healthcheck.py`

## Mechanism

The monitor is a standalone Python program installed under `~/.local/bin`,
not loaded from the Atlas runtime worktree. Its supported maintenance command
creates the marker and stops the service under the same lock used by recovery.
If the service is inactive without that marker, it calls the existing unit and runs the same
lead-intake OPTIONS probe that the current monitor uses. It records and notifies
only after the final outcome is known. The templates keep the timer independent
of `atlas-api.service` and load the private notification topic from a local
environment file rather than version control. The companion installer copies
the reviewed source and templates, migrates an existing private topic without
printing it, proves the installed service before enabling the timer, and rolls
back prior files/timer state on failure; later `--check` is read-only.

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
- Rejecting non-finite recovery intervals is parked as configuration hardening;
  ordinary numeric and negative-value validation remains unchanged, and this
  does not alter the provider-result precedence required for installer proof.
- Reporting a nonzero desktop `notify-send` return code is parked as secondary
  observability hardening; desktop notification remains best-effort and remote
  ntfy delivery plus provider outcome remain the operational contract.

Parking predicate: network-topology changes and active-process recovery are
parked unless they block this inactive-unit recovery path.

Parked hardening: non-finite recovery-interval validation and nonzero desktop
notification exit diagnostics, as listed above.

## Verification

- `pytest tests/test_atlas_api_healthcheck.py -q`
- `systemd-analyze verify config/atlas-api-healthcheck.service config/atlas-api-healthcheck.timer`
- `python scripts/sync_pr_plan.py plans/PR-EOM-API-Liveness-Contract.md --check`
- `bash scripts/push_pr.sh <body-file> -u origin HEAD` (runs the mechanical
  local review bundle exactly once before push).

## Estimated diff size

| File | LOC |
|---|---:|
| `config/atlas-api-healthcheck.service` | 19 |
| `config/atlas-api-healthcheck.timer` | 10 |
| `plans/PR-EOM-API-Liveness-Contract.md` | 393 |
| `scripts/atlas_api_healthcheck.py` | 649 |
| `scripts/install_atlas_api_healthcheck.py` | 482 |
| `tests/test_atlas_api_healthcheck.py` | 1420 |
| **Total** | **2973** |
