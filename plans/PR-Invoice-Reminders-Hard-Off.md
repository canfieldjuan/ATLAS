# PR-Invoice-Reminders-Hard-Off

## Why this slice exists

On 2026-08-03 the autonomous task `invoice_payment_reminders` emailed 17 unauthorised dunning messages to real EOM customers in one 10:00 run (Mid Illinois received four). It was stopped by setting `ATLAS_INVOICING_REMINDERS_ENABLED=false` in the host `.env`. #2270 scopes the duplicate/extra-reminder bugs and #2271 scopes the missing approval gate; **neither is fixed** — `run()` still loops `for inv in overdue:` (`invoice_payment_reminders.py:103`) and calls `email_provider.send` once per invoice (`:163`), and `InvoiceRepository.get_overdue` still returns one row per invoice with no per-customer grouping.

Operator decision 2026-08-08 (Juan): no autonomous payment reminders at all, indefinitely — not merely "off in this environment". This slice makes that decision a property of the code rather than of one hand-maintained file.

Audit of the current stop, verified against `main` @ `40bb24553` and the live ts.net DB:

- `InvoicingConfig.reminders_enabled` ships `default=True` (`config.py:2438`) — absent config means ON.
- The scheduler seed for the task carries no `enabled` key (`scheduler.py:545-556`), so `task_def.get("enabled", True)` (`scheduler.py:1135`) registers the `0 10 * * *` cron **enabled** on any fresh database.
- The task's metadata is `{builtin_handler, synthesis_skill}` with no `enabled_config_key`, so `enabled_config_managed` is False and `enabled_changed` (`scheduler.py:1075-1082`) is permanently False — the boot-time sync never reconciles this task's enabled flag in either direction.
- Live DB before this slice: `scheduled_tasks` row `enabled = t`, `last_run_at = 2026-08-08 10:00:00-05`. The cron was firing daily and returning at the config check.

So a single `.env` line was the only thing between the incident and its recurrence — and this deployment has already performed one runtime worktree cutover (#2254 S0, 2026-08-05) of exactly the kind that drops such a line.

### Problem-derived contract

- Root cause: the reminder send path has no code-level disable. Every layer that could hold it closed (config default, scheduler seed default, boot sync) defaults open or is inert, leaving an environment file as the sole control.
- Correct fix must touch/change: the task entrypoint (a guard that precedes the config read, the repository query, and every transport call), the config default (fail-closed), and the scheduler seed (fresh databases must not register the cron). Plus a regression proving a config re-enable cannot defeat the guard.
- Must not change: the reminder send logic, cadence maths (`_should_send_reminder`), `InvoiceRepository.get_overdue`, the invoicing MCP server, `invoice_overdue_check`, `monthly_invoice_generation`, or any other scheduled task. This slice does **not** fix #2270 and does **not** build #2271's approval gate; conflating them is what would make the disable negotiable.

## Scope (this PR)

Ownership lane: eom-invoicing/reminder-autopilot-off
Slice phase: production hardening

1. Add `_AUTOPILOT_DISABLED` to `atlas_brain/autonomous/tasks/invoice_payment_reminders.py`, checked as the first statement of `run()` — before the `settings` import, the `get_invoice_repo()` query, and any `email_provider.send` — returning `{"_skip_synthesis": _AUTOPILOT_DISABLED_REASON}`.
2. Flip `InvoicingConfig.reminders_enabled` from `default=True` to `default=False` so an absent or blank env value means OFF.
3. Add `"enabled": False` to the `invoice_payment_reminders` scheduler seed so a fresh database never registers the 10:00 cron enabled.
4. Add `tests/test_invoice_payment_reminders_disabled.py` (5 tests) pinning all three layers, including that `reminders_enabled=True` still produces no send.
5. Opt the two existing send-shape tests past the guard via `monkeypatch` so they keep documenting the behaviour #2271 must revive.

### Review Contract

- Acceptance criteria:
  1. `run()` returns without reading config, querying invoices, or constructing an email provider while the guard is set — settled by `tests/test_invoice_payment_reminders_disabled.py::test_run_returns_before_reading_config_or_invoices`, which patches `get_invoice_repo` and `get_email_provider` to raise `AssertionError`, so the test fails loudly if the guard is ever moved below them.
  2. Setting `reminders_enabled=True` produces no send — settled by `::test_enabling_config_does_not_defeat_the_guard` (same raise-on-call patches, with `settings.invoicing.enabled` and `.reminders_enabled` both True).
  3. The config default is fail-closed — settled by `::test_config_default_is_fail_closed` asserting `InvoicingConfig.model_fields["reminders_enabled"].default is False`.
  4. A fresh database seeds the task disabled — settled by `::test_scheduler_seeds_the_task_disabled` asserting `TaskScheduler._DEFAULT_TASKS` has exactly one `invoice_payment_reminders` entry with `.get("enabled") is False` (explicitly False, not merely absent — absence is the defect).
  5. The kill constant itself is present — settled by `::test_autopilot_disabled_flag_is_set`.
  6. No regression on the existing invoicing suite — settled by the base-vs-branch comparison in Verification below.
- Reachability proof: real entrypoint is the scheduler's builtin dispatch of `invoice_payment_reminders` (`autonomous/tasks/__init__.py:30` → `run`) on the `0 10 * * *` cron. Observable effect: the run logs the disable reason and returns `_skip_synthesis`; no `sent_emails` row, no `contact_interactions` `invoice` row, no outbound message. The live `scheduled_tasks.enabled` flag was additionally set false on the ts.net host so the dispatch does not occur at all.
- Affected surfaces: `atlas_brain/autonomous/tasks/invoice_payment_reminders.py`, `atlas_brain/config.py` (`InvoicingConfig.reminders_enabled` only), `atlas_brain/autonomous/scheduler.py` (`_DEFAULT_TASKS` entry only), `tests/test_monthly_invoice_generation.py` (two tests), new test module, `.github/workflows/atlas_invoicing_checks.yml` (path filters + one added step).
- Risk areas: guard placement relative to side effects; the config default flip reaching an unintended consumer of `reminders_enabled`; the seed edit perturbing other `_DEFAULT_TASKS` entries or the seeding loop; the two monkeypatched tests silently disabling the guard for the rest of a session.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R11, R12.
  - **R6 (error handling/observability):** the guard logs at INFO with the full reason string before returning, so a 10:00 run is diagnosable from the journal without reading code. No error is swallowed — the guard is not an exception path. No secondary-write behaviour changes.
  - **R8 (concurrency/idempotency):** strictly reduces duplicate-send exposure. The guard returns before `repo.update_reminder`, `email_provider.send` and `crm.log_interaction`, so no retry, duplicate dispatch, or out-of-order execution can produce a send or a partial bookkeeping write. The un-fixed per-invoice fan-out that motivated #2270 is now unreachable rather than merely gated.
  - **R11 (dependencies/config):** no new dependency. The one config change is a default flip on an existing typed `ATLAS_INVOICING_` field — no raw `os.environ` read is introduced. Production effect is intentional and stated: absent config inverts from ON to OFF, which is the fix.
  - **R12 (deployment safety/CI enrollment):** the disable path is the rollback path — clearing `_AUTOPILOT_DISABLED` restores prior behaviour exactly, with no migration or data change to unwind. The new test module and the three changed source files are enrolled in `.github/workflows/atlas_invoicing_checks.yml` (both the `pull_request` and `push` path filters) with a dedicated `Run payment-reminder autopilot-disabled guard tests` step, so the guard suite is actually exercised by CI rather than only locally.

### Boundary-change enumeration

This diff changes an admission boundary (whether an outbound customer-email path may execute).

- Boundary path/seam: `invoice_payment_reminders.run()` entry — the new `_AUTOPILOT_DISABLED` check is the outermost admission gate, ahead of the pre-existing `settings.invoicing.enabled` and `cfg.reminders_enabled` checks which are retained beneath it.
- Replaced-path behaviors: nothing is replaced. The prior gates are unchanged and unreachable while the new gate holds; they resume their exact previous behaviour if the constant is ever cleared. The send body, cadence maths, PDF attachment, CRM logging and result shape are untouched.
- Guard-relevant fields: `_AUTOPILOT_DISABLED` (module-level `bool`). It reads no input, no env, no database, and no request — it is deliberately not configurable, which is the point of the slice. `_AUTOPILOT_DISABLED_REASON` is the operator-facing string returned and logged.
- Caller x input shape: the only caller is the scheduler's builtin dispatch, which passes a `ScheduledTask`. The guard precedes every use of that argument, so all input shapes (including `None`, as the tests pass) take the same path.

### Deployed-config probing

- Deployed/default config values: `reminders_enabled` code default was `True`, now `False`. The ts.net host `.env:321` carries `ATLAS_INVOICING_REMINDERS_ENABLED=false`; env prefix is `ATLAS_INVOICING_` (`config.py` `InvoicingConfig.model_config`).
- Explicit value probe: `reminders_enabled=True` explicitly set → no send (`::test_enabling_config_does_not_defeat_the_guard`). `false` → no send (unchanged behaviour, and now also the default).
- Absent value probe: env var removed entirely → `False` by the new default, so the pre-existing `cfg.reminders_enabled` gate also refuses. Before this slice the same absence meant ON; that inversion is the fix.
- Default-session/default-context probe: a fresh database with no `scheduled_tasks` rows seeds this task `enabled=False` (`::test_scheduler_seeds_the_task_disabled`), so the cron is never registered rather than registered-and-refused.
- Side-effect ordering: the guard is the first statement in `run()`. `sent_emails`, `contact_interactions` and `repo.update_reminder` all live below it and are unreachable. The two monkeypatched tests set the constant via `monkeypatch.setattr`, which restores it at teardown, so the guard cannot leak False into another test.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/autonomous/scheduler.py`
- `atlas_brain/autonomous/tasks/invoice_payment_reminders.py`
- `atlas_brain/config.py`
- `plans/PR-Invoice-Reminders-Hard-Off.md`
- `tests/test_invoice_payment_reminders_disabled.py`
- `tests/test_monthly_invoice_generation.py`

## Mechanism

Three independent layers, each sufficient alone; the slice ships all three because the incident showed single-layer controls fail silently.

The **task guard** is authoritative: a module constant read before any effectful call, so it cannot be defeated by config (`ATLAS_INVOICING_REMINDERS_ENABLED=true`), by database state (`UPDATE scheduled_tasks SET enabled = true`), or by a deploy whose `.env` is incomplete. Changing it requires a code change, review, and merge — which is the intended cost.

The **config default** inverts the failure mode: absent config previously meant ON, which is what made a lost `.env` line dangerous; it now means OFF.

The **seed default** stops the cron being registered at all on a fresh database. It is deliberately scoped to fresh seeding: because this task carries no `enabled_config_key`, `enabled_config_managed` is False and `scheduler.py:1075-1082` will not reconcile an already-seeded row from the code side. That asymmetry is documented in the seed comment so a future reader does not mistake the seed edit for a live control. The live row was set `enabled=false` out of band; the same property means the boot sync will not flip it back.

## Intentional

- The pre-existing `settings.invoicing.enabled` and `cfg.reminders_enabled` checks are retained beneath the new guard rather than deleted, so clearing the constant restores exactly the prior gating rather than a laxer one.
- `_AUTOPILOT_DISABLED` is a plain module constant, not a config field or env var. Making it configurable would reintroduce the exact defect this slice removes.
- The two existing send-shape tests are monkeypatched past the guard rather than deleted or skipped: they still hold `#2271`'s revived behaviour (PDF attachment, text-only fallback) to a standard, and deleting them would lose that contract.
- The reason string names #2271 and the required sign-off so the log line and the API response are self-explaining at 10:00 on some future morning.

## Deferred

- #2270 (per-customer consolidation, no reminders on settled/"last" invoices, transactional bookkeeping) — not fixed here.
- #2271 (owner approval gate in the `list_pending_drafts` / `approve_and_send` shape) — not built here. This slice is its precondition, not its substitute.
- Enrolling the task in the `enabled_config_key` machinery so the boot sync reconciles its enabled flag. Not done deliberately: config-managing the flag would make the DB row follow config, which is the coupling this slice is removing. Revisit only as part of #2271.
- Auditing the other autonomous tasks for the same seed-defaults-open shape. Named as a follow-up, not swept here.

## Verification

Mechanical verification. Result: pass.

- New guard suite: `python -m pytest tests/test_invoice_payment_reminders_disabled.py -v` → `5 passed`. All five acceptance criteria above are covered.
- Regression, branch vs base, same command on both: `python -m pytest tests/test_monthly_invoice_generation.py -q` → **branch `4 failed, 39 passed`; clean `main` @ `40bb24553` `4 failed, 39 passed`** — identical counts and identical test names (`test_billing_month_override`, `test_billing_month_invalid_format`, `test_payment_reminder_attaches_pdf`, `test_payment_reminder_falls_back_when_pdf_fails`). Zero new failures.
- Those four are a pre-existing local-environment failure, red on `main` before this diff: all four assert against `{'_skip_synthesis': 'Invoicing disabled'}` because `settings.invoicing.enabled` is False in this dev environment (the dev-vs-CI env drift tracked in #2324). Confirmed the two reminder tests fail on the branch for that same pre-existing reason and **not** for the new guard — the failure payload after the monkeypatch reads `'Invoicing disabled'`, not the autopilot reason string.
- `tests/test_invoice_repository.py` → passed on both.
- Live: `UPDATE scheduled_tasks SET enabled = false WHERE name='invoice_payment_reminders'` applied on the ts.net host; re-read confirms `enabled=false`. Metadata carries no `enabled_config_key`, so `scheduler.py:1075-1082` cannot flip it back on boot.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 12 |
| `atlas_brain/autonomous/scheduler.py` | 12 |
| `atlas_brain/autonomous/tasks/invoice_payment_reminders.py` | 56 |
| `atlas_brain/config.py` | 9 |
| `plans/PR-Invoice-Reminders-Hard-Off.md` | 131 |
| `tests/test_invoice_payment_reminders_disabled.py` | 88 |
| `tests/test_monthly_invoice_generation.py` | 10 |
| **Total** | **318** |

## Cold diff reconstruction

- `invoice_payment_reminders.py`: module docstring gains an AUTOPILOT DISABLED line; after the logger, a commented block records the incident, the three default-open layers, and the ordered re-enable preconditions, then defines `_AUTOPILOT_DISABLED = True` and `_AUTOPILOT_DISABLED_REASON`. In `run()`, the docstring gains a paragraph on guard ordering and the first statement becomes `if _AUTOPILOT_DISABLED:` → log + `return {"_skip_synthesis": _AUTOPILOT_DISABLED_REASON}`, placed above the existing `from ...config import settings`.
- `config.py`: `reminders_enabled` default `True` → `False`, description gains "(default OFF; see #2271)", preceded by a comment explaining the fail-closed inversion.
- `scheduler.py`: the `invoice_payment_reminders` entry in `_DEFAULT_TASKS` gains `"enabled": False`, a description suffix "(DISABLED pending #2271 approval gate)", and a leading comment noting the fresh-seed-only scope and the `enabled_config_key` asymmetry.
- `tests/test_invoice_payment_reminders_disabled.py`: new module, 5 tests as enumerated in the Review Contract.
- `tests/test_monthly_invoice_generation.py`: `monkeypatch.setattr(task_mod, "_AUTOPILOT_DISABLED", False)` added after the `task_mod` import in `test_payment_reminder_attaches_pdf` and `test_payment_reminder_falls_back_when_pdf_fails`, each with a comment pointing at the guard suite.
- `.github/workflows/atlas_invoicing_checks.yml`: adds `atlas_brain/autonomous/scheduler.py`, `atlas_brain/autonomous/tasks/invoice_payment_reminders.py` and `tests/test_invoice_payment_reminders_disabled.py` to **both** the `pull_request` and `push` path filters (`atlas_brain/config.py` was already listed), plus a `Run payment-reminder autopilot-disabled guard tests` step running the new module. R12 enrollment for the new test.
