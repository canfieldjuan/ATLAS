# PR-Invoice-Reminders-Hard-Off

## Why this slice exists

On 2026-08-03 the autonomous task `invoice_payment_reminders` emailed 17 unauthorised dunning messages to real EOM customers in one 10:00 run (Mid Illinois received four). It was stopped by setting `ATLAS_INVOICING_REMINDERS_ENABLED=false` in the host `.env`. #2270 scopes the duplicate/extra-reminder bugs and #2271 scopes the missing approval gate; **neither is fixed** — `run()` still loops `for inv in overdue:` (`invoice_payment_reminders.py:103`) and calls `email_provider.send` once per invoice (`:163`), and `InvoiceRepository.get_overdue` still returns one row per invoice with no per-customer grouping.

Operator decision 2026-08-08 (Juan): no autonomous payment reminders at all, indefinitely — not merely "off in this environment". This slice makes that decision a property of the code rather than of one hand-maintained file.

### Why this slice exceeds the 400-LOC budget and is indivisible

The diff is over the soft cap. It is one safety invariant — a customer-email
path held closed at three layers — plus the proof that it is closed. Production
code is a small minority of it; the remainder is the plan doc the
plan-admission gate itself mandates, the guard test module, the send-shape test
edits and the CI enrollment.

Splitting it lands either the guard with no permitted-side proof — which is
exactly the P1 the automated reviewer raised on round 1 — or the proof with the
10:00 cron still live in the interval. Neither half is shippable alone, and the
incident this closes already happened once, to real customers.

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
Max files: 8

1. Add `_AUTOPILOT_DISABLED` to `atlas_brain/autonomous/tasks/invoice_payment_reminders.py`, checked as the first statement of `run()` — before the `settings` import, the `get_invoice_repo()` query, and any `email_provider.send` — returning `{"_skip_synthesis": _AUTOPILOT_DISABLED_REASON}`.
2. Flip `InvoicingConfig.reminders_enabled` from `default=True` to `default=False` so an absent or blank env value means OFF.
3. Add `"enabled": False` to the `invoice_payment_reminders` scheduler seed so a fresh database never registers the 10:00 cron enabled.
4. Add `tests/test_invoice_payment_reminders_disabled.py` (7 tests) pinning all three layers, including that `reminders_enabled=True` still produces no send.
5. Opt the two existing send-shape tests past the guard via `monkeypatch` so they keep documenting the behaviour #2271 must revive.
6. Coerce a blank `ATLAS_INVOICING_REMINDERS_ENABLED` to `False` via a `mode="before"` field validator. Pydantic's bool parser rejects `""`, so an env template rendering the key empty raised `ValidationError` at import and took the app down — and made the field's fail-closed claim false for exactly the shape a half-configured deployment produces.
7. Prove the guard's **permitted side** and enroll it: the two revived send-shape tests open both config gates, are removed from `tests/unit_gate_baseline.txt`, and run in a blocking CI step.
8. Extract the send body to `_send_due_reminders(task)` so `run()` is guard-plus-delegate. The send-shape tests call the unguarded function directly instead of monkeypatching `_AUTOPILOT_DISABLED` back to False — patching the kill switch was a first-party mock of the guard itself and tripped the maturity-sweep ratchet on two lanes (ATLAS #1877). The guard tests then need no mocks at all.

### Review Contract

- Acceptance criteria:
  1. The guard precedes **every** effectful boundary — settled by two tests, and
     demonstrated by mutation rather than asserted. `::test_run_touches_no_config_repository_or_transport`
     empties `atlas_brain.config`, `...storage.repositories.invoice`,
     `...services.email_provider` and `...services.invoice_pdf` from `sys.modules`,
     runs `run()`, and requires all four to still be absent — `run` imports each
     collaborator inside the function body, so absence proves the line was never
     reached. `::test_guard_is_the_first_statement_of_run` parses the source and
     requires the first statement of `run` (after the docstring) to be an
     `if _AUTOPILOT_DISABLED` whose body returns, which covers boundaries nobody
     has thought of yet. Neither patches a first-party symbol, so the maturity
     ratchet stays green.
     **Mutation evidence:** moving the guard below the config read and the
     repository import fails both tests, while
     `::test_run_returns_at_the_autopilot_gate_not_a_later_one` still passes —
     which is precisely the gap round 3 identified: a guard that fires late
     returns the same reason string.
  2. Setting `reminders_enabled=True` produces no send — settled by `::test_config_enabled_does_not_defeat_the_guard`, which opens **both** config gates on the settings object and still requires `_AUTOPILOT_DISABLED_REASON`, a reason only the guard can produce. This is the criterion the mock-free rewrite initially dropped: criterion 1 alone runs with invoicing disabled ambiently and so cannot distinguish the guard blocking from the master gate blocking.
  3. The config default is fail-closed — settled by `::test_config_default_is_fail_closed` asserting `InvoicingConfig.model_fields["reminders_enabled"].default is False`.
  4. A fresh database seeds the task disabled — settled by `::test_scheduler_seeds_the_task_disabled` asserting `TaskScheduler._DEFAULT_TASKS` has exactly one `invoice_payment_reminders` entry with `.get("enabled") is False` (explicitly False, not merely absent — absence is the defect).
  5. The kill constant itself is present — settled by `::test_autopilot_disabled_flag_is_set`.
  6. No regression on the existing invoicing suite — settled by the base-vs-branch comparison in Verification below.
  7. A blank or whitespace-only `ATLAS_INVOICING_REMINDERS_ENABLED` yields `False` rather than `ValidationError`, and an explicit `true` still parses — settled by `::test_blank_env_value_means_disabled_not_a_crash`, which probes `""`, `"   "` and `"true"`.
  9. **Ratchet:** the `atlas-brain-autonomous` and `atlas-brain-b2c-core-risk` maturity-sweep lanes pass with **no baseline edit** — settled by running both with the workflow's own arguments and sensitive globs, exit 0 each.
  8. **Permitted side:** the underlying send path works when its config gates are open — settled by `tests/test_monthly_invoice_generation.py -k payment_reminder` (3 tests), which drive `_send_due_reminders` directly and assert the PDF attachment and text-only fallback. Ordering against the guard is settled separately by the exact-`_skip_synthesis`-string assertion in `::test_run_returns_at_the_autopilot_gate_not_a_later_one`, since each gate returns a distinct reason.
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
- Explicit value probe: `reminders_enabled=True` explicitly set → no send (`::test_config_enabled_does_not_defeat_the_guard`). `false` → no send (unchanged behaviour, and now also the default). A non-boolean value such as `ture` still raises rather than being coerced (`::test_garbage_env_value_still_raises`) — the blank-value forgiveness must not become a catch-all that hides a typo.
- Absent value probe: env var removed entirely → `False` by the new default, so the pre-existing `cfg.reminders_enabled` gate also refuses. Before this slice the same absence meant ON; that inversion is the fix.
- Blank/whitespace value probe: `ATLAS_INVOICING_REMINDERS_ENABLED=` and `="   "` → `False`. Before the validator both raised `ValidationError` and crashed startup; reproduced directly before fixing.
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
- `tests/unit_gate_baseline.txt`

## Mechanism

Three independent layers, each sufficient alone; the slice ships all three because the incident showed single-layer controls fail silently.

The **task guard** is authoritative: a module constant read before any effectful call, so it cannot be defeated by config (`ATLAS_INVOICING_REMINDERS_ENABLED=true`), by database state (`UPDATE scheduled_tasks SET enabled = true`), or by a deploy whose `.env` is incomplete. Changing it requires a code change, review, and merge — which is the intended cost.

The **config default** inverts the failure mode: absent config previously meant ON, which is what made a lost `.env` line dangerous; it now means OFF.

The **seed default** stops the cron being registered at all on a fresh database. It is deliberately scoped to fresh seeding: because this task carries no `enabled_config_key`, `enabled_config_managed` is False and `scheduler.py:1075-1082` will not reconcile an already-seeded row from the code side. That asymmetry is documented in the seed comment so a future reader does not mistake the seed edit for a live control. The live row was set `enabled=false` out of band; the same property means the boot sync will not flip it back.

## Intentional

- The pre-existing `settings.invoicing.enabled` and `cfg.reminders_enabled` checks are retained beneath the new guard rather than deleted, so clearing the constant restores exactly the prior gating rather than a laxer one.
- `_AUTOPILOT_DISABLED` is a plain module constant, not a config field or env var. Making it configurable would reintroduce the exact defect this slice removes.
- The two existing send-shape tests call the extracted `_send_due_reminders` directly rather than being deleted or skipped: they still hold `#2271`'s revived behaviour (PDF attachment, text-only fallback) to a standard. They deliberately do **not** patch `_AUTOPILOT_DISABLED` — mocking the kill switch to test around it is what the maturity ratchet rejected.
- The reason string names #2271 and the required sign-off so the log line and the API response are self-explaining at 10:00 on some future morning.

## Deferred

**Parking predicate.** A finding is parked here iff it is adjacent work that
does not weaken the guarantee this slice ships — that no autonomous reminder can
be sent. Anything that could let a reminder out, or that leaves the guard
unproven, blocks and is fixed in-slice; that is why all three automated-review
rounds were fixed rather than deferred.

**Parked hardening: none.** Nothing was found and set aside during
implementation. The entries below are pre-existing follow-up issues, not
hardening carried out of this slice, and no `HARDENING.md` entries are created.

- #2270 (per-customer consolidation, no reminders on settled/"last" invoices, transactional bookkeeping) — not fixed here.
- #2271 (owner approval gate in the `list_pending_drafts` / `approve_and_send` shape) — not built here. This slice is its precondition, not its substitute.
- Enrolling the task in the `enabled_config_key` machinery so the boot sync reconciles its enabled flag. Not done deliberately: config-managing the flag would make the DB row follow config, which is the coupling this slice is removing. Revisit only as part of #2271.
- Auditing the other autonomous tasks for the same seed-defaults-open shape. Named as a follow-up, not swept here.

## Verification

Mechanical verification. Result: pass.

- New guard suite: `python -m pytest tests/test_invoice_payment_reminders_disabled.py -q` -> `8 passed`, covering acceptance criteria 1-5, 7 and 9. Every node named in this contract is asserted to exist by the run; the earlier revision cited two names that had been removed in the mock-free rewrite and failed collection.
- Permitted side: `python -m pytest tests/test_monthly_invoice_generation.py -k payment_reminder -q` -> `3 passed, 40 deselected`. These two send-shape tests were RED before this round (accepted failures in the baseline) because they opened only the new guard while `invoicing.enabled` and `reminders_enabled` both default `False`; they now open all three gates and reach the fake transport.
- Whole file: `python -m pytest tests/test_monthly_invoice_generation.py -q` -> `2 failed, 41 passed`, versus clean `main` @ `40bb24553` -> `4 failed, 39 passed`. Strictly better than base: the two `test_payment_reminder_*` failures are fixed; the two remaining `test_billing_month_*` failures are pre-existing on `main`, untouched by this diff, and stay in the baseline.
- Blank-env reproduction, before the fix: `ATLAS_INVOICING_REMINDERS_ENABLED='' python -c "InvoicingConfig(_env_file=None)"` raised `pydantic_core._pydantic_core.ValidationError: Input should be a valid boolean ... input_value=''`. After the validator, `""`, `"   "` -> `False` and `"true"` -> `True`.
- `tests/test_invoice_repository.py` -> passed on both branch and base.
- Live: `UPDATE scheduled_tasks SET enabled = false WHERE name='invoice_payment_reminders'` applied on the ts.net host; re-read confirms `enabled=false`. Metadata carries no `enabled_config_key`, so `scheduler.py:1075-1082` cannot flip it back on boot.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 23 |
| `atlas_brain/autonomous/scheduler.py` | 12 |
| `atlas_brain/autonomous/tasks/invoice_payment_reminders.py` | 68 |
| `atlas_brain/config.py` | 25 |
| `plans/PR-Invoice-Reminders-Hard-Off.md` | 184 |
| `tests/test_invoice_payment_reminders_disabled.py` | 213 |
| `tests/test_monthly_invoice_generation.py` | 26 |
| `tests/unit_gate_baseline.txt` | 2 |
| **Total** | **553** |

## Cold diff reconstruction

- `invoice_payment_reminders.py`: module docstring gains an AUTOPILOT DISABLED line; after the logger, a commented block records the incident, the three default-open layers, and the ordered re-enable preconditions, then defines `_AUTOPILOT_DISABLED = True` and `_AUTOPILOT_DISABLED_REASON`. In `run()`, the docstring gains a paragraph on guard ordering and the first statement becomes `if _AUTOPILOT_DISABLED:` → log + `return {"_skip_synthesis": _AUTOPILOT_DISABLED_REASON}`. The rest of the original body is **extracted into `_send_due_reminders(task)`**, which `run` tail-calls when the guard is clear — a production seam the send-shape tests exercise directly so they need not patch the constant.
- `config.py`: `reminders_enabled` default `True` → `False`, description gains "(default OFF; see #2271)", preceded by a comment explaining the fail-closed inversion.
- `scheduler.py`: the `invoice_payment_reminders` entry in `_DEFAULT_TASKS` gains `"enabled": False`, a description suffix "(DISABLED pending #2271 approval gate)", and a leading comment noting the fresh-seed-only scope and the `enabled_config_key` asymmetry.
- `tests/test_invoice_payment_reminders_disabled.py`: new module, **9 tests** as enumerated in the Review Contract — the kill-constant assertion, the reason-string ordering check, the config-cannot-defeat-the-guard check, the four-boundary `sys.modules` touch probe, the AST first-statement proof, the fail-closed config default, blank-value coercion, its garbage-value counterpart, and the disabled scheduler seed.
- `tests/test_monthly_invoice_generation.py`: the two send-shape tests open both config gates via `monkeypatch.setattr(settings.invoicing, ...)` and call the **extracted, unguarded `task_mod._send_due_reminders`** directly. They do **not** patch `_AUTOPILOT_DISABLED`; doing so was a first-party mock of the kill switch itself and tripped the maturity ratchet on two lanes.
- `.github/workflows/atlas_invoicing_checks.yml`: adds `atlas_brain/autonomous/scheduler.py`, `atlas_brain/autonomous/tasks/invoice_payment_reminders.py` and `tests/test_invoice_payment_reminders_disabled.py` to **both** the `pull_request` and `push` path filters (`atlas_brain/config.py` was already listed), plus two steps — `Run payment-reminder autopilot-disabled guard tests` (the new module) and `Run payment-reminder permitted-side tests` (`-k payment_reminder` on the existing file). R12 enrollment for both sides of the boundary.
- `atlas_brain/config.py`: adds `_blank_reminders_value_means_disabled`, a `@field_validator("reminders_enabled", mode="before")` coercing blank/whitespace strings to `False`.
- `tests/unit_gate_baseline.txt`: removes `test_payment_reminder_attaches_pdf` and `test_payment_reminder_falls_back_when_pdf_fails` (166 → 164 lines). Both genuinely pass now; the gate's exact-node shrink proof re-runs them.
