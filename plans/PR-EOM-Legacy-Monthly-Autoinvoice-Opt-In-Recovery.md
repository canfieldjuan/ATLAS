# PR-EOM-Legacy-Monthly-Autoinvoice-Opt-In-Recovery

## Why this slice exists

H-21 in [#2363](https://github.com/canfieldjuan/ATLAS/issues/2363) closes a
deployment-default gap after the Billing & Payments approval flow landed. The
legacy monthly task remains registered by
`atlas_brain/autonomous/scheduler.py:558-567`; when admitted it creates
invoices, writes PDFs, marks services invoiced, and can send mail in
`atlas_brain/autonomous/tasks/monthly_invoice_generation.py:321-435`. The new
commercial candidate preview intentionally left that scheduler unchanged.

Read-only deployment probing on 2026-08-19 confirmed the active
`atlas-api.service` explicitly sets `ATLAS_INVOICING_AUTO_INVOICE_ENABLED=false`
and review mode true, so the live task returns at its existing early guard
before provider loading. Code defaults still set both legacy automatic-write
controls true in `atlas_brain/config.py:2481-2482`, so a fresh or
misconfigured future deployment could re-admit the legacy path without an
explicit operator choice.

### Problem-derived contract

- Root cause: the legacy task has a safe early exit, but the flags governing
  entry to its invoice/PDF/mail path default to enabled rather than requiring
  explicit deployment opt-in.
- Correct fix must touch/change: change only the two legacy automatic-write
  defaults in `InvoicingConfig`, clarify their opt-in descriptions, and add
  isolated default, explicit-true, and pre-provider skip proof.
- Must not change: scheduler registration, explicit-true legacy behavior,
  current production environment values, database/migrations, financial
  history, commercial candidate/run/approval routes, Gmail recovery, Manual
  Square behavior, or tracker/Website consumers.

## Scope (this PR)

Ownership lane: eom/billing-payments-security-hardening
Slice phase: Production hardening
Max files: 4

1. Default `auto_invoice_enabled` and `auto_invoice_send_email` to false and
   describe both as legacy explicit opt-ins.
2. Add a no-DB contract suite proving absent configuration is fail-safe,
   explicit true remains valid, and the real disabled task returns before any
   provider import.
3. Make the three existing direct legacy-task integration tests explicitly set
   their enabled/auto-invoice settings so they retain their original enabled
   path coverage rather than depending on global defaults.
4. Preserve the cron task and all explicit-true behavior; this PR changes no
   production configuration or financial record.

### Review Contract

- Acceptance criteria:
  - With relevant `ATLAS_INVOICING_*` variables absent and env-file loading
    disabled, `InvoicingConfig` defaults both changed automatic-write flags to
    false, as the focused test proves.
  - Explicit true remains accepted for both flags, proving the legacy task is
    opt-in rather than removed.
  - Calling the real `monthly_invoice_generation.run` with automatic invoicing
    disabled returns `{"_skip_synthesis": "Auto-invoicing disabled"}` before
    calendar, CRM, invoice, PDF, email, service-invoiced, or notification
    collaborators can be imported; failure sentinels prove the import boundary.
  - Existing direct billing-month/contact-filter task tests explicitly enable
    their legacy operating path before `run()`, so their old enabled behavior
    remains tested without relying on a default.
  - The scheduler registration and task code remain unchanged; a diff check
    proves this config/test-only slice did not alter them.
  - The active deployment is not mutated. Its current explicit false value
    keeps behavior unchanged while the default protects a future absent setting.
- Reachability proof: `InvoicingConfig` is the live API settings model and the
  focused async test calls the real legacy task entrypoint, observing its skip
  result with provider imports guarded.
- Affected surfaces: `atlas_brain/config.py`, the unchanged legacy task direct
  caller, and focused configuration/task tests.
- Risk areas: accidental invoice/PDF/mail creation, explicit legacy opt-in
  compatibility, settings precedence, deployment rollback.
- Reviewer rules triggered: R1, R2, R3, R4, R11, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: `InvoicingConfig` default ->
  `settings.invoicing.auto_invoice_enabled` -> the task's existing early guard
  changes from implicit admission to explicit opt-in. The email flag is only
  read after admission.
- Replaced-path behaviors: absent configuration previously admitted legacy
  generation (and could admit email if review mode were false); it now uses the
  existing no-write skip. Explicit true retains the old enabled path.
- Guard-relevant fields: exactly `auto_invoice_enabled` and
  `auto_invoice_send_email`; unchanged `auto_invoice_review_mode=true` remains
  an additional send guard.
- Caller x input shape: absent -> both false; explicit boolean true -> accepted
  legacy opt-in; false -> existing pre-provider skip. No HTTP request or row
  controls either setting.

### Deployed-config probing

- Deployed/default config values: the active user `atlas-api.service` runs the
  #2434 billing-approval runtime with explicit
  `ATLAS_INVOICING_AUTO_INVOICE_ENABLED=false` and review mode true; this PR
  writes no production configuration.
- Explicit value probe: the focused test constructs `InvoicingConfig` with
  both flags true and verifies accepted opt-in.
- Absent value probe: it removes both process variables and disables env-file
  loading, then verifies false defaults.
- Default-session/default-context probe: the real task test sets false and
  fails if any writeful provider import occurs; no live service, financial row,
  Gmail account, or credential is used.
- Side-effect ordering: the task returns at its existing flag check before it
  imports calendar/CRM/repository providers. The config-only change adds no
  writer, migration, or restart.

### Closure declaration

- The changed automatic-write control set is **CLOSED** and **DERIVED** from
  the two `InvoicingConfig` fields read by the legacy task admission and send
  decisions. Unknown settings are ignored by the settings model and cannot
  enable the task; absent changed settings resolve false, the safe side because
  an unintended invoice or email costs more than explicit operator opt-in.

### Files touched

- `atlas_brain/config.py`
- `plans/PR-EOM-Legacy-Monthly-Autoinvoice-Opt-In-Recovery.md`
- `tests/test_legacy_monthly_autoinvoice_opt_in.py`
- `tests/test_monthly_invoice_generation.py`

## Mechanism

The task already makes `auto_invoice_enabled` its first operational decision.
This slice leaves that branch and scheduler untouched but reverses the unsafe
configuration default. An absent setting now returns through the established
skip before any financial/delivery collaborator is imported. An intentionally
operated legacy workflow still sets its flag(s) true; automatic email needs its
own opt-in and the existing review-mode condition to allow sending.

## Intentional

- Keep the task and cron registration. Scheduler retirement and reconciliation
  need a distinct financial-operating decision.
- Change the email default too: review mode is another default safeguard, but a
  later false review-mode setting must not inherit send consent silently.
- Do not modify the active `.env`, restart the API, or exercise production data.
  The live explicit false setting already has the desired behavior.

## Deferred

- H-21 follow-up: retire the scheduler and reconcile any remaining manual/MCP
  dependency only after a separately authorized financial plan. Tracking: #2363.
- H-22: move legacy billing-period validation ahead of provider construction so
  malformed task metadata is deterministic when calendar/CRM configuration is
  unavailable. This separate task-ordering defect was discovered by H-21's
  explicit-opt-in probe and is tracked in #2363.
- Legacy real-database test-harness cleanup is outside this config-boundary
  slice; this PR adds deterministic no-DB proof only.

Parking predicate: unrelated financial workflow, scheduler/data migration, UI,
and harness changes are parked unless required to prove these two defaults.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_legacy_monthly_autoinvoice_opt_in.py -q` -- 3
  passed (no database, provider, Gmail, or financial record).
- `python -m pytest tests/test_monthly_invoice_generation.py -k 'resolver or
  per_hour_line_items or notification_lines' -q` -- 13 passed, 30 deselected
  (pure existing helper coverage).
- `python -m pytest tests/test_monthly_invoice_generation.py --collect-only
  -q` -- 43 test nodes collected, including each newly explicit direct task
  invocation.
- `python -m ruff check atlas_brain/config.py
  tests/test_legacy_monthly_autoinvoice_opt_in.py` -- passed.
- `python -m py_compile atlas_brain/config.py
  tests/test_legacy_monthly_autoinvoice_opt_in.py
  tests/test_monthly_invoice_generation.py` -- passed.
- `git diff --check` -- passed.
- `python scripts/sync_pr_plan.py
  plans/PR-EOM-Legacy-Monthly-Autoinvoice-Opt-In-Recovery.md --check` and
  `python scripts/audit_plan_doc.py
  plans/PR-EOM-Legacy-Monthly-Autoinvoice-Opt-In-Recovery.md` -- passed.
- Skipped: real-database bodies in `tests/test_monthly_invoice_generation.py`.
  They can create then void invoice rows; no explicitly isolated test database
  was verified for this run, and H-21's deterministic no-DB proof covers the
  changed default boundary.
- Skipped: Black check for `atlas_brain/config.py`. The installed formatter
  proposes thousands of unrelated existing rewrites in that legacy file; the
  source diff was manually limited to the two intended fields and passed Ruff,
  compilation, and whitespace checks.
- Pending before push: managed local PR review with the final PR body.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/config.py` | 16 |
| `plans/PR-EOM-Legacy-Monthly-Autoinvoice-Opt-In-Recovery.md` | 199 |
| `tests/test_legacy_monthly_autoinvoice_opt_in.py` | 78 |
| `tests/test_monthly_invoice_generation.py` | 18 |
| **Total** | **311** |
