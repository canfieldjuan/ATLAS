# PR-EOM-Legacy-Billing-Month-Admission

## Why this slice exists

H-22 in [#2363](https://github.com/canfieldjuan/ATLAS/issues/2363) found that
the legacy monthly auto-invoice task evaluates `billing_month` only after it
loads calendar, CRM, service-repository, and invoice-repository providers.
With the legacy writer explicitly enabled, malformed persisted task metadata
can therefore fail as unavailable provider configuration instead of returning
the documented deterministic invalid-period skip.  That ordering is unsafe for
an invoice/PDF/mail writer: invalid admission must finish before it can reach
any financial or delivery collaborator.

The predecessor hardening slice [#2439](https://github.com/canfieldjuan/ATLAS/pull/2439)
made the legacy writer an explicit deployment opt-in and preserved its existing
task registration.  Read-only deployment probing on 2026-08-19 confirmed the
active `atlas-api.service` explicitly sets
`ATLAS_INVOICING_AUTO_INVOICE_ENABLED=false`; this slice does not alter that
configuration or restart the service.  It hardens only the already-explicit
enabled legacy path.  H-23's isolated writer-integration harness remains a
separate concern because this task-ordering proof must not provision a database
or admit any writer.

This slice is intentionally over the 400-LOC soft target because its inseparable
proof centralizes the pre-provider sentinel, checks the grammar's acceptance
and rejection classes, and calls the real async task entrypoint under explicit
opt-in.  Splitting that proof would leave either a financial admission change
without no-write reachability evidence or duplicated sentinel logic.

### Problem-derived contract

- Root cause: `monthly_invoice_generation.run` constructs write-capable
  provider seams before proving that the open `task.metadata.billing_month`
  value is an exact, calendar-valid billing period.
- Correct fix must touch/change: place one exact billing-month admission parser
  after the existing feature-flag exits and before every provider import or
  construction; add no-provider regression tests that exercise the real task
  entrypoint and the parser's accepted grammar/rejected complement.
- Must not change: feature-flag ordering or messages; scheduler registration;
  valid explicit and implicit billing-period behavior; contact filters;
  invoice/PDF/Gmail behavior after a valid admission; configuration defaults;
  database/migrations; financial history; commercial candidate/run/approval
  routes; Manual Square behavior; or tracker/Website consumers.

## Scope (this PR)

Ownership lane: eom/billing-payments-security-hardening
Slice phase: Production hardening
Max files: 3

1. Validate a supplied legacy `billing_month` as exact ASCII `YYYY-MM` plus a
   real calendar year/month before importing or constructing any provider.
2. Keep missing `billing_month` on the established previous-calendar-month
   path, and retain both existing disabled-task exits before metadata parsing.
3. Prove malformed metadata returns the existing invalid-period skip without a
   provider import, while valid/missing metadata still reaches the first
   provider seam only when the legacy writer is deliberately enabled.
4. Add grammar-derived acceptance/rejection tests with no database, Gmail,
   calendar, CRM, invoice, PDF, or financial-record interaction.
5. Keep H-23's isolated legacy-writer integration harness deferred in #2363;
   do not add a scheduler, configuration, CI, migration, or runtime change.

### Review Contract

- Acceptance criteria:
  - The existing `Invoicing disabled` and `Auto-invoicing disabled` exits in
    `monthly_invoice_generation.run` remain the first two operational
    decisions; focused sentinel tests prove their no-provider behavior rather
    than exercising a writer.
  - When both legacy feature flags admit the task, an explicit
    `billing_month` is admitted only if it is a string with exact ASCII
    `YYYY-MM` structure, a year within `date.min.year` through
    `date.max.year`, and a month from 1 through 12.  The private parser is the
    single choke point; ambiguous, non-string, malformed, and calendar-invalid
    input returns the established
    `Invalid billing_month format: ... (expected YYYY-MM)` skip by default.
  - The malformed-input regression calls the real async task entrypoint with
    the legacy writer explicitly enabled, installs its import sentinel before
    a forced fresh task-module import, and proves no writeful provider can be
    imported before the deterministic skip is observed.
  - The parser test derives accepted values from the fixed-width ASCII grammar
    and calendar bounds, then perturbs structure, digit class, separators, and
    calendar range to prove the input class rather than only the reported
    example.
  - Missing `billing_month` and an exact valid override preserve their prior
    admission behavior: with flags explicitly enabled they proceed to the
    calendar-provider seam, where the sentinel stops the test before any real
    collaborator or write.
  - No production state is mutated.  The tests run with a synthetic task,
    patched in-process settings, import-level provider sentinels, and no
    provider implementation.
  - The existing invoicing workflow already names both the task and this
    focused contract file in its PR/push path and runs the file in a dedicated
    pytest step; this slice adds no duplicate workflow.
- Reachability proof: the scheduler continues to resolve
  `atlas_brain.autonomous.tasks.monthly_invoice_generation.run`; the focused
  test forces a fresh import of that real module, then awaits its real `run`
  function.  Provider imports are observable at Python's import boundary
  before provider construction, database access, or delivery can occur.
- Affected surfaces: the legacy task's billing-period admission boundary and
  its existing no-write opt-in contract test; the task is already enrolled in
  `.github/workflows/atlas_invoicing_checks.yml`.
- Risk areas: accidental invoice/PDF/mail access on malformed task metadata,
  backwards compatibility for valid scheduled runs, malformed persisted
  metadata diagnostics, retry behavior, and overly broad parser acceptance.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: `ScheduledTask.metadata["billing_month"]` ->
  `monthly_invoice_generation.run` -> billing-period parser -> provider
  imports/construction.  The parser moves ahead of, rather than around, the
  established enabled writer path.
- Replaced-path behaviors: a supplied nonempty value formerly reached provider
  construction and loosely split on `-`; it now either proves exact structural
  and calendar evidence then follows the unchanged provider path, or returns
  the existing invalid-format skip before a provider import.  A missing value
  still derives the prior calendar month.  An explicitly supplied empty string
  is malformed rather than silently treated as absent.
- Guard-relevant fields: only `task.metadata["billing_month"]`; its value is
  producer-supplied `Any`.  `contact_ids` is intentionally untouched and is
  read only after billing-period admission as before.
- Caller x input shape:
  - `invoicing.enabled=false` -> unchanged `Invoicing disabled` exit before
    metadata or provider work.
  - `auto_invoice_enabled=false` -> unchanged `Auto-invoicing disabled` exit
    before metadata or provider work.
  - Both flags true + missing/null `billing_month` -> unchanged prior-month
    derivation, then the existing provider path.
  - Both flags true + exact calendar-valid ASCII `YYYY-MM` -> unchanged
    explicit-period admission, then the existing provider path.
  - Both flags true + every other value shape -> existing invalid-format skip
    before provider import/construction.

### Deployed-config probing

- Deployed/default config values: read-only H-21 probing on 2026-08-19 found
  the active service explicitly has legacy automatic invoicing disabled and
  review mode true.  The source defaults changed by #2439 are false.  This PR
  changes neither setting, environment, scheduler, nor service unit.
- Explicit value probe: tests patch both admission flags true and pass an
  exact valid override or a malformed override to the real task entrypoint;
  sentinel behavior distinguishes provider admission from pre-provider skip.
- Absent value probe: a task whose metadata omits `billing_month` follows the
  established default previous-month branch until the calendar-provider
  sentinel, without initializing a real provider.
- Default-session/default-context probe: the existing disabled-task sentinel
  remains a synthetic task with patched settings and no provider implementation;
  no live service, financial row, Gmail account, or credential participates.
- Side-effect ordering: both feature-flag exits and the new period admission
  complete before the first local provider import.  The safe rejection has no
  invoice, PDF, email, database, enqueue, or external-provider side effect.

### Closure declaration

- `billing_month` values are **OPEN** because persisted task metadata stores a
  producer-supplied `Any`, not a finite enum.  Recognition is **DERIVED** at
  the one parser from the task contract's exact `YYYY-MM` wire grammar and
  Python `date` year domain plus Gregorian 1-through-12 month semantics; no
  literal denylist decides admission.  Every value without affirmative
  structural and calendar evidence takes the existing no-write invalid-format
  skip.  That default is safer and cheaper than initializing financial/delivery
  providers for an invalid task.

### Files touched

- `atlas_brain/autonomous/tasks/monthly_invoice_generation.py`
- `plans/PR-EOM-Legacy-Billing-Month-Admission.md`
- `tests/test_legacy_monthly_autoinvoice_opt_in.py`

## Mechanism

The task continues to honor its two feature-flag exits first.  Only an enabled
legacy task reads metadata.  A small private parser recognizes exactly seven
ASCII characters in `YYYY-MM` form and checks the `date` type's exact year
domain with the Gregorian month domain; it returns a year/month tuple only on
affirmative recognition without an exception path.
`run` treats every other supplied value as its existing `_skip_synthesis`
response, then imports providers only after the billing period is known valid.
The contract suite intercepts every legacy writeful provider import before a
fresh task-module import.  It proves malformed metadata never crosses that
boundary, and that valid/missing metadata still does when explicitly enabled;
the sentinel halts the latter before a provider can be constructed.

## Intentional

- Preserve the current response wording for malformed values so callers retain
  their deterministic skip contract; improve when that message is designed in
  a separate API/operations slice.
- Treat a present empty string as invalid, because `billing_month` is an
  override only when it offers affirmative period evidence.  Missing/null is
  the explicit default-period request.
- Derive the supported year range from `date.min.year` and `date.max.year`,
  then use the Gregorian month domain directly, so the parser closes the
  grammar/calendar class without a throw-and-catch admission path.
- Keep provider imports inside `run` and move them only after admission; do not
  refactor task registration or the financial writer while proving this narrow
  ordering invariant.
- Do not turn on or exercise legacy automatic invoicing in production.  The
  real enabled-path proof deliberately dies at its first import seam.

## Deferred

- H-23 in [#2363](https://github.com/canfieldjuan/ATLAS/issues/2363): add a
  separately reviewed legacy monthly writer integration harness with an
  explicitly provisioned isolated Postgres target, non-delivery provider seams,
  deterministic fixtures, cleanup/rollback proof, and CI gating.  This
  admission-only slice does not create that infrastructure.
- The two existing payment-reminder tests at
  `tests/test_monthly_invoice_generation.py:1017-1145` do not configure the
  reminder task's general invoicing gate and fail identically on `origin/main`
  with `Invoicing disabled`.  That independent reminder-delivery work is
  already tracked by [#2271](https://github.com/canfieldjuan/ATLAS/issues/2271);
  H-22 does not change its task, configuration, or tests.

Parking predicate: scheduler retirement, legacy writer product changes,
database/provider integration, Website/tracker work, and environment/runtime
changes remain parked unless required to prove this pre-provider admission
boundary.

Parked hardening: none.

## Verification

- Focused pytest for `tests/test_legacy_monthly_autoinvoice_opt_in.py` -- 26
  passed.  The suite
  uses only synthetic tasks, patched in-process settings, and provider-import
  sentinels; it never constructs a provider or financial record.
- Targeted pytest for `tests/test_monthly_invoice_generation.py` billing-month
  and contact-filter admission cases -- 3 passed, 40
  deselected.  Existing direct metadata admission probes remain no-write.
- Targeted Ruff lint for
  `atlas_brain/autonomous/tasks/monthly_invoice_generation.py` and
  `tests/test_legacy_monthly_autoinvoice_opt_in.py` -- passed.
- Python compilation for
  `atlas_brain/autonomous/tasks/monthly_invoice_generation.py` and
  `tests/test_legacy_monthly_autoinvoice_opt_in.py` -- passed.
- `ruff format --check tests/test_legacy_monthly_autoinvoice_opt_in.py` --
  passed.  The task file's formatter check is intentionally not a gate: its
  current baseline proposes unrelated legacy reformatting outside this slice;
  the changed lines pass Ruff, compilation, whitespace, and focused tests.
- Full `tests/test_monthly_invoice_generation.py -q` -- 41 passed, 2 failed at
  the unrelated payment-reminder tests.  A detached, unchanged `origin/main`
  worktree reproduced the exact 41/2 result and the same `Invoicing disabled`
  response; #2271 tracks the independent reminder behavior.
- `python scripts/check_guard_class_closure.py --base origin/main --strict` --
  passed (advisory: no guard-shaped change without a property test).
- The exact `atlas_brain/autonomous` maturity-sweep ratchet invocation used by
  CI, with `tests/maturity_sweep/baseline_atlas_brain_autonomous.json` and its
  invoicing-sensitive globs, passed with no new brittleness above baseline.
- Pending final pre-push: `python scripts/sync_pr_plan.py
  plans/PR-EOM-Legacy-Billing-Month-Admission.md --check`, `python
  scripts/audit_plan_doc.py plans/PR-EOM-Legacy-Billing-Month-Admission.md`,
  `git diff --check`, and managed local PR review with the final body and
  checked-out branch head.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/autonomous/tasks/monthly_invoice_generation.py` | 65 |
| `plans/PR-EOM-Legacy-Billing-Month-Admission.md` | 264 |
| `tests/test_legacy_monthly_autoinvoice_opt_in.py` | 236 |
| **Total** | **565** |
