# PR-Content-Ops-Support-Ticket-Date-Window-Diagnostics

## Why this slice exists

Support-ticket uploads currently enter dated-window mode only when every included
row has a parseable date. That conservative rule is correct because date-window
copy and report annualization must not be inferred from partial evidence, but the
parser only accepts ISO-style dates today. Common US SaaS CSV exports such as
`05/01/2026` therefore lose dated-window positioning even when every included row
is dated. The same ISO-ish parsing logic is also duplicated between the input
package and FAQ/report date-span path.

This slice keeps the all-rows-must-be-dated contract, accepts common US export
formats, and adds an explicit warning when rows carry a date-field signal but
missing or unparseable dates disable the window. The slice is 416 LOC, slightly
over the soft cap, because the parser is shared and the warning contract needs
focused negative, blank-date, mixed-date, and smoke-fixture coverage in the same
PR.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Add a shared support-ticket source-date parser for ISO-style dates and common
   US export dates.
2. Use the shared parser in support-ticket input packaging and ticket FAQ/report
   source-date spans.
3. Keep dated-window mode all-or-nothing for included rows.
4. Emit a package warning when included rows carry a date-field signal but
   missing or unparseable dates disable dated-window mode.
5. Leave UI surfacing, live DB/app verification, and international date guessing
   out of scope.

### Files touched

- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/support_ticket_dates.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Content-Ops-Support-Ticket-Date-Window-Diagnostics.md`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_smoke_content_ops_gate_a_live_quality.py`

### Review Contract

- Acceptance criteria:
  - [ ] ISO dates/datetimes and common US export dates keep dated-window mode
        when every included support-ticket row has one.
  - [ ] Natural-language dates and mixed missing-date rows still disable
        dated-window mode.
  - [ ] Disabled dated-window mode emits a warning with included count, dated
        count, missing/unparseable count, and example source IDs when the upload
        has a date-field signal.
  - [ ] Intentionally undated uploads keep the uploaded-ticket fallback without
        adding date-window warning noise.
  - [ ] Ticket FAQ/report source-date spans accept the same US export dates.
  - [ ] Existing prompt/report behavior remains conservative when the window is
        disabled.
- Affected surfaces: extracted support-ticket package, ticket FAQ date-span
  helper, focused tests, and extracted manifest ownership.
- Risk areas: backcompat, false date-window claims, parser ambiguity.
- Reviewer rules triggered: R1, R2, R5, R10, R13, R14.

## Mechanism

A new owned helper module exposes `parse_support_ticket_source_date`. It keeps
the existing accepted inputs (`date`, `datetime`, ISO date strings, ISO datetimes,
and `Z` datetimes) and adds explicit US date formats: `M/D/YYYY`, `MM/DD/YYYY`,
`M/D/YY`, `MM/DD/YY`, plus the same month/day/year order with `-` separators.
The helper deliberately does not parse natural-language dates or infer non-US
locale order.

`support_ticket_input_package` uses the helper to compute per-row date
diagnostics. If every included row has a parseable date, it preserves the
existing dated-window fields. If the upload includes a date-field signal but any
included row is blank or unparseable, it keeps the existing uploaded-ticket
fallback and appends a `support_ticket_date_window_disabled` warning. Rows from
intentionally undated uploads still use the fallback without extra warning
noise. `ticket_faq_markdown` uses the same parser for item date spans so report
windows and package windows do not drift.

## Intentional

- **US export formats only.** Slash/dash month-day-year exports are common for
  the expected operator CSVs. Day-month-year inference is deferred because it can
  silently invert dates.
- **All-or-nothing remains.** A partial date set still disables dated-window
  copy and annualized report math.
- **Warning, not blocker.** Rows remain included; the warning explains why the
  package cannot make calendar-window claims.
- **Shared helper module.** This avoids importing the full input-package module
  from the FAQ renderer and keeps date parsing deterministic and package-owned.

## Deferred

- UI/product surfacing beyond existing package warnings.
- International date parsing or operator-configured locale.
- Live app/DB verification; this slice is pure package behavior.
- Parked hardening: none.

## Verification

- Command passed: python -m py_compile extracted_content_pipeline/support_ticket_dates.py extracted_content_pipeline/support_ticket_input_package.py extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py tests/test_smoke_content_ops_gate_a_live_quality.py.
- Command passed: pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py -- 279 passed.
- Command passed: pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py tests/test_smoke_content_ops_support_ticket_package.py tests/test_smoke_content_ops_gate_a_live_quality.py tests/test_atlas_content_ops_input_provider.py tests/test_support_ticket_provider_landing_blog_execute.py tests/test_extracted_content_ops_live_execute_harness.py -- 370 passed, 1 warning.
- Command passed: bash scripts/validate_extracted_content_pipeline.sh -- outside sandbox after sandbox startup failed with bwrap loopback RTM_NEWADDR.
- Command passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- outside sandbox after the same sandbox startup failure.
- Command passed: python scripts/audit_extracted_standalone.py --fail-on-debt -- outside sandbox after the same sandbox startup failure.
- Command passed: bash scripts/check_ascii_python.sh -- outside sandbox after the same sandbox startup failure.
- Command passed: git diff --check.
- Command completed: bash scripts/run_extracted_pipeline_checks.sh -- 4009 passed, 10 skipped, 1 warning; one unrelated order-dependent failure remains in tests/test_send_content_ops_deflection_report_deliveries.py::test_validate_fails_closed_before_pool_for_missing_destination. The same test passes by itself on this branch and on detached origin/main at f44086c52, and this PR has no diff in that deflection-delivery surface.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/support_ticket_dates.py` | 50 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 92 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 18 |
| `plans/PR-Content-Ops-Support-Ticket-Date-Window-Diagnostics.md` | 127 |
| `tests/test_extracted_support_ticket_input_package.py` | 99 |
| `tests/test_extracted_ticket_faq_markdown.py` | 16 |
| `tests/test_smoke_content_ops_gate_a_live_quality.py` | 18 |
| **Total** | **423** |
