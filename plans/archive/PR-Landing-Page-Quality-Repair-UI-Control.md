# PR-Landing-Page-Quality-Repair-UI-Control

## Why this slice exists

PR #721 added a backend/operator input for landing-page quality repair attempts:
landing_page_quality_repair_attempts. The New Run screen can already use that
through raw inputs JSON, but operators should not have to remember the exact
key or valid range for a common landing-page debugging and cost-control knob.

This slice adds a first-class UI control while keeping raw inputs JSON as the
source of truth.

## Scope (this PR)

1. Show a landing-page-only quality repair attempt control when the selected
   outputs include landing_page.
2. Let operators choose backend default, 0, or 1 through 10.
3. Write the selected value into the existing inputs JSON object.
4. Delete the JSON key when the operator chooses backend default.
5. Surface invalid existing JSON/control values without overwriting raw JSON.
6. Reject invalid landing-page repair values before preview, plan, or execute.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Quality-Repair-UI-Control.md` | Plan doc for this UI slice. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Add a landing-page-only repair attempt select synchronized with inputs JSON. |

## Mechanism

The New Run screen already treats the raw inputs JSON textarea as the request
payload source. This PR keeps that contract. The new select parses the current
inputs JSON, reads landing_page_quality_repair_attempts, and renders one of:
backend default, a valid integer from 0 to 10, or an invalid-value marker.

Changing the select parses the current JSON object, updates or deletes only the
landing_page_quality_repair_attempts key, pretty-prints the JSON back into the
textarea, and marks preview/plan/execution state stale. If the current inputs
JSON is invalid or not an object, the select is disabled so the UI cannot
discard operator edits.

Preview, plan, and execute also validate any raw
landing_page_quality_repair_attempts value when landing_page is selected. That
keeps the select warning and request behavior aligned: values outside 0 to 10
do not proceed through the run path.

## Intentional

- No backend changes. PR #721 owns validation and execution behavior.
- No separate React state for the repair count; inputs JSON remains source of
  truth.
- No generic repair-attempt control for reports, blog posts, sales briefs, or
  FAQ output.
- No cost-estimate changes.
- No component tests because the Atlas Intel UI does not currently include a
  component test runner.

## Deferred

- PR-Generated-Asset-Repair-Telemetry can expose intermediate repair failures
  in operator-facing diagnostics.
- PR-Content-Ops-Quality-Cost-Model can revisit budget estimation if repair
  attempts should affect preflight costs.

## Verification

- atlas-intel-ui npm run lint
- atlas-intel-ui npm run build
- git diff --check

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| New Run UI | ~205 |
| Total | ~280 |
