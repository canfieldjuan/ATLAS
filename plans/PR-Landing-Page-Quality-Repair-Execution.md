# PR-Landing-Page-Quality-Repair-Execution

## Why this slice exists

PR #717 added quality repair attempts to
`LandingPageGenerationService.generate(...)`, but the Content Ops plan and
execution dispatcher still do not carry that value through. A direct caller can
use the repair loop, while an operator running landing pages through Content Ops
only sees the service construction default.

That is an integration gap at the source of the execution path: the generated
plan should expose the landing-page repair default, and the executor should
pass the planned value into the service call just like it already does for
temperature, max tokens, parse retry, and quality gates.

## Scope (this PR)

1. Add quality repair attempts to the landing-page generation plan step
   config.
2. Thread quality repair attempts from the landing-page step config into
   `LandingPageGenerationService.generate(...)`.
3. Update focused plan and execution tests so this integration point cannot
   silently regress.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Quality-Repair-Execution.md` | Plan doc for this execution wiring slice. |
| `extracted_content_pipeline/generation_plan.py` | Emit the landing-page quality repair attempt default in step config. |
| `extracted_content_pipeline/content_ops_execution.py` | Pass the planned repair attempt count into the landing-page service call. |
| `tests/test_extracted_content_generation_plan.py` | Assert landing-page plan config includes the repair attempt count. |
| `tests/test_extracted_content_ops_execution.py` | Assert landing-page execution dispatch passes the repair attempt count. |

## Mechanism

`_landing_page_config_for_request(...)` already returns a
`LandingPageGenerationConfig`, which now owns the repair-attempt default. The
landing-page branch in `_step_for_output(...)` includes that value in
`step.config`.

`_dispatch_landing_page(...)` reads the integer from `step.config` with the
same typed helper used for other numeric controls and passes it to
`service.generate(...)`.

## Intentional

- No new request input override in this slice. This only closes the default
  plan-to-execution wiring gap.
- No behavior change for direct `LandingPageGenerationService` callers.
- No quality-gate or prompt changes.
- No UI or API control-surface schema changes.
- No equivalent repair loop added to other asset generators.

## Deferred

- PR-Landing-Page-Quality-Repair-Input-Override can add a validated operator
  input if we want users to tune repair attempts per run.
- PR-Generated-Asset-Repair-Telemetry can expose intermediate repair failures
  in operator-facing diagnostics.
- PR-Other-Asset-Quality-Repair can evaluate whether reports, sales briefs, or
  blog posts should get the same repair-loop behavior.

## Verification

- `pytest tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py -q`
  -> passed 84/84 tests.
- Python compile command over `extracted_content_pipeline/generation_plan.py`,
  `extracted_content_pipeline/content_ops_execution.py`,
  `tests/test_extracted_content_generation_plan.py`, and
  `tests/test_extracted_content_ops_execution.py` -> passed 4/4 files.
- `git diff --check` -> passed with 0 whitespace errors.
- bash scripts/validate_extracted_content_pipeline.sh -> passed mapped-file
  and hard-import checks.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  -> passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -> passed with
  0 Atlas runtime import findings.
- bash scripts/check_ascii_python.sh -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~80 |
| Generation plan wiring | ~5 |
| Execution dispatcher wiring | ~5 |
| Tests | ~15 |
| Total | ~105 |
