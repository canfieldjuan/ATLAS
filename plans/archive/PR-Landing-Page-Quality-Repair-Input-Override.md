# PR-Landing-Page-Quality-Repair-Input-Override

## Why this slice exists

PR #717 added the landing-page quality repair loop, and PR #720 threaded the
default repair-attempt count through Content Ops planning and execution. The
remaining operator gap is control: Content Ops runs still cannot tune that
count per request.

Operators need a small validated input for cost control and debugging. Setting
the value to 0 should disable the repair pass for a run; setting a positive
integer should allow more repair attempts without changing the service default
for every caller.

## Scope (this PR)

1. Add a landing-page-only landing_page_quality_repair_attempts input.
2. Allow non-negative integers, including 0.
3. Reject booleans, floats, negative values, values above 10, and non-integer
   strings.
4. Keep all other landing-page generation inputs and defaults unchanged.
5. Add focused plan and execution coverage for the override.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Quality-Repair-Input-Override.md` | Plan doc for this operator override slice. |
| `extracted_content_pipeline/generation_plan.py` | Read and validate the landing-page repair-attempt override from request inputs. |
| `tests/test_extracted_content_generation_plan.py` | Cover default, explicit zero, positive string, and invalid override behavior. |
| `tests/test_extracted_content_ops_execution.py` | Prove the override reaches the landing-page service call. |

## Mechanism

_landing_page_config_for_request stops discarding the request. It reads
landing_page_quality_repair_attempts from request inputs and returns a
LandingPageGenerationConfig with the default repair-attempt count unless the
input is present.

A new non-negative integer input helper mirrors the existing positive-integer
helper but accepts 0, because disabling repair is a valid operator choice. The
landing-page override also caps the value at 10 so a typo cannot queue a runaway
repair loop. The generated plan already emits quality_repair_attempts, and the
executor already dispatches that value, so the override only needs to change
the planned config.

## Intentional

- No UI form field in this slice; the New Run screen already supports raw
  inputs JSON.
- No change to the default repair count.
- No generic quality_repair_attempts input shared by other assets.
- No cost-estimate change; repair attempts are a quality fallback, while the
  current preview budget is still parse-retry based.
- No repair-loop changes for reports, sales briefs, or blog posts.

## Deferred

- PR-Landing-Page-Quality-Repair-UI-Control can add a first-class UI control if
  operators use this frequently enough.
- PR-Generated-Asset-Repair-Telemetry can expose intermediate repair failures
  in operator-facing diagnostics.
- PR-Content-Ops-Quality-Cost-Model can revisit budget estimation if repair
  attempts should become part of preflight cost limits.

## Verification

- pytest tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py -q
  -> passed 93/93 tests.
- Python compile command over `extracted_content_pipeline/generation_plan.py`,
  `tests/test_extracted_content_generation_plan.py`, and
  `tests/test_extracted_content_ops_execution.py` -> passed 3/3 files.
- git diff --check -> passed with 0 whitespace errors.
- bash scripts/validate_extracted_content_pipeline.sh -> passed mapped-file
  and hard-import checks.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  -> passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -> passed with
  0 Atlas runtime import findings.
- bash scripts/check_ascii_python.sh -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Generation plan override | ~45 |
| Tests | ~75 |
| Total | ~230 |
