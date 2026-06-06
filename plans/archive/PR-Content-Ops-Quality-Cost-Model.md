# PR-Content-Ops-Quality-Cost-Model

## Why this slice exists

PR #721 added a capped landing-page repair-attempt override, and PR #724 added
a first-class New Run control for it. The remaining cost gap is in preview
planning: Content Ops estimates still count parse retries but do not count the
landing-page quality-repair loop.

That can make a run look cheaper than the worst-case number of LLM calls the
operator has selected. Cost preview and budget gating should reflect the same
repair-attempt setting that execution will use.

## Scope (this PR)

1. Add quality-repair attempt metadata to the output catalog.
2. Include landing-page quality-repair attempts in retry-adjusted unit costs.
3. Let preview cost estimation read landing_page_quality_repair_attempts from
   request inputs.
4. Ignore landing-page repair attempts when quality gates are disabled.
5. Keep the existing 0 to 10 validation bound aligned with generation planning.
6. Update preview tests and control-surface docs for the new estimate behavior.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Quality-Cost-Model.md` | Plan doc for this cost-model slice. |
| `extracted_content_pipeline/control_surfaces.py` | Include quality-repair attempts in preview cost estimation. |
| `extracted_content_pipeline/api/control_surfaces.py` | Expose default quality-repair attempts in the catalog payload. |
| `extracted_content_pipeline/docs/control_surface_preview_api.md` | Document that landing-page estimates include quality repairs. |
| `tests/test_extracted_content_control_surfaces.py` | Cover default, override, disabled-gate, and invalid repair cost behavior. |
| `tests/test_extracted_content_control_surface_api.py` | Cover the catalog's default quality-repair metadata. |
| `atlas-intel-ui/src/api/contentOps.ts` | Add the catalog wire field for default quality-repair attempts. |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | Add the domain field for default quality-repair attempts. |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | Map the catalog field into the domain model. |
| `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json` | Keep the local catalog fixture aligned with the backend contract. |

## Mechanism

The existing preview estimate multiplies unit cost by
default_parse_retry_attempts + 1. Landing pages now have an additional
quality-repair loop that can call the LLM again after parsed output fails the
quality gate.

This slice keeps cost math deterministic by deriving a per-output attempt
multiplier:

- parse multiplier: default_parse_retry_attempts + 1
- landing-page repair multiplier: quality_repair_attempts + 1 when quality
  gates are enabled
- all other outputs: repair multiplier 1

The landing-page repair count defaults to the service default and can be
overridden by landing_page_quality_repair_attempts in request inputs. Invalid
override values raise before preview returns a runnable plan, matching the
planning and execution validation path.

## Intentional

- No change to actual execution behavior.
- No change to landing-page repair defaults.
- No UI changes in this slice; PR #724 already added the operator control.
- No repair cost model for reports, sales briefs, FAQ, or email campaigns.
- No live provider pricing integration; these remain placeholder unit-cost
  estimates.

## Deferred

- Provider-specific token/cost estimates based on observed usage.
- Operator-facing telemetry for failed intermediate repair candidates.
- A shared repair-attempt model if other generated assets gain repair loops.

## Verification

- pytest tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_control_surface_api.py -q
  -> passed 99/99 tests.
- Python compile command over changed control-surface Python files and focused
  tests -> passed 4/4 files.
- atlas-intel-ui npm run lint -> passed.
- atlas-intel-ui npm run build -> passed.
- git diff --check -> passed.
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
| Plan | ~90 |
| Control-surface cost model | ~95 |
| API/docs | ~35 |
| Tests | ~105 |
| Frontend contract fixture/types | ~35 |
| Total | ~350 |
