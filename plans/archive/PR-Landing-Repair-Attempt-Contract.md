# PR-Landing-Repair-Attempt-Contract

## Why this slice exists

Landing-page quality repair attempts were validated in multiple layers with
separate max/default literals. Preview, planning, execution, and the Intel UI
could disagree when a cap changed because each layer owned part of the same
business rule.

This slice makes the backend rule canonical and exposes the same contract to
the UI through the control-surface catalog.

## Scope (this PR)

1. Add one backend contract module for the landing-page repair-attempt input
   key, default, min, max, and validator.
2. Route control-surface preview, generation planning, and landing-page runtime
   repair limits through that contract.
3. Expose the input contract in the control-surface catalog and map it through
   the Intel UI wire/domain layers.
4. Update the New Run UI to derive its repair-attempt range/default text from
   the catalog contract instead of a local max literal.
5. Add tests that pin backend defaults and catalog contract shape.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Repair-Attempt-Contract.md` | Plan doc for this validation-contract slice. |
| `extracted_content_pipeline/landing_page_repair_contract.py` | New canonical repair-attempt input contract and validator. |
| `extracted_content_pipeline/control_surfaces.py` | Uses the canonical contract for preview costs and catalog metadata. |
| `extracted_content_pipeline/generation_plan.py` | Uses the canonical validator when building landing-page config. |
| `extracted_content_pipeline/landing_page_generation.py` | Uses the canonical default and runtime validator. |
| `extracted_content_pipeline/api/control_surfaces.py` | Emits input contracts in the control-surface catalog. |
| `atlas-intel-ui/src/api/contentOps.ts` | Adds input-contract wire types. |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | Adds input-contract domain types. |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | Maps input contracts from wire to domain. |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | Exports the input-contract mapper/type through the domain barrel. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Derives repair-attempt UI options and validation from the catalog contract. |
| `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json` | Updates the pinned catalog fixture with the backend contract. |
| `tests/test_landing_page_repair_contract.py` | Adds backend contract/default agreement tests. |
| `tests/test_extracted_content_control_surface_api.py` | Pins the catalog input-contract payload. |
| `tests/test_extracted_content_control_surfaces.py` | Pins preview validation parity when quality gates are disabled. |
| `tests/test_extracted_landing_page_generation.py` | Pins runtime rejection for invalid direct repair-attempt overrides. |

## Mechanism

`landing_page_repair_contract.py` owns:

- `LANDING_PAGE_QUALITY_REPAIR_INPUT`
- `LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_DEFAULT`
- `LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MIN`
- `LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS_MAX`
- `landing_page_quality_repair_attempts_from_inputs(...)`
- `normalize_landing_page_quality_repair_attempts(...)`

Backend preview and plan code call the same parser. The landing-page generation
service also validates direct per-call overrides against the same max, so calls
that bypass the plan path cannot silently run with a different rule.

The catalog now includes an `input_contracts` object. The Intel UI maps it into
domain state and uses that catalog value to build the select options, default
label, help text, and client-side preflight validation.

## Intentional

- The UI still validates before submitting because that is a product affordance,
  but the range/default now comes from the backend catalog instead of a local
  cap literal.
- The backend still returns the existing error wording for invalid inputs so
  existing callers and tests keep the same failure shape.
- This PR does not change the max value or default value; it changes ownership
  of those values.
- The Intel UI keeps a legacy compatibility fallback for version skew when the
  frontend deploys before the catalog includes `input_contracts`; the backend
  catalog remains the primary source when available.
- This PR is over the normal 400-LOC target because the source fix crosses the
  backend validator, catalog wire contract, UI consumer, fixture, and tests in
  one coherent business-rule slice.

## Deferred

- A future generated-asset queue UI slice can add repair-attempt list badges
  after PR #736 lands.
- A future fixture-generation cleanup can replace the stale
  `dump_content_ops_fixtures.py` reference in frontend comments if the repo
  wants a documented fixture regeneration command.

## Verification

- `pytest tests/test_landing_page_repair_contract.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_control_surface_api.py tests/test_extracted_landing_page_generation.py -q` -> passed, 179 tests.
- `npm run lint` from `atlas-intel-ui` -> passed.
- `npm run build` from `atlas-intel-ui` -> passed; Vite built successfully,
  generated the sitemap, and pre-rendered public routes.
- `python -m json.tool atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json >/dev/null` -> passed.
- `git diff --check` -> passed with 0 whitespace errors.
- `bash scripts/local_pr_review.sh origin/main` -> passed all wrapper checks:
  pre-push audit wrapper, plan/code consistency, and `git diff --check`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~95 |
| Backend contract and callers | ~190 |
| Frontend contract wiring | ~155 |
| Tests/fixture | ~155 |
| Total | ~595 |
