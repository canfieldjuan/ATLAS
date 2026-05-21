# PR-Landing-Page-Repair-Attempt-Contract

## Why this slice exists

Landing-page quality repair attempt validation drifted across layers. The same integer range and max cap were duplicated in generation planning, preview cost estimation, and the Intel UI page. That caused a window where backend-valid values could be shown as invalid in the UI, and future changes would require manual edits in multiple places.

This slice closes the drift path with one backend validator, one frontend validator module, and a contract test that pins the shared max/default expectations across those layers.

## Scope (this PR)

1. Promote landing-page repair attempt input name, max value, and validation to a shared backend control-surface helper.
2. Make generation planning call the shared backend helper instead of reimplementing nonnegative integer validation.
3. Move the Intel UI repair-attempt constant/options/normalizer out of the page into a small content-ops domain module.
4. Add contract tests that assert backend preview, backend generation planning, landing-page generation defaults, catalog defaults, and UI max constant agree.

### Files touched

| File | Intent |
|---|---|
| `extracted_content_pipeline/control_surfaces.py` | Own the backend repair-attempt input constant, max constant, and validator helper. |
| `extracted_content_pipeline/generation_plan.py` | Reuse the backend repair-attempt validator. |
| `tests/test_extracted_content_control_surfaces.py` | Add backend/UI contract coverage for repair-attempt max/default agreement. |
| `atlas-intel-ui/src/domain/contentOps/repairAttempts.ts` | Own the frontend repair-attempt constants/options/normalizer. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Use the frontend repair-attempt domain helper instead of local literals. |
| `plans/PR-Landing-Page-Repair-Attempt-Contract.md` | Plan contract for this slice. |

## Mechanism

Backend code calls:

```python
landing_page_quality_repair_attempt_input(inputs)
```

for both preview and generation planning. That helper enforces integer-only values from `0` through `MAX_LANDING_PAGE_QUALITY_REPAIR_ATTEMPTS`.

Frontend code imports the same-shaped UI contract from `domain/contentOps/repairAttempts.ts`, so the page no longer owns the max literal or normalizer. A Python contract test reads the UI max export and asserts it matches the backend max, while also asserting the catalog default mirrors `LandingPageGenerationConfig().quality_repair_attempts`.

## Intentional

- The backend and frontend cannot share one runtime module without a larger generated contract. This slice uses shared helpers inside each runtime plus a cross-language contract test.
- The UI still validates client-side for fast feedback, but the test prevents the UI cap from drifting from backend validation.
- This does not change the allowed range; valid values remain `0..10`.

## Deferred

- Generated frontend constants from backend catalog/schema. That would remove the final cross-language manual mirror but is larger than this slice.
- Parse-retry default mirror cleanup. This slice focuses only on landing-page quality repair attempts.

## Verification

Local verification:

```bash
pytest tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_generation_plan.py -q
# 72 passed

bash scripts/run_extracted_pipeline_checks.sh
# extracted_reasoning_core: 295 passed
# extracted_content_pipeline: 1616 passed, 1 warning

npm --prefix atlas-intel-ui run build
# passed

bash scripts/local_pr_review.sh origin/main
# passed
```

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/control_surfaces.py` | 24 |
| `extracted_content_pipeline/generation_plan.py` | 32 |
| `tests/test_extracted_content_control_surfaces.py` | 35 |
| `atlas-intel-ui/src/domain/contentOps/repairAttempts.ts` | 42 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 47 |
| `plans/PR-Landing-Page-Repair-Attempt-Contract.md` | 79 |
| **Total** | **259** |
