# PR-FAQ-Deflection-Submit-Request-Mapping-Fix

## Why this slice exists

The hosted FAQ deflection submit handoff still fails for multipart CSV uploads.
The source route is no longer JSON-only, but a local FastAPI `TestClient`
reproduction exposed the real runtime bug: Starlette `Request` implements
`Mapping`, and `_load_deflection_submit_rows_from_request` checks
`isinstance(request, Mapping)` before it checks the HTTP content type. Live
FastAPI requests therefore enter the legacy JSON/blob mapping path instead of
the multipart parser.

The prior direct endpoint tests passed because their fake multipart request was
not a `Mapping`. This slice closes that test gap and fixes the route order so
the actual hosted request object follows the multipart branch.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Functional validation

1. Treat HTTP request-like objects as requests before accepting plain mapping
   payloads for the legacy JSON blob-url submit path.
2. Preserve the existing direct mapping support for tests and non-HTTP callers.
3. Add a FastAPI `TestClient` multipart regression that exercises real request
   dispatch instead of direct endpoint invocation.

### Files touched

- `plans/PR-FAQ-Deflection-Submit-Request-Mapping-Fix.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

`_load_deflection_submit_rows_from_request` now identifies request-like objects
by their HTTP surface (`headers` plus `form` or `json`) before falling back to
the plain `Mapping` path:

```python
if _is_deflection_submit_http_request(request):
    inspect content-type and parse multipart/json
elif isinstance(request, Mapping):
    legacy blob_url mapping path
```

This keeps dict payloads working while preventing Starlette `Request` from
being validated as `DeflectionReportSubmitModel`.

## Intentional

- This slice does not change the public submit contract, artifact gating, or
  paid trust boundary.
- The legacy JSON blob-url submit path remains available; only live HTTP
  request classification changes.
- No Docker or deployment scripts are touched; this is the route runtime bug
  that shows up under real FastAPI dispatch.

## Deferred

- Parked hardening: none. Existing `HARDENING.md` entries do not touch this
  submit route.
- Live hosted submit proof remains a follow-up operator run after this PR
  lands and ATLAS is redeployed.

## Verification

- `python -m pytest tests/test_extracted_content_deflection_submit.py -q` - 18 passed.
- `python -m py_compile extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_deflection_submit.py` - passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed; 295 reasoning-core
  tests passed, then 2865 extracted-content-pipeline tests passed, 10 skipped,
  1 warning.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/faq-deflection-submit-request-mapping-fix-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 76 |
| Route fix | 20 |
| TestClient regression | 55 |
| **Total** | **151** |

Actual diff is 3 files, +169 / -27.
