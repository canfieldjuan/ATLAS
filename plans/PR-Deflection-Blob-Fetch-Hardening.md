# PR-Deflection-Blob-Fetch-Hardening

## Why this slice exists

#1561 drained the CSV blob same-root memory copy, and review verified the
streaming behavior. The remaining review NITs were all small fetch-boundary
survivability gaps on the same public paid-submit path: response close should
not be able to leak the underlying connection, validated-response setup should
cleanup symmetrically if a future edit raises after response creation, dead
transport branches should not obscure the actual http.client path, and transport
failures need a warning for operational diagnosis. This slice tightens those
edges without changing submit behavior.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Make `_PinnedBlobResponse.close()` close the pinned HTTPS connection even if
   response close raises.
2. Make `_open_validated_https_blob_response(...)` cleanup any opened response
   on all exception exits.
3. Remove the dead `urllib.error.HTTPError` branch from the pinned
   `http.client` fetch helper and keep redirect rejection on the status check.
4. Log transport/staging failures before returning the existing generic 400
   envelope.
5. Add focused regression tests for connection-close-on-response-close-error and
   fetch-failure logging/cleanup.

### Review Contract

- Acceptance criteria:
  - A response-close failure cannot prevent the connection from closing.
  - A fetch helper exception after response creation closes that response.
  - Blob URL public error envelopes remain unchanged.
  - Transport failures are logged without leaking URLs with credentials.
- Affected surfaces: `extracted_content_pipeline/api/control_surfaces.py`,
  `tests/test_extracted_content_deflection_submit.py`.
- Risk areas: accidentally changing SSRF validation, surfacing internal fetch
  details to users, and broad catch blocks swallowing actionable exceptions.
- Reviewer rules triggered: R1, R2, R8, R9, R10, R14.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Blob-Fetch-Hardening.md`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

`_PinnedBlobResponse.close()` becomes a `try/finally`: attempt to close the
HTTP response, always close the pinned connection. Fetch helper cleanup also uses
a small local helper so every exception branch closes an already-opened response
before raising. The public exception detail remains the existing
`"Blob URL could not be fetched."` or redirect-specific message.

Transport exceptions are logged with the URL host/path only after validation has
accepted the target. The log keeps user-facing envelopes stable while giving
operators a diagnostic reason for origin-side failures.

## Intentional

- This PR does not change the redirect/status behavior from #1561. Redirects are
  still rejected by status code and not followed.
- This PR does not surface raw exception details to the buyer-facing API
  response; it logs them server-side and preserves the existing generic 400.
- This PR does not take over #1560 or any repeat-gate work.

## Deferred

- None.

Parked hardening: none.

## Verification

- python -m py_compile extracted_content_pipeline/api/control_surfaces.py
  tests/test_extracted_content_deflection_submit.py
- python -m pytest tests/test_extracted_content_deflection_submit.py -q
  (60 passed)
- bash scripts/check_ascii_python.sh
- bash scripts/validate_extracted_content_pipeline.sh
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline
- python scripts/audit_extracted_standalone.py --fail-on-debt
- bash scripts/run_extracted_pipeline_checks.sh (reasoning core: 295 passed;
  content pipeline: 4189 passed, 10 skipped)

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 46 |
| `plans/PR-Deflection-Blob-Fetch-Hardening.md` | 96 |
| `tests/test_extracted_content_deflection_submit.py` | 77 |
| **Total** | **219** |
