# PR-FAQ-Deflection-Blob-Redirect-Hardening

## Why this slice exists

PR #1166 added the portfolio blob submit route and fixed the blocking SSRF
review by resolving blob hostnames before fetch and disabling redirects. The
review left one skip-worthy but useful robustness gap: the route-level redirect
test stubs the fetch helper, so the actual `_open_https_blob_request` wiring is
not pinned by a regression. This slice locks in that opener-level behavior and
parks the larger DNS-rebinding TOCTOU follow-up outside this small test slice.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-submit
Slice phase: Robust testing

1. Add a focused test that exercises `_read_bounded_https_blob` through the
   real `_open_https_blob_request` boundary.
2. Assert the opener is built with `_NoRedirectHandler` and maps a 3xx
   `HTTPError` to the customer-safe redirect rejection.
3. Park the DNS-rebinding pinned-connection follow-up in `HARDENING.md`.

### Files touched

- `plans/PR-FAQ-Deflection-Blob-Redirect-Hardening.md` - plan for this slice.
- `tests/test_extracted_content_deflection_submit.py` - opener-level redirect
  regression.
- `HARDENING.md` - parked DNS-rebinding hardening follow-up.

## Mechanism

The new test calls `_read_bounded_https_blob(...)` directly after mocking public
DNS resolution and `urllib.request.build_opener`. The mock opener records the
request and raises an HTTP 302. Because the test does not replace
`_open_https_blob_request`, it verifies the production helper constructs the
opener with `_NoRedirectHandler`, passes the expected URL/timeout, and returns
the same 400 "redirects are not allowed" error that the route surfaces.

## Intentional

- This slice does not implement full DNS-rebinding protection. The current
  preflight DNS check closes common hostname-to-private cases, but urllib can
  still re-resolve at connect time; pinning the validated IP belongs in a
  larger networking-hardening slice.
- No route behavior changes are expected. This is a regression-test slice for
  behavior already merged in #1166.

## Deferred

- `PR-FAQ-Deflection-Blob-DNS-Pinning`: connect to the validated IP with an
  explicit `Host` header, or restrict submit blobs to a trusted blob host set,
  so DNS rebinding cannot swap the target after preflight validation.

Parked hardening: `FAQ deflection blob submit DNS-rebinding TOCTOU`.

## Verification

- python -m pytest tests/test_extracted_content_deflection_submit.py -q - 9 passed.
- python -m py_compile tests/test_extracted_content_deflection_submit.py - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| Test | ~45 |
| Hardening | ~10 |
| Total | ~120 |
