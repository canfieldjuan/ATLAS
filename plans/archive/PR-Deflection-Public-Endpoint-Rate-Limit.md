# PR-Deflection-Public-Endpoint-Rate-Limit

## Why this slice exists

#1386's current-state verification marks paid-report access control as
half-done: request ID entropy is now closed, but the public deflection submit,
snapshot, artifact, and checkout-authorization endpoints are still reachable
without route-level throttling. That leaves the paid funnel exposed to brute
force attempts against request IDs and public submit abuse even though the
shared Atlas slowapi limiter already exists for other API surfaces.

Root cause: the deflection routes live in the standalone extracted router while
the concrete limiter lives in the Atlas host. Fixing one route body would be a
symptom patch; the upstream boundary is the router dependency seam that lets
the host attach abuse controls without making the extracted package import
Atlas auth code.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Add an extracted-router dependency seam for the public paid-deflection
   endpoints: submit, snapshot, artifact, and checkout authorization.
2. Wire Atlas to pass the existing slowapi limiter through that seam using a
   shared paid-funnel scope, so rotating request IDs does not reset the bucket.
3. Set the rate-limit identity from the authenticated content-ops user or
   scoped API key before the limiter runs, and add B2B plan entries to the
   shared rate table so `b2b_growth` does not silently fall back to `trial`.
4. Preserve the trusted paid-release endpoint's separate operator dependency
   path; Stripe/webhook unlocks must not inherit the public limiter.
5. Add focused tests that prove the public deflection routes receive the seam
   dependency, the paid route does not, Atlas mounts the seam, request-id route
   rotation shares a bucket, and separate accounts get separate budgets.

### Review Contract

- Acceptance criteria:
  - [ ] Public paid-deflection submit, snapshot, artifact, and checkout
        authorization routes have the injected public dependency when a host
        supplies it.
  - [ ] The paid-release route remains protected only by its existing trusted
        dependency path and does not inherit public throttling.
  - [ ] Atlas host wiring supplies a slowapi-backed limiter dependency for the
        public deflection routes without importing Atlas auth code into the
        extracted package.
  - [ ] Rotating request IDs across snapshot/artifact/checkout paths does not
        bypass the public paid-funnel throttle.
  - [ ] Content-ops authenticated accounts/API keys receive account-scoped
        rate-limit buckets instead of collapsing all traffic to proxy IP.
  - [ ] Existing route behavior and response contracts are unchanged aside from
        FastAPI dependency execution.
- Affected surfaces: API, auth, CI, extracted package boundary.
- Risk areas: security, backcompat, standalone package imports.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12, R14.

### Files touched

- `.github/workflows/atlas_content_ops_auth_checks.yml`
- `.github/workflows/extracted_pipeline_checks.yml`
- `atlas_brain/api/__init__.py`
- `atlas_brain/auth/rate_limit.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `plans/PR-Deflection-Public-Endpoint-Rate-Limit.md`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_llm_gateway_plan_tier.py`

## Mechanism

The extracted router factory gains an optional
`deflection_report_public_dependencies` sequence. The four public paid-funnel
routes use that sequence in their route decorators. The existing
`deflection_report_paid_dependencies` sequence stays attached only to the
paid-release route.

Atlas defines a small host-local dependency by applying the existing slowapi
shared limiter with the existing dynamic rate selection and a stable
`content_ops_deflection_public` scope, then passes that dependency to the
extracted router. The content-ops auth bridge writes `rate_limit_key` and
`rate_limit_plan` onto `request.state` from the authenticated user/API key
before the route-level limiter runs. The shared rate table now includes B2B
plans, so `b2b_growth|account_id` selects the intended B2B bucket instead of
falling back to the default.

## Intentional

- No Atlas auth import inside `extracted_content_pipeline`: the extracted
  package remains standalone and host-neutral.
- No new env vars in this slice: the shared Atlas rate table remains the
  policy source, with B2B tier entries added because content-ops uses B2B plans.
- No rate limit on the trusted paid-release path: Stripe/webhook unlocks are
  already behind the trusted operator dependency and should not be throttled by
  buyer/browser traffic.

## Deferred

- Stronger paid-funnel-specific limits or captcha can be added later if public
  traffic needs a stricter policy than the shared trial fallback.
- #1419 remains the report-quality proof gate on a real resolution-bearing
  export; this slice only closes the access-control abuse-control gap.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_content_control_surface_api.py tests/test_atlas_content_ops_generated_assets_api.py tests/test_llm_gateway_plan_tier.py -q`
  - 185 passed, 1 skipped, 1 warning.
- `scripts/run_extracted_pipeline_checks.sh`
  - 4300 passed, 10 skipped, 1 warning.
- `python -m pytest tests/test_auth_dependencies.py tests/test_llm_gateway_plan_tier.py -q`
  - 43 passed, 1 warning.
- `scripts/audit_extracted_pipeline_ci_enrollment.py`
  - Passed; 182 matching tests enrolled.
- Pending before push:
  - `python scripts/sync_pr_plan.py plans/PR-Deflection-Public-Endpoint-Rate-Limit.md --check`
  - local review via `scripts/local_pr_review.sh`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_auth_checks.yml` | 5 |
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `atlas_brain/api/__init__.py` | 19 |
| `atlas_brain/auth/rate_limit.py` | 4 |
| `extracted_content_pipeline/api/control_surfaces.py` | 23 |
| `plans/PR-Deflection-Public-Endpoint-Rate-Limit.md` | 132 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 133 |
| `tests/test_extracted_content_control_surface_api.py` | 14 |
| `tests/test_llm_gateway_plan_tier.py` | 10 |
| **Total** | **342** |
