# Plan: Content Ops API-key auth

## Why this slice exists

The hosted FAQ deflection funnel currently uses `ATLAS_B2B_JWT` as the portfolio
server's ATLAS credential. That token is a normal login JWT with the deployed
ATLAS default 24-hour lifetime, so the live funnel breaks every day unless an
operator manually renews and redeploys the portfolio.

ATLAS already has long-lived `atls_live_*` API keys for server-to-server calls,
but the Content Ops route mount still gates requests through JWT-only
`require_b2b_plan("b2b_growth")`. This slice makes the Content Ops routes accept
a scoped B2B service API key while preserving the existing B2B plan/product gate.
The final diff is slightly over the 400-line soft cap because the review-driven
coverage must exercise the real API-key dispatch path, scope tagging, and B2B
gate together for this revenue-path auth boundary.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Production hardening

1. Add a B2B plan dependency that accepts either the existing dashboard JWT or a
   customer/service `atls_live_*` API key.
2. Require Content Ops API keys to carry an explicit Content Ops scope such as
   `content_ops:deflection:*` or `content_ops:*`; JWT callers remain unchanged.
3. Mount the Content Ops routers through the new dependency so the deflection
   submit/snapshot/artifact paths can use long-lived service keys.
4. Add focused auth tests for JWT compatibility, scoped API-key acceptance,
   full API-key dispatch tagging, insufficient API-key scope rejection,
   plan/product rejection, and the Content Ops route mount.
5. Add a dedicated `.github/workflows/atlas_content_ops_auth_checks.yml`
   workflow and enroll the new atlas-brain test in both path filters and the
   pytest run step.

### Files touched

- `plans/PR-Content-Ops-API-Key-Auth.md` - this plan doc.
- `atlas_brain/auth/dependencies.py` - B2B JWT-or-API-key dependency and scope
  enforcement.
- `atlas_brain/api/__init__.py` - Content Ops route mount uses the new
  dependency.
- `tests/test_auth_dependencies.py` - focused dependency and route-mount tests.
- `.github/workflows/atlas_content_ops_auth_checks.yml` - dedicated CI lane.

## Mechanism

`require_api_key()` already resolves an `atls_live_*` bearer token into the same
`AuthUser` shape as JWT auth. This slice records the API-key scopes on that
`AuthUser`, then adds:

```python
def require_b2b_plan_or_api_key(min_plan: str):
    async def _check(user: AuthUser = Depends(require_auth_or_api_key)) -> AuthUser:
        ...
```

The shared B2B plan check still rejects past-due accounts, non-B2B products, and
plans below `b2b_growth`. If the caller is an API key, the dependency also
requires one of:

- `content_ops:*`
- `content_ops:deflection:*`

The Content Ops mount swaps `_capture_content_ops_auth_user` from
`require_b2b_plan("b2b_growth")` to `require_b2b_plan_or_api_key("b2b_growth")`.
Downstream scope capture still writes the authenticated account into the
ContextVar, so tenant scoping stays unchanged.

## Intentional

- This does not lengthen JWT expiry. Login JWTs stay short-lived; service access
  moves to revocable API keys.
- Existing JWT dashboard callers continue to work without scope strings because
  JWT authorization is still governed by the B2B product/plan gate.
- Existing default `llm:*` API keys and broad `*` keys do not pass the Content
  Ops gate. The portfolio service key must be minted with an explicit Content
  Ops scope.
- Cross-layer caller hints were inspected: existing `AuthUser` constructors are
  unaffected because the new fields have defaults, and existing
  `require_b2b_plan` callers keep the same plan/product/past-due behavior via
  the shared helper covered in `tests/test_auth_dependencies.py`.

## Deferred

- Portfolio env migration is the follow-up slice: mint
  `atlas-portfolio-deflection-prod` with `content_ops:deflection:*`, set
  `ATLAS_B2B_SERVICE_TOKEN`, update portfolio to prefer it over
  `ATLAS_B2B_JWT`, redeploy, then remove the short-lived JWT.

Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/auth/dependencies.py atlas_brain/api/__init__.py tests/test_auth_dependencies.py` - passed.
- `python -m pytest tests/test_auth_dependencies.py -q` - 24 passed after
  reviewer coverage update.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-api-key-auth-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| Auth dependency | ~60 |
| Route mount | ~5 |
| Focused tests | ~210 |
| Workflow | ~55 |
| **Total** | 435 |
