# PR-Security-Blog-Admin-Authz

## Why this slice exists

Security issue #1656 calls out `/admin/blog/*` as authenticated but not
administrator-authorized. The router exposes platform blog draft, quality, edit,
publish, and generation controls; regular authenticated account users should not
be able to operate that console.

## Scope (this PR)

Ownership lane: security/rbac-blog-admin
Slice phase: Production hardening

1. Require platform-admin authorization for every `/admin/blog/*` endpoint.
2. Add route-level proof that account owners/admins without platform-admin
   authority receive `403`, while the existing blog admin route coverage uses a
   platform-admin test principal.

### Review Contract

- Acceptance criteria:
  - [ ] Every `/admin/blog/*` route depends on `require_blog_admin_user`
        instead of depending on `require_auth` directly.
  - [ ] An authenticated account owner/admin without `is_platform_admin` gets
        `403` before blog admin route handler or database work can run.
  - [ ] A platform-admin principal retains access to the existing blog admin
        route behaviors covered by the route tests.
- Affected surfaces: FastAPI blog admin API authorization and route tests.
- Risk areas: security, backward compatibility, accidental platform-admin
  lockout.
- Reviewer rules triggered: R1, R2, R3, R5, R14.

### Files touched

- `atlas_brain/api/blog_admin.py`
- `plans/PR-Security-Blog-Admin-Authz.md`
- `tests/test_truthful_artifact_routes.py`

## Mechanism

`blog_admin.py` keeps composing through the existing `require_auth` dependency so
the session/token contract remains unchanged. A local `require_blog_admin_user`
dependency then checks `AuthUser.is_platform_admin` and raises
`403 Platform admin access required` for authenticated users who are only account
owners/admins. Every router endpoint depends on that helper instead of depending
on `require_auth` directly.

The route tests assert that each `/admin/blog/*` operation is wired to the new
dependency, exercise every route method with a non-platform-admin user, and keep
the existing success-path route tests on a platform-admin fixture.

## Intentional

- This slice does not import `require_admin_user` from `admin_costs.py`; that
  helper permits account owners/admins and pulls in unrelated cost telemetry
  imports. Blog administration is a platform operation, so the gate is local and
  checks `is_platform_admin`.
- This slice does not change token validation, plan checks, database access, or
  route response shapes. It only tightens authorization for the existing admin
  surface.

## Deferred

The remaining #1656 work is still deferred to its own slices or operational
tasks: provider credential rotation, backend report deletion/TTL, report RLS,
encryption at rest, alert delivery channel setup, `?token=` link removal,
scanner ratchets, IR/security docs, CVE SLA docs, and structured
logging/error-tracking hardening.

Parked hardening: none.

## Verification

- Reviewer rules triggered: R1, R2, R3, R5, R14.
- `python -m pytest tests/test_truthful_artifact_routes.py -k "blog_admin or blog_quality or blog_draft_evidence or blog_publish or manual_blog_generate" -q` — passed (`21 passed, 32 deselected`, one existing `pynvml` deprecation warning).
- `python -m pytest tests/test_truthful_artifact_routes.py tests/test_blog_admin_inputs.py -q` — passed (`56 passed`, one existing `pynvml` deprecation warning).
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-security-blog-admin-authz-pr-body.md` — passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/blog_admin.py` | 24 |
| `plans/PR-Security-Blog-Admin-Authz.md` | 87 |
| `tests/test_truthful_artifact_routes.py` | 90 |
| **Total** | **201** |
