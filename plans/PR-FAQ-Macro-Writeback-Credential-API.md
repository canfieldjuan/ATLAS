# PR-FAQ-Macro-Writeback-Credential-API

## Why this slice exists

FAQ macro writeback now has encrypted tenant Zendesk credential storage and a
fail-closed tenant resolver, but operators still have no product/API surface to
provision those credentials. Without a scoped create/list/revoke path, the
storage only works through manual database writes, which blocks real tenant
setup.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add a host Content Ops Zendesk credential router with authenticated tenant
   list plus admin-only add/rotate and revoke endpoints.
2. Mount the router through the existing Content Ops auth capture path so all
   operations use the calling tenant's `account_id`.
3. Return only display-safe credential views: token prefix, endpoint, label,
   timestamps, and ids; never plaintext or ciphertext.
4. Add route tests for account scoping, display-safe responses, revoke behavior,
   invalid ids, and API aggregator mounting.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Credential-API.md` — plan for this slice.
- `atlas_brain/api/content_ops_zendesk_credentials.py` — credential management routes.
- `atlas_brain/api/__init__.py` — host route mounting.
- `scripts/run_extracted_pipeline_checks.sh` — CI enrollment for the new route test.
- `tests/test_content_ops_zendesk_credentials_api.py` — route behavior tests.
- `tests/test_atlas_content_ops_generated_assets_api.py` — API aggregator mount coverage.

## Mechanism

The new router is host-owned rather than extracted because it touches Atlas auth,
Postgres pool access, and encrypted host credential storage. A factory accepts a
`pool_provider` and `auth_dependency`, mirroring the existing Content Ops host
mount style. The default pool provider lazy-loads Atlas database access so the
route module stays importable in the lean extracted CI environment. Each route
resolves the authenticated `AuthUser`, parses `user.account_id` as a UUID,
checks database readiness, and calls the credential service using that account
id. Credential writes are additionally gated to tenant owner/admin users because
the Zendesk credential is a tenant-wide integration secret.

`POST /content-ops/zendesk-credentials` accepts Zendesk email, API token,
subdomain or base URL, and an optional label. The service validates the endpoint,
encrypts the API token, revokes any current active row for the tenant, and
returns a display-safe DTO. `GET /content-ops/zendesk-credentials` lists active
rows for the tenant. `DELETE /content-ops/zendesk-credentials/{credential_id}`
soft-revokes a tenant-owned row and returns 404 for invalid ids or rows outside
the tenant.

## Intentional

- This slice does not add dashboard UI. It creates the minimum authenticated
  API surface the UI or admin tooling can call next.
- This slice does not call Zendesk to validate credentials. Format validation
  and encrypted storage are enough for the provisioning surface; live publish
  already reports credential failures through the macro writeback provider.
- API tokens are accepted only on create/rotate and are never returned.
- Listing is available to authenticated Content Ops users, while create/rotate
  and revoke require tenant owner/admin rights.
- The route fails closed when the database is not initialized.

## Deferred

- `PR-FAQ-Macro-Writeback-Publish-UI`: review UI action for macro publish.
- `PR-FAQ-Macro-Writeback-Credential-UI`: dashboard controls for adding,
  listing, and revoking Zendesk credentials.

Parked hardening: none

## Verification

- python -m pytest tests/test_content_ops_zendesk_credentials_api.py tests/test_atlas_content_ops_generated_assets_api.py -q — 21 passed, 1 warning.
- python -m py_compile atlas_brain/api/content_ops_zendesk_credentials.py tests/test_content_ops_zendesk_credentials_api.py tests/test_atlas_content_ops_generated_assets_api.py — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| API route / mount / CI | ~166 |
| Tests | ~356 |
| Total | ~584 |

This is slightly over the 400 LOC soft cap because the API surface needs scoped
create/list/revoke behavior and auth/mount coverage in the same vertical slice.
