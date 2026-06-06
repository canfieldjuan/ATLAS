# PR-FAQ-Macro-Writeback-Tenant-Credentials

## Why this slice exists

FAQ macro writeback can now publish, reconcile, and recover pending Zendesk
macro mappings, but hosted credentials are still process-level config. That is
not enough for real multi-customer use: one tenant's macro publish path must
resolve that tenant's Zendesk credentials, not a shared deployment credential.
This slice adds encrypted, tenant-scoped Zendesk credential storage and wires
macro writeback to prefer it while preserving the existing config fallback only
for unscoped local/single-tenant operation.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Production hardening

1. Add a Postgres table for active/revoked Content Ops Zendesk credentials.
2. Add an Atlas service that encrypts Zendesk API tokens using the existing
   BYOK Fernet KEK helpers and decrypts them on tenant-scoped lookup.
3. Add a host credentials provider that resolves tenant credentials from the
   DB first, and fails closed for tenant-scoped publishes when no row exists.
4. Wire the macro publish provider to use the tenant-aware credentials source.
5. Add tests for migration shape, encrypted storage behavior, lookup
   scoping/decryption, and provider fallback order.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Tenant-Credentials.md` — plan for this slice.
- `atlas_brain/storage/migrations/330_content_ops_zendesk_credentials.sql` — credential storage.
- `atlas_brain/_content_ops_zendesk_credentials.py` — encrypted credential service.
- `atlas_brain/_content_ops_macro_writeback.py` — tenant-aware provider wiring.
- `scripts/run_extracted_pipeline_checks.sh` — CI enrollment for the new credential test.
- `tests/test_content_ops_zendesk_credentials.py` — service and migration coverage.
- `tests/test_atlas_content_ops_macro_writeback.py` — host wiring coverage.

## Mechanism

The new table stores one active Zendesk credential row per SaaS account. API
tokens are encrypted via `atlas_brain.auth.encryption.encrypt_secret`, reusing
the existing BYOK KEK rotation model through lazy wrapper functions so the
module remains importable in lean extracted CI. The table stores display-safe
fields (`email`, `subdomain`, `base_url`, token prefix) plus
`encrypted_api_token` and `encryption_kid`; active rows are soft-revoked with
`revoked_at`.

`lookup_zendesk_credentials(...)` filters by account id and active rows only,
decrypts the API token with `decrypt_secret`, and returns
`ZendeskMacroCredentials`. If no row exists it returns `None`; if a row exists
but decrypt fails, it also returns `None` so the publish path does not use a
broken credential.

`TenantZendeskMacroCredentialsProvider` asks the tenant store when an account id
is available. If no tenant credential exists, it returns `None` so tenant-scoped
publish attempts fail closed instead of borrowing deployment-level Zendesk
credentials. The existing `ConfigZendeskMacroCredentialsProvider` remains
available only for unscoped local/single-tenant calls.

## Intentional

- This slice does not add dashboard/API endpoints to create credentials. It
  adds the storage and resolver foundation the API can call next.
- API tokens never appear in display DTOs or logs; tests assert display rows do
  not carry plaintext or ciphertext.
- Config fallback is only used when the scope has no account id. Tenant-scoped
  publishes fail closed without a tenant credential to avoid cross-tenant
  writes.
- The service uses the existing BYOK KEK instead of introducing a second
  encryption secret.

## Deferred

- `PR-FAQ-Macro-Writeback-Credential-API`: tenant-authenticated endpoints or
  admin CLI for adding/listing/revoking Zendesk credentials.
- `PR-FAQ-Macro-Writeback-Publish-UI`: review UI action for the macro publish
  route.

Parked hardening: none

## Verification

- python -m pytest tests/test_content_ops_zendesk_credentials.py tests/test_atlas_content_ops_macro_writeback.py -q — 14 passed.
- python -m py_compile atlas_brain/_content_ops_zendesk_credentials.py atlas_brain/_content_ops_macro_writeback.py tests/test_content_ops_zendesk_credentials.py tests/test_atlas_content_ops_macro_writeback.py — passed.
- python - <<'PY' ... import atlas_brain._content_ops_zendesk_credentials with cryptography blocked ... PY — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-tenant-credentials.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~95 |
| Migration | ~34 |
| Service / wiring / CI | ~300 |
| Tests | ~280 |
| Total | ~714 |

This is over the 400 LOC soft cap because credential storage must ship with the
encryption boundary, scoped lookup, provider wiring, and tests together. A
smaller slice would leave either unused storage or unproven credential routing.
