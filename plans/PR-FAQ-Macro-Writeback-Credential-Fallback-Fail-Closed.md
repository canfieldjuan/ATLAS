# PR-FAQ-Macro-Writeback-Credential-Fallback-Fail-Closed

## Why this slice exists

PR #1146 added tenant Zendesk credential lookup for FAQ macro writeback and
intended config credentials to remain available only for unscoped
local/single-tenant operation. The remaining gap is the boundary between
"unscoped" and "tenant-scoped but missing an account id": a scope that carries
tenant/user markers without a usable `account_id` can still reach the config
fallback. That can let a mis-scoped hosted publish borrow shared Zendesk
credentials, so this production-hardening slice closes that fallback path.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Production hardening

1. Tighten FAQ macro writeback credential resolution so config fallback is
   available only for an empty `TenantScope`.
2. Fail closed when a scope contains authenticated tenant markers but no
   usable account id.
3. Add focused regression coverage proving user, role, and vendor-scoped
   requests do not call the config fallback.
4. Add a caller-layer smoke regression proving config credentials do not let a
   user-scoped live smoke proceed without an account id.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Credential-Fallback-Fail-Closed.md` - plan for
  this slice.
- `atlas_brain/_content_ops_macro_writeback.py` - fallback boundary for
  tenant Zendesk credentials.
- `tests/test_atlas_content_ops_macro_writeback.py` - regression tests for
  tenant-marker fail-closed behavior.
- `tests/test_faq_macro_writeback_live_zendesk_smoke.py` - caller-layer
  regression for the default live smoke credential provider.

## Mechanism

`TenantZendeskMacroCredentialsProvider.credentials_for_scope(...)` keeps the
existing tenant-row lookup when `scope.account_id` is non-empty after trimming.
If the account id is missing or blank but the scope still carries hosted tenant
markers (`user_id`, `allowed_vendors`, or `roles`), it returns `None` instead
of calling the fallback provider. A completely empty `TenantScope()` remains
eligible for `ConfigZendeskMacroCredentialsProvider`, preserving local
single-tenant usage and existing tests.

## Intentional

- This slice does not remove config-based Zendesk credentials. They still
  support unscoped local/single-tenant runs where no tenant context exists.
- The fail-closed predicate stays local to the host credential adapter because
  package-level macro publishing only knows whether credentials resolved, not
  whether the host scope was safely scoped.
- No config fields are added; the existing typed `B2BCampaignConfig` Zendesk
  fields remain the only config-backed source.
- Cross-layer caller hints were inspected. The package publisher already maps
  missing credentials to `zendesk_credentials_missing`; this slice adds a
  live-smoke caller regression for the default host provider path.

## Deferred

- None.

Parked hardening: none

## Verification

- python -m pytest tests/test_atlas_content_ops_macro_writeback.py tests/test_content_ops_faq_macro_writeback_flow.py tests/test_content_ops_zendesk_credentials.py tests/test_faq_macro_writeback_live_zendesk_smoke.py -q - 28 passed.
- python -m py_compile atlas_brain/_content_ops_macro_writeback.py tests/test_atlas_content_ops_macro_writeback.py tests/test_content_ops_faq_macro_writeback_flow.py tests/test_content_ops_zendesk_credentials.py tests/test_faq_macro_writeback_live_zendesk_smoke.py - passed.
- bash scripts/run_extracted_pipeline_checks.sh - passed; extracted reasoning core 295 passed, extracted content pipeline 2792 passed / 10 skipped.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Provider guard | ~25 |
| Tests | ~110 |
| Total | ~210 |
