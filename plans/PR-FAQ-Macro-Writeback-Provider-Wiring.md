# PR-FAQ-Macro-Writeback-Provider-Wiring

## Why this slice exists

The FAQ macro publish route is mounted, but it only works when the host injects
a `MacroPublishProvider`. The next missing production slice is Atlas host
wiring: build the Zendesk provider with the existing database pool for
idempotency mappings and env-backed Zendesk credentials, then pass that provider
factory into the generated asset router.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add typed Zendesk macro writeback settings to `B2BCampaignConfig`.
2. Add a host helper that builds `ZendeskMacroPublishProvider` with
   `PostgresFAQMacroWritebackMappingRepository`.
3. Read Zendesk credentials through centralized config without storing or
   logging secrets.
4. Wire the helper into `create_generated_asset_router` in the Atlas API
   aggregator.
5. Add focused tests for config credential parsing, missing credential behavior,
   provider construction, and API aggregator wiring.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Provider-Wiring.md` — plan for this slice.
- `atlas_brain/config.py` — typed Content Ops Zendesk credential settings.
- `atlas_brain/_content_ops_macro_writeback.py` — host macro provider factory.
- `atlas_brain/api/__init__.py` — generated asset router provider injection.
- `tests/test_atlas_content_ops_macro_writeback.py` — host provider factory coverage.
- `tests/test_atlas_content_ops_generated_assets_api.py` — API aggregator wiring assertion.
- `scripts/run_extracted_pipeline_checks.sh` — extracted-package CI enrollment for the new host wiring test.

## Mechanism

`build_content_ops_macro_publish_provider` accepts the same `get_db_pool` style
provider used by the rest of the Content Ops host wiring. It returns a
`ZendeskMacroPublishProvider` composed from:

- `EnvZendeskMacroCredentialsProvider`
- `PostgresFAQMacroWritebackMappingRepository(pool)`

Credentials are read through `B2BCampaignConfig` fields backed by these primary
env vars:

- `ATLAS_CONTENT_OPS_ZENDESK_EMAIL`
- `ATLAS_CONTENT_OPS_ZENDESK_API_TOKEN`
- `ATLAS_CONTENT_OPS_ZENDESK_SUBDOMAIN`
- `ATLAS_CONTENT_OPS_ZENDESK_BASE_URL`

The generic `ZENDESK_*` names are accepted as typed fallback aliases for local
operator convenience. Incomplete credentials return `None` from the credential
provider; the Zendesk adapter then produces per-macro
`zendesk_credentials_missing` failures without exposing secrets.

## Intentional

- No tenant-specific credential table yet. This slice wires the deployed route
  to one host-managed Zendesk credential source; multi-tenant credential storage
  belongs in a later product/security slice.
- No env logging and no token-bearing metadata. Credentials only enter the
  Zendesk adapter.
- The provider factory still returns a provider when credentials are missing, so
  the route remains reachable and the publish summary can explain the credential
  failure per macro.

## Deferred

- `PR-FAQ-Macro-Writeback-Tenant-Credentials`: replace host env credentials with
  tenant-scoped encrypted credential storage when more than one customer
  support platform account needs live writeback.
- `PR-FAQ-Macro-Writeback-Pending-Reconcile`: backfill pending Zendesk mappings
  by looking up reserved title/category metadata.
- `PR-FAQ-Macro-Writeback-Publish-UI`: add the review UI action now that the
  route and host provider wiring exist.

Parked hardening: none

## Verification

- python -m pytest tests/test_atlas_content_ops_macro_writeback.py tests/test_atlas_content_ops_generated_assets_api.py -q — 17 passed, 1 warning.
- python -m py_compile atlas_brain/config.py atlas_brain/_content_ops_macro_writeback.py atlas_brain/api/__init__.py tests/test_atlas_content_ops_macro_writeback.py tests/test_atlas_content_ops_generated_assets_api.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- python scripts/check_extracted_imports.py — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py — passed.
- python scripts/smoke_extracted_pipeline_imports.py — passed.
- python scripts/smoke_extracted_pipeline_standalone.py — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-provider-wiring.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~105 |
| Config | ~40 |
| Host helper | ~95 |
| API mount | ~5 |
| Tests | ~150 |
| CI enrollment | ~5 |
| Total | ~400 |
