# PR: Content Ops Support-Ticket Input Provider

## Why this slice exists

PR-Content-Ops-Support-Ticket-Input-Package added the concrete package builder
for already-loaded support-ticket rows, but hosts still need a small
`ContentOpsInputProvider` implementation they can pass into the existing
control-surface API wiring. Without that adapter, every host route would need to
rebuild the same glue between tenant scope, request payload, and the support
ticket package builder.

This slice keeps the work in the input-provider lane. It does not implement FAQ
article generation, file upload/import, DB persistence, or new public routes.

## Scope (this PR)

Ownership lane: content-ops/input-provider-ticket-package

1. Add a support-ticket input provider that implements the existing
   `ContentOpsInputProvider` protocol.
2. Let hosts provide support-ticket source material directly or through a sync
   or async loader called with `scope` and `request`.
3. Delegate package construction to `build_support_ticket_input_package(...)`.
4. Keep explicit request/operator overrides in the existing API merge path, not
   in this provider.
5. Add focused tests proving direct source material, sync loader, async loader,
   missing source configuration, and API preview wiring.
6. Enroll the new test file in extracted pipeline CI.

### Files touched

- `extracted_content_pipeline/support_ticket_input_provider.py`
- `tests/test_extracted_support_ticket_input_provider.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Content-Ops-Support-Ticket-Input-Provider.md`

## Mechanism

`SupportTicketInputProvider` is a dataclass with either `source_material` or a
`source_material_loader`. When `build_content_ops_input_package(scope, request)`
runs, it resolves the source material, then delegates to
`build_support_ticket_input_package(...)`.

If the loader returns an awaitable, the provider returns an awaitable package so
the already-wired API helper can await it. This matches the existing
`ContentOpsInputProvider` protocol and keeps host integrations free to load from
memory, a repository, blob storage, or another service.

## Intentional

- No new API route. `create_content_ops_control_surface_router` already accepts
  `input_provider`.
- No FAQ implementation. FAQ generation and standalone FAQ article ownership
  remain with the FAQ session.
- No file parsing or DB access. The loader is host-owned and this provider only
  adapts already-loaded source material.
- Request/operator precedence stays in `merge_content_ops_input_package(...)`
  and the API wiring from the prior slice.

## Deferred

- Future PR: mounted host routes can provide a source loader backed by uploaded
  ticket files or persisted import rows.
- Future PR owned by the FAQ session: standalone FAQ article output contract can
  consume this provider's `source_material`.
- Parked hardening: none.

## Verification

- `py_compile` for `extracted_content_pipeline/support_ticket_input_provider.py`
  and `tests/test_extracted_support_ticket_input_provider.py` - passed.
- `pytest` for `tests/test_extracted_support_ticket_input_provider.py` and
  `tests/test_extracted_support_ticket_input_package.py` - 20 passed.
- `scripts/validate_extracted_content_pipeline.sh` - passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` for
  `extracted_content_pipeline` - passed.
- `scripts/audit_extracted_standalone.py` with `--fail-on-debt` - passed.
- `scripts/check_ascii_python.sh` - passed.
- `scripts/run_extracted_pipeline_checks.sh` - 1902 passed, 1 skipped.
- `scripts/local_pr_review.sh` with `--allow-dirty` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~80 |
| Provider module | ~125 |
| Tests | ~160 |
| CI enrollment | ~5 |
| **Total** | **~370** |
