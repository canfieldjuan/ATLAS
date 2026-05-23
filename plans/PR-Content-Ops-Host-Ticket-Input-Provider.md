# PR: Content Ops Host Ticket Input Provider

## Why this slice exists

PR-Content-Ops-Support-Ticket-Input-Provider added the extracted provider
adapter, but the Atlas host still mounts the Content Ops control-surface router
without an input provider. That means uploaded/imported support-ticket material
can be sent as raw `inputs.source_material`, but the host does not yet expand it
into the FAQ Report defaults for landing pages, FAQ Markdown, and blog planning.

This slice wires a host-owned provider into the existing router mount. It stays
thin: no new route, no FAQ generator implementation, no file import changes, and
no database lookup.

The diff is slightly over the 400 LOC target because review feedback required
shipping the host bridge, CI enrollment, and route-safety regressions together;
splitting those would leave the globally mounted provider under-tested or too
broad for one PR cycle.

## Scope (this PR)

Ownership lane: content-ops/input-provider-ticket-package

Slice phase: Vertical slice

1. Add an Atlas host input-provider factory that reads support-ticket
   `inputs.source_material` from the request payload.
2. Return a no-op package when no source material is present so unrelated
   Content Ops previews keep their existing behavior.
3. Delegate package construction to `SupportTicketInputProvider` only when the
   source material is support-ticket-shaped.
4. Wire the provider into `create_content_ops_control_surface_router(...)` in
   the Atlas API mount.
5. Add host tests for no-op behavior, support-ticket expansion, and router mount
   wiring.
6. Enroll the host bridge module and tests in Extracted Pipeline CI.

### Files touched

- `atlas_brain/_content_ops_input_provider.py`
- `atlas_brain/api/__init__.py`
- `.github/workflows/extracted_pipeline_checks.yml`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_atlas_content_ops_input_provider.py`
- `plans/PR-Content-Ops-Host-Ticket-Input-Provider.md`

## Mechanism

`build_content_ops_input_provider()` returns an object implementing
`build_content_ops_input_package(scope, request)`. The provider checks
`request["inputs"]["source_material"]`:

- if absent/empty, or present but not support-ticket-shaped, it returns an empty
  `ContentOpsInputPackage` with empty target/ingestion defaults so the merge
  path does not alter ordinary requests;
- if present and support-ticket-shaped, it delegates to
  `SupportTicketInputProvider` so the package gets support-ticket
  `source_material`, FAQ defaults, landing-page context, and blog topic/filter
  inputs.
- Support-ticket shape detection accepts explicit ticket/case/conversation
  fields, known ticket source types, and unlabeled rows with the same
  subject/body-style fields accepted by the downstream package builder. Explicit
  non-ticket source types still no-op.

The existing API merge helper still keeps request/operator fields authoritative.

## Intentional

- No new public route. The existing `/content-ops/preview`, `/plan`, and
  `/execute` paths already call the configured input provider.
- No file parsing. File upload/import remains owned by the ingestion lane.
- No FAQ generation changes. FAQ article work remains with the FAQ session.
- The no-op branch is explicit because mounting a provider globally must not
  change unrelated Content Ops requests.
- Generic `source_material` remains generic. The host adapter only expands
  support-ticket-shaped bundles or rows.
- The full API aggregator test is skipped when `asyncpg` is not importable
  because the dependency-light extracted CI lane does not install the host
  database driver. The provider tests still run in that lane.

## Deferred

- Future PR: persisted import rows can back the source loader once the ingestion
  owner exposes that lookup contract.
- Future PR owned by the FAQ session: standalone FAQ article output contract can
  consume the same `source_material`.
- Parked hardening: none.

## Verification

- `py_compile` for `atlas_brain/_content_ops_input_provider.py` and
  `atlas_brain/api/__init__.py` and
  `tests/test_atlas_content_ops_input_provider.py` - passed.
- `pytest` for `tests/test_atlas_content_ops_input_provider.py`,
  `tests/test_extracted_support_ticket_input_provider.py`, and
  `tests/test_extracted_support_ticket_input_package.py` - 27 passed.
- `scripts/local_pr_review.sh` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~110 |
| Host provider | ~175 |
| API mount | ~5 |
| CI enrollment | ~6 |
| Tests | ~250 |
| **Total** | **~550** |
