# PR: Content Ops Input Provider Contract

## Why this slice exists

Content Ops now has working generation paths for blog posts, landing pages, and
FAQ Markdown, but each path still depends on request callers knowing the raw
`ContentOpsRequest.inputs` keys. Ingestion layers need a stable handoff point so
they can provide real business inputs without creating a second content layer or
reimplementing generator-specific wiring.

This slice adds a thin, pure input-provider contract. It lets an upstream
ingestion layer return a normalized input package that can be merged into the
existing Content Ops control-surface payload. The generated assets still use the
current services: blog blueprints, landing-page `MarketingCampaign.context`, and
FAQ `source_material`.

The diff is slightly over the 400 LOC soft cap because the contract ships with
its plan and focused tests proving the package flows through the existing
control-surface and generation-plan path. Splitting those tests from the
contract would leave the first slice underverified.

## Scope (this PR)

Ownership lane: content-ops/input-provider-contract

1. Add a standalone input-provider module with:
   - an immutable `ContentOpsInputPackage`;
   - a `ContentOpsInputProvider` protocol;
   - a merge helper that combines provider inputs with an existing request
     payload.
2. Keep operator/request inputs authoritative when the same key appears in both
   places.
3. Prove the merged payload is accepted by the existing control-surface and
   generation-plan path.
4. Ignore `None` request override values so optional `null` payload fields do
   not erase provider defaults.
5. Enroll the new test file in extracted pipeline CI.
6. Document that FAQ generation remains owned by the existing FAQ path and this
   slice only supplies `source_material`.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Input-Provider-Contract.md` | Plan doc for this slice. |
| `extracted_content_pipeline/content_ops_input_provider.py` | New pure input-provider contract and merge helpers. |
| `tests/test_extracted_content_ops_input_provider.py` | Unit and integration-style coverage for request payload merging. |
| `scripts/run_extracted_pipeline_checks.sh` | Enroll the new test file in extracted pipeline CI. |
| `.github/workflows/extracted_pipeline_checks.yml` | Trigger extracted checks when the new test file changes. |

## Mechanism

The new module defines:

- `ContentOpsInputPackage`: the normalized output of an ingestion/provider layer.
- `ContentOpsInputProvider`: a protocol for async providers that can build a
  package for a tenant/request context.
- `content_ops_payload_from_input_package(...)`: builds a plain dict payload
  accepted by `request_from_mapping(...)`.
- `merge_content_ops_input_package(...)`: overlays provider inputs under an
  existing request payload while letting explicit request inputs win.

The helper does not inspect FAQ item internals, generate FAQ articles, seed blog
blueprints, or publish landing pages. It only produces the same request shape the
existing control surface already understands.

`None` values from the request payload are treated as unset. This preserves
provider defaults when a host serializes optional fields as `null`.

## Intentional

- No new API route in this slice. The contract is host-usable first; route
  integration can be added after we prove the shape.
- No FAQ implementation. FAQ remains owned by the existing FAQ generator lane.
- No blog-blueprint persistence. Blog generation already uses the blueprint
  repository; a future provider can persist blueprints before returning the
  request payload.
- No LLM calls. This is a control-surface contract, not generated-copy work.

## Deferred

- Future PR: `PR-Content-Ops-Input-Provider-API-Wiring` can let hosted routes
  call configured providers before preview/execute.
- Future PR: `PR-Content-Ops-Ticket-Input-Package` can map uploaded support
  tickets into provider inputs such as `source_material`, landing-page context,
  and blog topic/filters.
- Future PR owned by the FAQ session: standalone FAQ article output contract.
- Parked hardening: none.

## Verification

- `py_compile` for `extracted_content_pipeline/content_ops_input_provider.py`
  and `tests/test_extracted_content_ops_input_provider.py` -> passed.
- `pytest` for `tests/test_extracted_content_ops_input_provider.py`,
  `tests/test_extracted_content_control_surfaces.py`, and
  `tests/test_extracted_content_generation_plan.py` -> 76 passed.
- `scripts/validate_extracted_content_pipeline.sh` -> passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` for
  `extracted_content_pipeline` -> passed.
- `scripts/audit_extracted_standalone.py` with `--fail-on-debt` -> passed.
- `scripts/check_ascii_python.sh` -> passed.
- `scripts/run_extracted_pipeline_checks.sh` -> 1880 passed, 1 skipped.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~110 |
| Contract module | ~190 |
| Tests | ~165 |
| CI enrollment | ~3 |
| **Total** | **~470** |
