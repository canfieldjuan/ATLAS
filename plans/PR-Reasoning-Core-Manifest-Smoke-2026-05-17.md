# PR: Reasoning Core Manifest And Standalone Smoke

## Why this slice exists

`extracted_reasoning_core` has a real public API, ports, graph/state modules,
packs, and tests, but it is the only active extracted product without a
`manifest.json`. That keeps it out of the shared manifest validation path and
makes standalone readiness less explicit than the other extracted packages.

## Scope

Add manifest-backed ownership and a lightweight standalone smoke for
`extracted_reasoning_core`.

### Files touched

- `.github/workflows/extracted_umbrella_checks.yml`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/queue.md`
- `docs/extraction/coordination/state.md`
- `extracted_reasoning_core/api.py`
- `extracted_reasoning_core/manifest.json`
- `plans/PR-Reasoning-Core-Manifest-Smoke-2026-05-17.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_extracted_reasoning_core_standalone.py`
- `tests/test_extracted_reasoning_core_manifest.py`

## Mechanism

- Declare every current `extracted_reasoning_core` file as product-owned in a
  new manifest.
- Add a standalone smoke script that imports the manifest's Python modules and
  executes `run_reasoning` with a fake LLM port.
- Wire shared validation to run manifest sync, ASCII, import, forbidden Atlas
  reasoning import, and smoke checks for the reasoning core.
- Add tests proving the manifest is product-owned, all targets exist, Python
  targets do not import `atlas_brain.reasoning`, and the smoke works at the CLI
  boundary.
- Replace four non-ASCII arrows in an existing reasoning-core docstring so the
  new ASCII check can pass for the full product.

## Intentional

- Keep the manifest fully product-owned. There is no Atlas source mapping for
  core files because this package is already the canonical reasoning boundary.
- Keep the smoke offline and port-driven. It uses a fake LLM port and does not
  touch network, database, provider routing, or Atlas settings.
- Clean the stale #568 inflight row while claiming this new slice.

## Deferred

- No graph/state wrapper split in this PR.
- No new `EXTRACTED_REASONING_CORE_STANDALONE` runtime branching. The smoke
  sets the env var for consistency, but the core has no Atlas fallback path to
  switch away from.
- No new reasoning capability or output behavior.

## Verification

- Focused reasoning-core manifest tests pass.
- Standalone reasoning-core smoke passes.
- Shared manifest, ASCII, import, and forbidden Atlas reasoning checks pass.
- Local PR review passes.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Manifest and smoke script | ~150 |
| Tests | ~60 |
| Check wiring | ~15 |
| Coordination docs | ~10 |
| Plan doc | ~80 |
| Existing docstring ASCII cleanup | ~5 |
| **Total** | ~320 |
