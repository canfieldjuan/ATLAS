# Content Ops Reasoning Status Capabilities

## Why This Slice Exists

The Content Ops control-surface API exposes a host-provided reasoning status,
but it currently preserves only scalar fields. Hosts cannot report useful
capability lists such as available reasoning modes, pack names, or supported
knobs without those lists being stripped.

## Scope

- Allow bounded list/tuple fields containing scalar values in
  `extracted_content_pipeline/api/control_surfaces.py` reasoning status output.
- Keep nested objects and mixed complex values filtered out.
- Add focused API tests.
- Refresh control-surface docs and coordination state.

## Mechanism

- Add a small scalar-sequence sanitizer used only by reasoning status.
- Preserve lists such as `modes`, `packs`, and `capabilities` when each item is
  a string, number, or boolean.
- Drop nested mappings and object values exactly as before.

## Intentional

- No new route.
- No provider construction.
- No change to execution behavior.
- No requirement that hosts supply capability fields.

## Deferred

- Standardized capability taxonomy.
- Per-output reasoning capability matching.
- UI rendering for richer capability lists.

## Verification

- Focused control-surface API tests.
- Compile check for touched Python files.
- Local PR review gate.

### Files Touched

- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `plans/PR-Content-Ops-Reasoning-Status-Capabilities.md`
- `tests/test_extracted_content_control_surface_api.py`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| API sanitizer | ~25 |
| Tests | ~30 |
| Docs and coordination | ~20 |
| Plan doc | ~55 |
| **Total** | ~130 |
