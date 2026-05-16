# Content Ops File Reasoning Target-Mode Parity

## Why This Slice Exists

The Postgres reasoning context provider filters mode-specific rows by
`target_mode`, but the file-backed provider ignored `target_mode`. That could
let a host-provided JSON context for one asset type satisfy another asset type
when selectors overlap. The file-backed path is used by examples, smoke tests,
and lightweight installs, so it should match the durable provider contract.

## Scope

- Preserve the existing selector matching order.
- Add target-mode aware selection to the file-backed provider.
- Keep blank target-mode rows as global fallback contexts.
- Document the file-backed target-mode behavior.
- Add focused regression tests for mode-specific rows and fallback rows.

## Mechanism

The provider indexes each selector to one or more contexts, keyed by normalized
`target_mode`. Reads still walk candidate selectors in the existing order. For
each selector, an exact target-mode row wins; if none exists, a blank-mode row
can satisfy the request. When the caller passes a blank target mode, the first
indexed row is returned to preserve legacy broad lookup behavior.

## Intentional

- No changes to the Postgres provider.
- No changes to the reasoning provider Protocol.
- No migration or schema change.
- No changes to generated asset services.

## Deferred

- Rich conflict diagnostics when a JSON file contains multiple rows for the
  same selector and target mode.
- Settings-level defaulting for file-backed context paths.

## Verification

- Focused file-backed reasoning provider tests.
- Python compile check for provider and tests.
- Diff whitespace check.
- Local PR review script.

### Files Touched

- `extracted_content_pipeline/campaign_reasoning_data.py`
- `tests/test_extracted_campaign_reasoning_data.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/reasoning_handoff_contract.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-File-Reasoning-Target-Mode.md`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Provider index/selection | ~55 |
| Tests | ~110 |
| Docs and coordination | ~20 |
| Plan doc | ~70 |
| **Total** | ~255 |
