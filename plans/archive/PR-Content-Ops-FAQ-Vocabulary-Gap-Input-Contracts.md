# PR-Content-Ops-FAQ-Vocabulary-Gap-Input-Contracts

## Why this slice exists

Hosted FAQ vocabulary-gap generation is now wired end to end, but the public
control-surface catalog still does not describe the two FAQ-specific inputs the
UI can send: `faq_documentation_terms` and `faq_vocabulary_gap_rules`.

That leaves the UI labels/placeholders hardcoded and creates avoidable drift
risk between backend request keys and frontend controls. This slice publishes
the FAQ input contracts through the existing control-surface catalog.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-input-contracts

1. Add backend FAQ input-contract metadata for documentation terms and
   vocabulary-gap rules.
2. Include those contracts in `GET /content-ops/control-surfaces`.
3. Pin the catalog response in backend API coverage.
4. Refresh the frontend catalog fixture so compile-time contract checks see the
   new fields.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Vocabulary-Gap-Input-Contracts.md` | Plan doc for this catalog contract slice. |
| `extracted_content_pipeline/ticket_faq_input_contract.py` | Defines FAQ input contract metadata. |
| `extracted_content_pipeline/api/control_surfaces.py` | Adds FAQ contracts to the static control-surface catalog payload. |
| `tests/test_extracted_content_control_surface_api.py` | Verifies the two FAQ input contracts in the catalog response. |
| `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json` | Refreshes the frontend wire fixture with the new catalog contracts. |

## Mechanism

The control-surface API already composes static input contracts from dedicated
contract modules:

```python
"input_contracts": {
    LANDING_PAGE_QUALITY_REPAIR_INPUT: ...,
    **landing_page_seo_geo_aeo_input_contracts(),
}
```

This slice adds a parallel FAQ contract module and merges its two entries into
that same static payload. No request parsing, generation, or UI behavior changes
in this PR.

## Intentional

- Backend/catalog contract only. The UI continues to use its current hardcoded
  FAQ labels/placeholders until the next slice consumes the catalog entries.
- No generator, dispatcher, execute result, persistence, or CLI changes.
- No new input type is introduced; the existing frontend contract type already
  accepts string labels such as `string_list` and `nested_string_list`.

## Deferred

- Updating the Content Ops form to read FAQ labels/placeholders from these
  catalog contracts remains the next UI slice.
- Rich per-line validation metadata for vocabulary-gap rules remains separate.
- Current `HARDENING.md` entries were scanned; they are landing-page repair
  items and do not touch this FAQ catalog lane.

## Verification

- `python -m pytest tests/test_extracted_content_control_surface_api.py::test_describe_control_surfaces_route_returns_catalog_and_presets` - passed, 1 test.
- `python -m py_compile extracted_content_pipeline/ticket_faq_input_contract.py extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_control_surface_api.py` - passed.
- `npm run build` from `atlas-intel-ui/` - passed; frontend contract fixtures compiled and Vite generated sitemap/prerender output.
- `git diff --check` - passed.
- `npm run lint` from `atlas-intel-ui/` - passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed, 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed, 1761 tests and 1 existing `torch`/`pynvml` warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| FAQ input contract module | ~45 |
| API catalog merge | ~5 |
| Backend test assertions | ~20 |
| Frontend catalog fixture | ~20 |
| **Total** | ~175 |
