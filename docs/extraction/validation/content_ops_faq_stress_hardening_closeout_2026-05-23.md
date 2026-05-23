# Content Ops FAQ Stress Hardening Closeout - 2026-05-23

This note closes the two root `HARDENING.md` FAQ stress items created by
`content_ops_faq_scale_stress_probe_2026-05-23.md`.

## FAQSTRESS-1

Original issue: large FAQ uploads passed deterministic and DB lifecycle smokes
through 50,000 rows, but that scale is not safe as a synchronous hosted request
without an async job boundary, explicit request limits, or backpressure.

Current hosted state:

- Inline `/content-ops/execute` source material is bounded by
  `ContentOpsRequestModel` and `_validate_input_shape`.
- Top-level `inputs.source_material` lists and recognized source-material
  bundle lists are capped at `_MAX_INGESTION_ROWS == 1000`.
- File upload ingestion is bounded separately by `_MAX_INGESTION_FILE_BYTES`
  and `_MAX_FILE_INGESTION_ROWS`, currently 25 MB and 10,000 normalized rows.
- The file upload response reports those limits so the UI and operators can see
  the synchronous hosted ceiling.

Closeout decision: resolved for the synchronous hosted path by explicit limits.
Supporting FAQ uploads above those caps would be a new product/runtime slice
with a background job boundary, not an unbounded request-response path.

## FAQSTRESS-2

Original issue: 100 concurrent Postgres-backed lifecycle smoke processes
produced 97 successes and 3 `TooManyConnectionsError` failures. The failed
processes also exited before writing their requested `--output-result` files.

Current hosted and smoke state:

- `scripts/smoke_content_ops_faq_lifecycle.py` now builds the lifecycle payload
  even when database pool creation fails and writes `--output-result` when
  requested.
- `/content-ops/execute` now uses `_ExecuteConcurrencyGate` and returns a
  machine-readable `429` response when the router-local admission limit is full.
- `ContentOpsControlSurfaceApiConfig.execute_max_concurrency` validates that the
  configured limit is positive.

Closeout decision: resolved for the current single-router hosted contract and
for lifecycle failure artifact visibility. A cross-process/global admission
controller remains a future deployment-topology decision if production runs
multiple workers against one database pool.

## Regression Coverage

Relevant existing tests:

- `tests/test_smoke_content_ops_faq_lifecycle.py::test_faq_lifecycle_smoke_writes_result_when_pool_creation_fails`
- `tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_rejects_source_material_over_1000_as_422`
- `tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_accepts_source_material_bundle_1000_rows`
- `tests/test_extracted_content_control_surface_api.py::test_ingestion_file_inspect_route_rejects_oversized_upload_bytes`
- `tests/test_extracted_content_control_surface_api.py::test_ingestion_file_inspect_route_rejects_more_than_file_row_cap`
- `tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_rejects_when_concurrency_gate_is_full`
- `tests/test_extracted_content_control_surface_api.py::test_content_ops_config_rejects_invalid_execute_concurrency`
