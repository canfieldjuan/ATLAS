# PR-LLM-Usage-Schema-Cache-Telemetry

## Why this slice exists

`HARDENING.md` parks a Content Ops cost telemetry gap: live generation can
surface `generation_usage` with cache hits, but local `llm_usage` persistence
logged `_store_local failed for span=content_ops.llm.complete: column
"account_id" of relation "llm_usage" does not exist`. The canonical schema has
`account_id` after migration 313, but partially migrated local/host databases
with migrations 127/252/253 can still store token, cache, cost, and tenant
metadata if the insert does not hard-fail on the missing top-level column.
This is slightly over the 400-LOC soft cap because `tracing.py` is a synced
LLM-infrastructure file, so the source fix must ship with the extracted copy in
the same PR.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Production hardening

1. Keep the canonical llm_usage account_id insert for migrated databases.
2. Add a narrow retry path only for Postgres undefined-column failures where the
   missing column is llm_usage account_id.
3. Preserve cache metrics and metadata on the fallback insert so Content Ops
   usage summaries can still filter by metadata account_id.
4. Remove the parked hardening item this slice closes.

### Files touched

- `atlas_brain/services/tracing.py` - add account-column fallback for local usage writes.
- `extracted_llm_infrastructure/services/tracing.py` - synced extracted tracing copy.
- `tests/test_llm_tracing.py` - cover cache metrics persistence when the account column is missing.
- `tests/test_account_scoping.py` - keep account-scoped insert assertions pointed at the SQL builder.
- `HARDENING.md` - remove the closed parked telemetry item.
- `plans/PR-LLM-Usage-Schema-Cache-Telemetry.md` - plan for this slice.

## Mechanism

The tracing client still builds the full modern `llm_usage` insert first. If
Postgres rejects that statement because the `account_id` column is missing, the
client retries the same payload without the top-level `account_id` column. That
fallback keeps the original metadata JSON, including tenant/run/cache metadata,
and keeps `billable_input_tokens`, `cached_tokens`, `cache_write_tokens`,
endpoint, and provider request IDs in the persisted row.

## Intentional

- This does not replace migration 313 or weaken the canonical schema. Fully
  migrated production databases continue using the `account_id` column.
- The fallback is limited to the known undefined-column shape. Other write
  failures still log and do not retry, so real schema or data issues do not get
  silently hidden.

## Deferred

- A separate migration-readiness diagnostic could warn operators before live
  generation starts when local cost telemetry columns are missing, but this
  slice fixes the data-loss path for the known partial-schema case.
- Parked hardening: none

## Verification

- pytest tests/test_llm_tracing.py -q -> 5 passed, 1 warning.
- pytest tests/test_llm_tracing.py tests/test_tracing_context.py::test_store_local_uses_standalone_connection_when_shared_pool_disabled -q -> 6 passed, 1 warning.
- pytest tests/test_llm_tracing.py tests/test_account_scoping.py::test_tracing_store_local_insert_includes_account_id_column -q -> 7 passed, 1 warning.
- pytest tests/test_account_scoping.py tests/test_llm_gateway_router.py::test_cache_hit_insert_sql_does_not_route_through_tracer_drop_filter tests/test_llm_tracing.py -q -> 32 passed, 1 warning.
- python -m py_compile atlas_brain/services/tracing.py extracted_llm_infrastructure/services/tracing.py tests/test_llm_tracing.py tests/test_account_scoping.py -> passed.
- bash scripts/validate_extracted_llm_infrastructure.sh -> passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_llm_infrastructure -> passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt -> passed.
- bash scripts/check_ascii_python.sh -> passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-llm-usage-schema-cache-telemetry-body.md -> passed after review fix.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `atlas_brain/services/tracing.py` | ~165 |
| `extracted_llm_infrastructure/services/tracing.py` | ~165 |
| `tests/test_llm_tracing.py` | ~109 |
| `tests/test_account_scoping.py` | ~13 |
| `HARDENING.md` | ~9 |
| `plans/PR-LLM-Usage-Schema-Cache-Telemetry.md` | ~83 |
| Total | ~544 |
