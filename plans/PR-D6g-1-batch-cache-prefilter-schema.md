# PR-D6g-1: Schema scaffolding for /batch cache prefilter accounting

## Why this slice exists

Item #6 in the post-D6b LLM Gateway follow-up queue: apply the
existing exact-cache to `/api/v1/llm/batch` so customer batches dedup
against cached single-call responses.

The full feature has two coupled halves:

1. **Schema scaffolding** -- column, dataclass field, API view field,
   plumbing through SELECT/INSERT statements. (THIS PR.)
2. **Prefilter behavior** -- the cache lookup loop, zero-token
   `llm_usage` row writes, 100%-hit short-circuit (skip Anthropic),
   partial-hit handling. (PR-D6g-2 follow-up.)

Splitting along the schema/behavior seam keeps each slice reviewable
and lands the persistence shape first so the behavior PR doesn't
need to coordinate schema + logic in one diff.

## Scope (this PR)

Schema only. No behavior change today -- `cache_prefiltered_items`
will always read as `0` until PR-D6g-2 wires the prefilter loop.

Touches:

1. **Migration `323_llm_gateway_batches_cache_prefiltered_items.sql`**
   -- adds `cache_prefiltered_items INTEGER NOT NULL DEFAULT 0` to
   `llm_gateway_batches`.
2. **`atlas_brain/services/llm_gateway_batch.py`**:
   - New field `cache_prefiltered_items: int = 0` on
     `CustomerBatchRecord`.
   - `_row_to_record` reads the new column with the same defensive
     `.get`-style pattern as `usage_tracked` /
     `anthropic_call_initiated_at`.
   - All 6 `SELECT id, account_id, ...` statements updated to
     include the new column. INSERT statements unchanged because
     the column has a default.
3. **`atlas_brain/api/llm_gateway.py`**:
   - New field `cache_prefiltered_items: int = 0` on `BatchView`.
   - `_batch_record_to_view` maps the field through.

## Why no behavior change

Wiring the prefilter loop into `submit_customer_batch` requires:

- Lazy imports of `build_request_envelope` /
  `lookup_cached_text` / `is_llm_gateway_exact_cache_enabled` from
  `services/b2b/llm_exact_cache`.
- A 100%-hit short-circuit that bypasses Anthropic entirely and
  inserts a row with `status='ended'`, `cache_prefiltered_items=N`.
- A partial-hit path that submits only misses to Anthropic but
  writes zero-token `llm_usage` rows for the hits and stores the
  prefilter count on the row.
- Coordination with the existing idempotency replay path so a
  retry doesn't double-write the cache-hit usage rows.
- Coordination with the resume / refresh path
  (`refresh_customer_batch_status`) so the per-batch
  `total_items` vs `completed_items` accounting stays consistent
  when Anthropic completes the misses.

That's its own slice. Landing the schema first means PR-D6g-2 is a
pure-behavior diff without schema entanglement.

## Intentional (looks wrong but is deliberate)

- **Field defaults to 0.** Back-compat with rows from before the
  migration; clients that ignore the field are unaffected.
- **All 6 SELECTs updated, not just one.** Whichever read path
  serves a `BatchView` needs the column populated; missing it on
  any one would silently default to 0 even when there should be
  hits.
- **No INSERT change.** The column has a default; explicit `0` on
  INSERT would just match the default. PR-D6g-2 will write the
  real count.
- **`_row_to_record` uses defensive `.get`-style** pattern so test
  mocks with shorter row dicts still work. Matches the precedent
  set by `usage_tracked` (PR-D4c) and `anthropic_call_initiated_at`
  (PR-D4e).
- **No test for the migration itself.** Migration testing is
  handled separately by the migration runner; this PR's tests
  pin the dataclass / Pydantic shape.

## Deferred (looks missing but is on purpose)

- **Prefilter loop, zero-token usage rows, 100%-hit short-circuit.**
  PR-D6g-2.
- **Partial-hit submission flow.** Likely PR-D6g-3 -- requires
  coordinated changes to `_persist_batch_usage` and the
  refresh path.
- **`Cache-Control: no-store` on /batch.** Symmetric with PR-D6f
  on /chat; small follow-up.

## Verification

- New regression tests in
  `tests/test_llm_gateway_batch_cache_prefilter_schema.py`:
  - Migration file exists and contains the ALTER TABLE.
  - `BatchView` Pydantic model declares the field with default 0.
  - `CustomerBatchRecord` dataclass declares the field with default 0.
  - All SELECT statements in `llm_gateway_batch.py` reference the
    column.
  - `_row_to_record` defensively reads the column.
- `python3 -m py_compile atlas_brain/api/llm_gateway.py atlas_brain/services/llm_gateway_batch.py`
  -> clean.
- `bash scripts/check_ascii_python.sh` -> passed.

## Conflict check

No file overlap with any open PR.

## Diff size

- Migration: 12 LOC (one ALTER TABLE).
- Source: ~30 LOC across 2 files (1 BatchView field + 1
  CustomerBatchRecord field + _row_to_record reader + 6 SELECT
  edits + 1 _batch_record_to_view mapping).
- Tests: ~80 LOC, 5 source-text and structure assertions.
- Plan doc: ~110 LOC.

Source-only ~30 LOC. Smallest scope that lands the schema
infrastructure for the prefilter feature.

## After this lands

PR-D6g-2 wires the prefilter behavior on top of this scaffolding.
Customers see `cache_prefiltered_items > 0` on `/batch/{id}` when
their batches dedup against the cache.
