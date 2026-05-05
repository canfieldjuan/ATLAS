-- LLM Gateway batch per-item idempotency (PR-D4d).
--
-- Closes the partial-write loss gap PR-D4c documented as deferred:
-- _persist_batch_usage previously routed per-item writes through
-- the fire-and-forget tracer dispatch, so a process crash between
-- "usage_tracked = TRUE" and the queued INSERTs landing would lose
-- billing data permanently. PR-D4d switches to a direct INSERT
-- with ``ON CONFLICT DO NOTHING`` keyed on (account_id, batch_id,
-- custom_id) so a retry can safely re-emit any subset of rows
-- without double-counting.
--
-- Two columns added to ``llm_usage``:
--   batch_id   -- references llm_gateway_batches(id) for batch
--                 traffic; NULL for sync /chat traffic. Lets the
--                 unique index scope only batch rows.
--   custom_id  -- echoes the customer-supplied per-item id from
--                 the original Anthropic batch request. NULL for
--                 sync traffic. Combined with batch_id this is
--                 the natural key for batch items.
--
-- Partial UNIQUE index: only batch rows (batch_id IS NOT NULL)
-- participate. Non-batch /chat traffic (batch_id NULL) is unaffected,
-- which preserves the high-volume insert path's perf and avoids
-- requiring a value to satisfy the index.

ALTER TABLE llm_usage
    ADD COLUMN IF NOT EXISTS batch_id  UUID,
    ADD COLUMN IF NOT EXISTS custom_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_usage_batch_item
    ON llm_usage (account_id, batch_id, custom_id)
    WHERE batch_id IS NOT NULL;

-- Lookup-by-batch helper (used by /api/v1/llm/usage rollups that
-- want to break out batch traffic by submission). Cheap to add and
-- gives the planner an option for the partial-aggregate query.
CREATE INDEX IF NOT EXISTS idx_llm_usage_batch_id
    ON llm_usage (batch_id, created_at DESC)
    WHERE batch_id IS NOT NULL;
