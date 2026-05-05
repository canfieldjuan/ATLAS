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

-- Copilot on PR-D4d: a NULL custom_id is treated as distinct in
-- the UNIQUE index, so without this CHECK two retries with NULL
-- custom_id under the same batch_id could both insert -- exactly
-- the double-write the index is meant to prevent. Empty string
-- gets the same treatment because Anthropic shouldn't return one
-- and we'd rather fail loud than silently group all blanks
-- together as one item. Postgres rejects empty CHECK names so
-- we use a DO-block guard for re-run safety.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'llm_usage_batch_requires_custom_id'
    ) THEN
        ALTER TABLE llm_usage
        ADD CONSTRAINT llm_usage_batch_requires_custom_id
        CHECK (
            batch_id IS NULL
            OR (custom_id IS NOT NULL AND custom_id <> '')
        );
    END IF;
END $$;

-- Indexes are created in companion migrations 320 and 321 with
-- ``CREATE INDEX CONCURRENTLY`` so the build doesn't take an
-- ACCESS EXCLUSIVE lock on llm_usage and block live inserts.
-- ``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction
-- block, so each one needs its own migration file (same pattern
-- as migration 090). Copilot on PR-D4d.
