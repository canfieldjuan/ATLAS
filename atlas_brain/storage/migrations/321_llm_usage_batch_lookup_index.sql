-- LLM Gateway batch lookup helper index (PR-D4d).
--
-- Companion to 319 + 320. Separate file because
-- ``CREATE INDEX CONCURRENTLY`` cannot share a transaction
-- with anything else -- one CONCURRENTLY per migration file is
-- the guaranteed-safe pattern (see migration 090).
--
-- Use case: /api/v1/llm/usage rollups that want to break out
-- batch traffic by submission (e.g., "what did I spend on
-- batch_id X?"). The composite (batch_id, created_at DESC)
-- ordering matches the typical "show me this batch's items
-- newest-first" query shape. Predicate-only (batch_id IS NOT
-- NULL) so the index doesn't bloat with the high-volume sync
-- /chat traffic.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_llm_usage_batch_id
    ON llm_usage (batch_id, created_at DESC)
    WHERE batch_id IS NOT NULL;
