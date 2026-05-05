-- LLM Gateway batch idempotency + usage tracking (PR-D4c).
--
-- Adds two follow-up columns deferred from PR-D4b:
--
-- 1. idempotency_key: customer-supplied dedup token. POST /api/v1/llm/batch
--    accepts an ``Idempotency-Key`` header; we store it on the row and
--    refuse duplicate submits within the same account. Closes the
--    accepted-upstream-but-timeout-locally retry case where a customer
--    would otherwise create duplicate paid batches.
--
--    Format note: customers typically send a UUID here, but any 1-128
--    char string is accepted. The constraint is per-account so two
--    accounts can independently use the same value.
--
-- 2. usage_tracked: marks a batch whose per-item usage has been
--    written to ``llm_usage`` already. Set true atomically with the
--    transition to a terminal state (ended/canceled/expired) so
--    repeat polls don't double-count tokens against the customer's
--    /api/v1/llm/usage rollup.

ALTER TABLE llm_gateway_batches
    ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(128),
    ADD COLUMN IF NOT EXISTS usage_tracked   BOOLEAN NOT NULL DEFAULT FALSE;

-- Per-account idempotency: account A and B can independently use the
-- same idempotency_key without collision. NULL keys (the optional
-- "no header sent" case) skip the constraint.
CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_gateway_batches_idempotency
    ON llm_gateway_batches (account_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- Used by the poll-completion path to find batches that are
-- terminal but whose usage hasn't been written yet -- rare in
-- practice (we write inline on the transition) but covers a
-- case where atlas crashed mid-write.
CREATE INDEX IF NOT EXISTS idx_llm_gateway_batches_usage_pending
    ON llm_gateway_batches (status, updated_at DESC)
    WHERE usage_tracked = FALSE
      AND status IN ('ended', 'canceled', 'expired');
