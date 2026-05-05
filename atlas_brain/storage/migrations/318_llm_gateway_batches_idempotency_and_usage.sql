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

-- Surfaces batches that ended/canceled/expired but whose
-- ``usage_tracked`` flag is still FALSE. This catches the case
-- where the Anthropic results pre-fetch failed (network /
-- rate-limit / SDK timeout) so the atomic claim was never made.
-- The refresh path retries on the next /batch/{id} poll, but a
-- batch a customer never polls again could otherwise stay
-- pending forever -- a future cron worker can scan this index
-- and run _persist_batch_usage on stragglers.
--
-- NOTE: this index does NOT catch partial-write loss after the
-- claim. trace_llm_call exceptions during the persist phase are
-- logged but do not roll back the flag, because rolling back
-- would let the next retry double-count the items that already
-- landed. Per-item idempotency (unique key on llm_usage by
-- account_id+batch_id+custom_id) is the proper fix and is
-- planned as a follow-up; until then, partial-loss > double-
-- count is the deliberate trade-off.
CREATE INDEX IF NOT EXISTS idx_llm_gateway_batches_usage_pending
    ON llm_gateway_batches (status, updated_at DESC)
    WHERE usage_tracked = FALSE
      AND status IN ('ended', 'canceled', 'expired');
