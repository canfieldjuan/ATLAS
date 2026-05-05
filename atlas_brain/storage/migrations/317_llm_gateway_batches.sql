-- LLM Gateway customer-facing batch tracking (PR-D4b).
--
-- Wraps Anthropic's Message Batches API so customers get the 50%
-- batch discount via /api/v1/llm/batch. atlas's existing
-- ``anthropic_message_batches`` table is for the B2B internal
-- pipeline (vendor_name / artifact_type / stage_id semantics);
-- customer batches stay in this dedicated table so the schemas
-- don't entangle.
--
-- Status values mirror Anthropic's batch states:
--   queued      -- atlas accepted; not yet sent to Anthropic
--   in_progress -- Anthropic accepted; processing
--   ended       -- all items completed (success or fail)
--   canceling   -- cancellation in flight
--   canceled    -- finalized canceled
--   expired     -- Anthropic 24h TTL hit before completion
--
-- Per-batch item results live as text inside ``results_jsonl`` once
-- the batch ends (Anthropic returns a JSONL file URL; we fetch and
-- store inline for v1). Customer reads via GET /api/v1/llm/batch/{id}.

CREATE TABLE IF NOT EXISTS llm_gateway_batches (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id         UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    provider           VARCHAR(32) NOT NULL,
    provider_batch_id  VARCHAR(128) UNIQUE,
    model              VARCHAR(256) NOT NULL,
    status             VARCHAR(32) NOT NULL DEFAULT 'queued',
    total_items        INTEGER NOT NULL DEFAULT 0,
    completed_items    INTEGER NOT NULL DEFAULT 0,
    failed_items       INTEGER NOT NULL DEFAULT 0,
    results_jsonl      TEXT,
    error_text         TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at       TIMESTAMPTZ,
    completed_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_llm_gateway_batches_account_created
    ON llm_gateway_batches (account_id, created_at DESC);

-- Used during status polling to find batches that aren't yet terminal.
CREATE INDEX IF NOT EXISTS idx_llm_gateway_batches_active
    ON llm_gateway_batches (status, updated_at DESC)
    WHERE status IN ('queued', 'in_progress', 'canceling');
