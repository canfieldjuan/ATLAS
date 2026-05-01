-- Migration 252: Persist cache-aware LLM usage and call-level debug fields.

ALTER TABLE llm_usage
    ADD COLUMN IF NOT EXISTS billable_input_tokens INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS cached_tokens INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS cache_write_tokens INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS queue_time_ms INT,
    ADD COLUMN IF NOT EXISTS api_endpoint TEXT,
    ADD COLUMN IF NOT EXISTS provider_request_id TEXT;

UPDATE llm_usage
SET billable_input_tokens = input_tokens
WHERE billable_input_tokens = 0
  AND input_tokens > 0;

CREATE INDEX IF NOT EXISTS idx_llm_usage_operation_created
    ON llm_usage (operation_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_llm_usage_provider_request_id
    ON llm_usage (provider_request_id)
    WHERE provider_request_id IS NOT NULL;
