-- PR-D6g: LLM Gateway batch cache prefilter accounting.
--
-- Tracks how many items in a customer batch were satisfied from the
-- exact-text cache and never sent to Anthropic. Customer sees this
-- on GET /api/v1/llm/batch/{id}; the per-item zero-token llm_usage
-- rows still drive /api/v1/llm/usage cache_savings_usd.
--
-- Defaults to 0 -- back-compat with existing rows.

ALTER TABLE llm_gateway_batches
    ADD COLUMN IF NOT EXISTS cache_prefiltered_items INTEGER NOT NULL DEFAULT 0;
