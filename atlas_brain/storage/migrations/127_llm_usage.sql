-- LLM usage tracking for cost monitoring.
-- Stores a row per LLM call so we can aggregate costs locally
-- without depending on FTL API for queries.

CREATE TABLE IF NOT EXISTS llm_usage (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    span_name       VARCHAR(256) NOT NULL,
    operation_type  VARCHAR(64)  NOT NULL DEFAULT 'llm_call',
    model_name      VARCHAR(256),
    model_provider  VARCHAR(64),
    input_tokens    INT NOT NULL DEFAULT 0,
    output_tokens   INT NOT NULL DEFAULT 0,
    total_tokens    INT NOT NULL DEFAULT 0,
    cost_usd        DECIMAL(10, 6) NOT NULL DEFAULT 0,
    duration_ms     INT NOT NULL DEFAULT 0,
    ttft_ms         INT,
    inference_time_ms INT,
    tokens_per_second REAL,
    status          VARCHAR(32) NOT NULL DEFAULT 'completed',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Primary query pattern: cost by day/provider
CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_usage_provider ON llm_usage (model_provider, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_usage_span ON llm_usage (span_name, created_at DESC);
