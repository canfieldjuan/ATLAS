-- Local tracking for Anthropic Message Batch executions and per-item outcomes.

CREATE TABLE IF NOT EXISTS anthropic_message_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stage_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    run_id TEXT,
    provider_batch_id TEXT UNIQUE,
    status TEXT NOT NULL DEFAULT 'preparing',
    total_items INTEGER NOT NULL DEFAULT 0,
    submitted_items INTEGER NOT NULL DEFAULT 0,
    cache_prefiltered_items INTEGER NOT NULL DEFAULT 0,
    fallback_single_call_items INTEGER NOT NULL DEFAULT 0,
    completed_items INTEGER NOT NULL DEFAULT 0,
    failed_items INTEGER NOT NULL DEFAULT 0,
    estimated_sequential_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
    estimated_batch_cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
    provider_error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_anthropic_message_batches_stage_created
    ON anthropic_message_batches (stage_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_anthropic_message_batches_run_created
    ON anthropic_message_batches (run_id, created_at DESC)
    WHERE run_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS anthropic_message_batch_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL REFERENCES anthropic_message_batches(id) ON DELETE CASCADE,
    custom_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    vendor_name TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    cache_prefiltered BOOLEAN NOT NULL DEFAULT FALSE,
    fallback_single_call BOOLEAN NOT NULL DEFAULT FALSE,
    response_text TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    billable_input_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
    provider_request_id TEXT,
    error_text TEXT,
    request_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE (batch_id, custom_id)
);

CREATE INDEX IF NOT EXISTS idx_anthropic_message_batch_items_status
    ON anthropic_message_batch_items (batch_id, status);

CREATE INDEX IF NOT EXISTS idx_anthropic_message_batch_items_stage_created
    ON anthropic_message_batch_items (stage_id, created_at DESC);
