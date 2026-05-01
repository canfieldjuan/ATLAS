CREATE TABLE IF NOT EXISTS b2b_enrichment_stage_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID NOT NULL REFERENCES b2b_reviews(id) ON DELETE CASCADE,
    stage_id TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    backend TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'planned',
    result_source TEXT NOT NULL DEFAULT 'generated',
    batch_id UUID REFERENCES anthropic_message_batches(id) ON DELETE SET NULL,
    batch_custom_id TEXT,
    run_id TEXT,
    usage_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_text TEXT,
    error_code TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE (review_id, stage_id, request_fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_b2b_enrichment_stage_runs_state_stage
    ON b2b_enrichment_stage_runs (state, stage_id, backend);

CREATE INDEX IF NOT EXISTS idx_b2b_enrichment_stage_runs_review_stage
    ON b2b_enrichment_stage_runs (review_id, stage_id, created_at DESC);
